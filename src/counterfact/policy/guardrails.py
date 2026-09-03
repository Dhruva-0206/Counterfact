"""Hard guardrails. Code, not prompts. Every rejection carries a machine-readable reason code.

Rules (all enforced before any executor is called):

* ``MERCHANT_KILL_SWITCH``          merchant paused automation -> ``no_action``
* ``MANDATORY_ESCALATION_AMOUNT``   amount above merchant threshold -> ``escalate_human``
* ``MANDATORY_ESCALATION_RISK``     ``risk_declined`` -> ``escalate_human``
* ``RETRY_BUDGET_EXHAUSTED``        no retry attempts left (max 3 per failure) -> retry arms rejected
* ``RETRY_BUDGET_TRUNCATED``        fewer than 3 attempts left -> schedule truncated (annotation)
* ``CARD_EXPIRED_NO_RETRY``         never retry an expired card -> retry arms rejected
* ``CONTACT_CAP_24H`` / ``CONTACT_CAP_7D``  max 1 contact per 24h, 3 per 7d -> reminder rejected
* ``QUIET_HOURS_DEFERRED``          21:00-08:00 IST -> message deferred to 08:00 (annotation)
* ``BELOW_MIN_EV``                  chosen arm's net EV under merchant threshold -> ``no_action``

The policy proposes a ranking of arms by net EV; the guardrails walk that ranking and return the
first arm that passes, recording every check (passed or not) for the audit trail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from counterfact.config import (
    ARMS,
    ESCALATE_HUMAN,
    MAX_CONTACTS_7D,
    MAX_CONTACTS_24H,
    MAX_RETRIES_PER_FAILURE,
    NO_ACTION,
    QUIET_HOURS_END,
    QUIET_HOURS_START,
    REMIND_AND_RETRY,
    RETRY_DELAYED,
    RETRY_DELAYS,
    RETRY_NOW,
    primitive_name,
)
from counterfact.sim.schema import Merchant

RETRY_ARMS = (RETRY_NOW, RETRY_DELAYED, REMIND_AND_RETRY)


@dataclass
class Check:
    rule: str
    passed: bool
    code: str
    detail: str = ""


@dataclass
class GuardrailDecision:
    """Final, executable decision plus the full check log."""

    arm: int
    delay_days: int
    proposed_arm: int
    overridden: bool
    effective_retries: int
    message_send_at: str | None
    checks: list[Check] = field(default_factory=list)

    @property
    def action_name(self) -> str:
        return primitive_name(self.arm, self.delay_days)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action_name"] = self.action_name
        return d


def in_quiet_hours(hour_ist: int) -> bool:
    """True between 21:00 and 08:00 IST (start inclusive, end exclusive)."""
    return hour_ist >= QUIET_HOURS_START or hour_ist < QUIET_HOURS_END


def next_send_time(failed_at: datetime) -> datetime:
    """Next 08:00 IST at or after ``failed_at`` (naive timestamps are treated as IST)."""
    day = failed_at.replace(hour=QUIET_HOURS_END, minute=0, second=0, microsecond=0)
    if failed_at.hour >= QUIET_HOURS_START:
        day += timedelta(days=1)
    elif failed_at.hour >= QUIET_HOURS_END:
        return failed_at
    return day


def remaining_retry_budget(attempt_number: int) -> int:
    """Attempts still allowed for this failure: 3 minus prior attempts."""
    return max(0, MAX_RETRIES_PER_FAILURE - (int(attempt_number) - 1))


def apply_guardrails(
    row: pd.Series | dict,
    merchant: Merchant,
    net_ev: np.ndarray,
    proposed_arm: int,
    retry_delay: int,
) -> GuardrailDecision:
    """Apply every rule to one event.

    ``net_ev`` is the (5,) vector used to pick fallbacks; ``retry_delay`` is the delay to execute
    if ``retry_delayed`` ends up chosen (proposed or as a fallback).
    """
    r = dict(row)
    if int(retry_delay) not in RETRY_DELAYS:
        raise ValueError(f"retry_delay must be one of {RETRY_DELAYS}, got {retry_delay}")
    checks: list[Check] = []
    amount = float(r["amount"])
    category = str(r["failure_category"])
    attempt = int(r["attempt_number"])
    c24, c7 = int(r["contacts_last_24h"]), int(r["contacts_last_7d"])
    hour = int(r["hour_ist"])
    budget = remaining_retry_budget(attempt)

    def finish(arm: int, delay: int) -> GuardrailDecision:
        eff = min(budget, MAX_RETRIES_PER_FAILURE) if arm in RETRY_ARMS else 0
        if arm in RETRY_ARMS and eff < MAX_RETRIES_PER_FAILURE:
            checks.append(Check("max_retries", True, "RETRY_BUDGET_TRUNCATED",
                                f"{eff} of {MAX_RETRIES_PER_FAILURE} attempts remain (attempt {attempt})"))
        send_at = None
        if arm == REMIND_AND_RETRY:
            if in_quiet_hours(hour):
                failed_at = pd.Timestamp(r["failed_at"]).to_pydatetime()
                send_at = next_send_time(failed_at).isoformat()
                checks.append(Check("quiet_hours", True, "QUIET_HOURS_DEFERRED", f"message deferred to {send_at}"))
            else:
                checks.append(Check("quiet_hours", True, "OK", f"hour {hour} outside quiet hours"))
        return GuardrailDecision(
            arm=int(arm), delay_days=int(delay) if arm == RETRY_DELAYED else 0,
            proposed_arm=int(proposed_arm), overridden=int(arm) != int(proposed_arm),
            effective_retries=eff, message_send_at=send_at, checks=checks,
        )

    # 1. kill switch
    if merchant.kill_switch:
        checks.append(Check("kill_switch", False, "MERCHANT_KILL_SWITCH", "merchant paused automation"))
        return finish(NO_ACTION, 0)
    checks.append(Check("kill_switch", True, "OK"))

    # 2. mandatory escalation
    if amount > merchant.escalation_amount_threshold:
        checks.append(Check("mandatory_escalation", False, "MANDATORY_ESCALATION_AMOUNT",
                            f"amount {amount:.0f} > threshold {merchant.escalation_amount_threshold:.0f}"))
        return finish(ESCALATE_HUMAN, 0)
    if category == "risk_declined":
        checks.append(Check("mandatory_escalation", False, "MANDATORY_ESCALATION_RISK", "risk_declined requires human review"))
        return finish(ESCALATE_HUMAN, 0)
    checks.append(Check("mandatory_escalation", True, "OK"))

    # 3. walk arms by descending net EV, proposed arm first
    order = [int(proposed_arm)] + [int(a) for a in np.argsort(-net_ev) if int(a) != int(proposed_arm)]
    for arm in order:
        if arm == NO_ACTION:
            return finish(NO_ACTION, 0)
        name = ARMS[arm]
        if arm in RETRY_ARMS and budget <= 0:
            checks.append(Check("max_retries", False, "RETRY_BUDGET_EXHAUSTED", f"{name}: attempt {attempt} has no retries left"))
            continue
        if arm in RETRY_ARMS and category == "card_expired":
            checks.append(Check("card_expired", False, "CARD_EXPIRED_NO_RETRY", f"{name}: never retry an expired card"))
            continue
        if arm == REMIND_AND_RETRY and c24 >= MAX_CONTACTS_24H:
            checks.append(Check("contact_frequency", False, "CONTACT_CAP_24H", f"{c24} contact(s) in last 24h"))
            continue
        if arm == REMIND_AND_RETRY and c7 >= MAX_CONTACTS_7D:
            checks.append(Check("contact_frequency", False, "CONTACT_CAP_7D", f"{c7} contacts in last 7d"))
            continue
        if float(net_ev[arm]) < merchant.min_ev_threshold:
            checks.append(Check("min_ev", False, "BELOW_MIN_EV", f"{name}: net EV {net_ev[arm]:.1f} < {merchant.min_ev_threshold:.1f}"))
            continue
        if arm in RETRY_ARMS:
            checks.append(Check("max_retries", True, "OK", f"{budget} attempts available"))
            checks.append(Check("card_expired", True, "OK"))
        if arm == REMIND_AND_RETRY:
            checks.append(Check("contact_frequency", True, "OK", f"{c24}/24h, {c7}/7d"))
        checks.append(Check("min_ev", True, "OK", f"{name}: net EV {net_ev[arm]:.1f}"))
        return finish(arm, retry_delay if arm == RETRY_DELAYED else 0)
    return finish(NO_ACTION, 0)


FALLBACK_PREFERENCE = (RETRY_DELAYED, ESCALATE_HUMAN, NO_ACTION)
"""Fallback order for policies that do not produce net-EV vectors (rule tables)."""


def guardrailed_actions(
    df: pd.DataFrame,
    merchants: dict[str, Merchant],
    arms: np.ndarray,
    delays: np.ndarray,
    fallback_delay: int = 1,
) -> np.ndarray:
    """Run a rule-table policy through the same guardrails as the ML policy.

    A synthetic preference vector stands in for net EV: the proposed arm first, then the
    Razorpay-default schedule (``retry_delayed(1)``), then escalation, then nothing. Values are
    large enough to clear every merchant's minimum-EV threshold so that only the hard rules bite.
    """
    pref = np.zeros((len(df), len(ARMS)))
    for rank, arm in enumerate(FALLBACK_PREFERENCE):
        pref[:, arm] = 1e6 - 10 * (rank + 1)
    pref[np.arange(len(df)), arms.astype(int)] = 1e6
    retry_delay = np.where(arms.astype(int) == RETRY_DELAYED, delays.astype(int), fallback_delay)
    out, _ = apply_guardrails_frame(df, merchants, pref, arms.astype(int), retry_delay)
    return out["action_name"].to_numpy(dtype=object)


def apply_guardrails_frame(
    df: pd.DataFrame,
    merchants: dict[str, Merchant],
    net_ev: np.ndarray,
    proposed_arm: np.ndarray,
    retry_delay: np.ndarray,
) -> tuple[pd.DataFrame, list[GuardrailDecision]]:
    """Row-wise guardrails over a frame; returns a compact frame plus the full decisions."""
    decisions: list[GuardrailDecision] = []
    records = df.to_dict("records")
    for i, rec in enumerate(records):
        d = apply_guardrails(rec, merchants[rec["merchant_id"]], net_ev[i], int(proposed_arm[i]), int(retry_delay[i]))
        decisions.append(d)
    out = pd.DataFrame(
        {
            "arm": [d.arm for d in decisions],
            "delay_days": [d.delay_days for d in decisions],
            "action_name": [d.action_name for d in decisions],
            "proposed_arm": [d.proposed_arm for d in decisions],
            "overridden": [d.overridden for d in decisions],
            "effective_retries": [d.effective_retries for d in decisions],
            "message_send_at": [d.message_send_at for d in decisions],
            "rejection_codes": ["|".join(c.code for c in d.checks if not c.passed) for d in decisions],
        },
        index=df.index,
    )
    return out, decisions
