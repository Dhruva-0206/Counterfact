"""Every guardrail rule, one test each, plus fallback ordering and the audit check log."""

from __future__ import annotations

import numpy as np
import pytest

from counterfact.config import (
    ESCALATE_HUMAN,
    NO_ACTION,
    REMIND_AND_RETRY,
    RETRY_DELAYED,
    RETRY_NOW,
)
from counterfact.policy.guardrails import (
    apply_guardrails,
    guardrailed_actions,
    in_quiet_hours,
    next_send_time,
    remaining_retry_budget,
)
from counterfact.sim.schema import Merchant

MERCHANT = Merchant(
    merchant_id="m_test", name="Test", segment="b2c", plan_amount=999,
    min_ev_threshold=10.0, escalation_amount_threshold=5_000, ops_cost=150,
)


def row(**over) -> dict:
    base = {
        "merchant_id": "m_test", "amount": 999.0, "failure_category": "insufficient_funds",
        "attempt_number": 1, "contacts_last_24h": 0, "contacts_last_7d": 0,
        "hour_ist": 12, "failed_at": "2025-03-10T12:00:00",
    }
    base.update(over)
    return base


def ev(**vals) -> np.ndarray:
    v = np.zeros(5)
    for arm, x in vals.items():
        v[{"retry_now": RETRY_NOW, "retry_delayed": RETRY_DELAYED, "remind": REMIND_AND_RETRY,
           "escalate": ESCALATE_HUMAN}[arm]] = x
    return v


def codes(d) -> list[str]:
    return [c.code for c in d.checks if not c.passed]


def test_kill_switch_forces_no_action() -> None:
    m = MERCHANT.model_copy(update={"kill_switch": True})
    d = apply_guardrails(row(), m, ev(retry_now=500), RETRY_NOW, 1)
    assert d.arm == NO_ACTION and d.overridden and codes(d) == ["MERCHANT_KILL_SWITCH"]


def test_mandatory_escalation_amount() -> None:
    d = apply_guardrails(row(amount=6_000), MERCHANT, ev(retry_now=900), RETRY_NOW, 1)
    assert d.arm == ESCALATE_HUMAN and "MANDATORY_ESCALATION_AMOUNT" in codes(d)


def test_mandatory_escalation_risk_declined() -> None:
    d = apply_guardrails(row(failure_category="risk_declined"), MERCHANT, ev(retry_now=900), RETRY_NOW, 1)
    assert d.arm == ESCALATE_HUMAN and "MANDATORY_ESCALATION_RISK" in codes(d)


def test_retry_budget_exhausted_falls_back_to_next_best() -> None:
    d = apply_guardrails(row(attempt_number=4), MERCHANT, ev(retry_now=300, escalate=200), RETRY_NOW, 1)
    assert d.arm == ESCALATE_HUMAN and "RETRY_BUDGET_EXHAUSTED" in codes(d)


def test_retry_budget_truncated_is_annotated_not_rejected() -> None:
    d = apply_guardrails(row(attempt_number=3), MERCHANT, ev(retry_now=300), RETRY_NOW, 1)
    assert d.arm == RETRY_NOW and d.effective_retries == 1
    assert any(c.code == "RETRY_BUDGET_TRUNCATED" and c.passed for c in d.checks)
    assert remaining_retry_budget(1) == 3 and remaining_retry_budget(3) == 1 and remaining_retry_budget(9) == 0


def test_never_retry_card_expired() -> None:
    d = apply_guardrails(row(failure_category="card_expired"), MERCHANT, ev(retry_delayed=400, escalate=100), RETRY_DELAYED, 3)
    assert d.arm == ESCALATE_HUMAN and "CARD_EXPIRED_NO_RETRY" in codes(d)


def test_contact_cap_24h() -> None:
    d = apply_guardrails(row(contacts_last_24h=1), MERCHANT, ev(remind=300, retry_delayed=200), REMIND_AND_RETRY, 1)
    assert d.arm == RETRY_DELAYED and "CONTACT_CAP_24H" in codes(d)


def test_contact_cap_7d() -> None:
    d = apply_guardrails(row(contacts_last_7d=3), MERCHANT, ev(remind=300, retry_delayed=200), REMIND_AND_RETRY, 1)
    assert d.arm == RETRY_DELAYED and "CONTACT_CAP_7D" in codes(d)


def test_quiet_hours_defers_message_but_keeps_arm() -> None:
    d = apply_guardrails(row(hour_ist=23, failed_at="2025-03-10T23:15:00"), MERCHANT, ev(remind=300), REMIND_AND_RETRY, 1)
    assert d.arm == REMIND_AND_RETRY and d.message_send_at == "2025-03-11T08:00:00"
    assert any(c.code == "QUIET_HOURS_DEFERRED" for c in d.checks)
    assert in_quiet_hours(21) and in_quiet_hours(3) and not in_quiet_hours(8) and not in_quiet_hours(20)
    from datetime import datetime
    assert next_send_time(datetime(2025, 1, 1, 2, 30)) == datetime(2025, 1, 1, 8, 0)
    assert next_send_time(datetime(2025, 1, 1, 14, 0)) == datetime(2025, 1, 1, 14, 0)


def test_below_min_ev_falls_to_no_action() -> None:
    d = apply_guardrails(row(), MERCHANT, ev(retry_now=5.0), RETRY_NOW, 1)
    assert d.arm == NO_ACTION and "BELOW_MIN_EV" in codes(d)


def test_delay_parameter_survives_for_retry_delayed_only() -> None:
    d = apply_guardrails(row(), MERCHANT, ev(retry_delayed=300), RETRY_DELAYED, 7)
    assert d.arm == RETRY_DELAYED and d.delay_days == 7 and d.action_name == "retry_delayed_7"
    d2 = apply_guardrails(row(), MERCHANT, ev(retry_now=300), RETRY_NOW, 7)
    assert d2.delay_days == 0
    # fallback from a rejected reminder to retry_delayed keeps the model's preferred delay
    d3 = apply_guardrails(row(contacts_last_24h=1), MERCHANT, ev(remind=300, retry_delayed=200), REMIND_AND_RETRY, 7)
    assert d3.arm == RETRY_DELAYED and d3.delay_days == 7 and d3.action_name == "retry_delayed_7"
    with pytest.raises(ValueError):
        apply_guardrails(row(), MERCHANT, ev(retry_now=300), RETRY_NOW, 0)


def test_no_action_is_always_allowed_and_check_log_is_complete() -> None:
    d = apply_guardrails(row(), MERCHANT, np.zeros(5), NO_ACTION, 1)
    assert d.arm == NO_ACTION and not d.overridden
    assert {c.rule for c in d.checks} >= {"kill_switch", "mandatory_escalation"}
    assert all(isinstance(c.code, str) and c.code for c in d.checks)


@pytest.mark.parametrize("hour", [21, 22, 23, 0, 4, 7])
def test_quiet_hours_window(hour: int) -> None:
    assert in_quiet_hours(hour)


def test_guardrailed_actions_for_rule_tables() -> None:
    import pandas as pd
    df = pd.DataFrame([
        row(failure_category="card_expired"),           # retry proposed -> escalation fallback
        row(contacts_last_24h=1),                        # reminder proposed -> retry_delayed_1
        row(amount=9_000),                               # mandatory escalation
        row(),                                           # allowed as proposed
    ])
    arms = np.array([RETRY_DELAYED, REMIND_AND_RETRY, RETRY_NOW, RETRY_DELAYED])
    delays = np.array([3, 0, 0, 7])
    out = guardrailed_actions(df, {"m_test": MERCHANT}, arms, delays)
    assert out.tolist() == ["escalate_human", "retry_delayed_1", "escalate_human", "retry_delayed_7"]
