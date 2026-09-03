"""The TRUE recovery process. Three variants. Hidden from the models by construction.

Everything the agent can see lives in ``obs``; everything here additionally uses ``hidden``:
latent liquidity / engagement / churn intent, a hidden bank-outage duration, a hidden payday
segment, and the common random numbers (CRN) that realise coherent counterfactual outcomes.

A *plan* (see :class:`counterfact.sim.schema.Plan`) is a bounded set of retries, at most one
reminder message, and an optional escalation. The realised outcome for a plan is a
deterministic function of (obs, hidden, plan), so any policy can be evaluated exactly on the
counterfactual table without re-simulating.

Variants
--------
``calibrated``    Razorpay's default T+1/T+2/T+3 schedule lands in the public 55-65% band.
``misspecified``  Same skeleton, different hidden structure: heterogeneous message effects that
                  backfire on disengaged customers, a hidden 40% segment paid on the 7th, bank
                  outages clustered by (bank, day), threshold fatigue, an amount cliff for B2C,
                  steeper attempt decay, and +/-20% perturbed category parameters.
``null_uplift``   No arm has any causal effect on recovery (retries never succeed beyond
                  self-serve, messages do not lift, escalation resolves nothing). Messages can
                  still cause churn. The correct policy is to abstain.
``drifted``       Calibrated everywhere except two merchants whose failures depart from the
                  taxonomy (see ``DRIFT``): FitPulse bank errors behave like insufficient funds
                  and its customers churn when reminded about insufficient funds; ScaleOps
                  insufficient-funds failures only recover on the day-7 invoice cycle and its
                  failed mandates need a human. Merchant id is a feature, so a learner can pick
                  this up; a category rule table cannot.

This module must never be imported by ``counterfact.features`` or ``counterfact.models``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from counterfact.config import PRIMITIVE_ACTIONS, SimVariant, primitive_name
from counterfact.sim.schema import (
    MAX_RETRIES,
    RAZORPAY_DEFAULT_PLAN,
    RAZORPAY_T123_PLAN,
    Plan,
    plan_for,
)

RETRY_T = np.array([0.0, 1.0, 2.0, 3.0, 7.0])
P_MAX = 0.98
MONEY_CATEGORIES = ("insufficient_funds", "mandate_failed", "auth_failed")
RETRY_FIXABLE = ("insufficient_funds", "bank_technical", "gateway_5xx", "auth_failed", "mandate_failed")
"""Categories where a retry is the natural fix; their escalation resolve rate is a sensitivity axis."""

SENSITIVITY_KEYS = (
    "esc_level",        # escalation resolve rate for RETRY_FIXABLE categories (absolute)
    "retry_scale_mult",  # multiplier on the calibrated retry scale
    "self_mult",        # multiplier on self-serve probability
    "msg_lift_mult",    # multiplier on (reminder lift - 1)
    "churn_mult",       # multiplier on reminder-induced churn hazard
    "decay",            # per-attempt decay (absolute)
    "payday_mult",      # multiplier on (payday boost - 1)
)


@dataclass(frozen=True)
class CategoryParams:
    """Per-failure-category ingredients of the true process."""

    hard: float  # P(retries cannot succeed at all)
    self0: float  # base P(customer self-serves within the window)
    retry: tuple[float, float, float, float, float]  # base P(retry succeeds) at t = 0,1,2,3,7
    lift_self: float  # multiplicative lift on self-serve if the reminder is seen
    lift_retry: float  # multiplicative lift on retry success if the reminder is seen
    churn0: float  # base P(reminder triggers cancellation)
    esc0: float  # base P(human escalation resolves)
    payday_boost: float = 1.0  # retry multiplier once the retry lands on/after payday


CALIBRATED: dict[str, CategoryParams] = {
    "insufficient_funds": CategoryParams(
        hard=0.10, self0=0.22, retry=(0.10, 0.24, 0.27, 0.30, 0.36),
        lift_self=1.4, lift_retry=1.5, churn0=0.02, esc0=0.35, payday_boost=1.8,
    ),
    "bank_technical": CategoryParams(  # retry curve replaced by the hidden outage clock
        hard=0.12, self0=0.15, retry=(0.0, 0.0, 0.0, 0.0, 0.0),
        lift_self=1.05, lift_retry=1.05, churn0=0.01, esc0=0.40,
    ),
    "gateway_5xx": CategoryParams(
        hard=0.02, self0=0.12, retry=(0.85, 0.80, 0.78, 0.75, 0.70),
        lift_self=1.0, lift_retry=1.0, churn0=0.01, esc0=0.30,
    ),
    "auth_failed": CategoryParams(
        hard=0.20, self0=0.28, retry=(0.12, 0.18, 0.18, 0.18, 0.16),
        lift_self=1.8, lift_retry=1.6, churn0=0.02, esc0=0.40,
    ),
    "card_expired": CategoryParams(  # 3% of "expired" cards are refreshed by issuer updaters
        hard=0.97, self0=0.18, retry=(0.50, 0.50, 0.50, 0.50, 0.50),
        lift_self=1.9, lift_retry=1.0, churn0=0.015, esc0=0.60,
    ),
    "risk_declined": CategoryParams(
        hard=0.75, self0=0.12, retry=(0.15, 0.15, 0.15, 0.15, 0.15),
        lift_self=1.0, lift_retry=1.0, churn0=0.01, esc0=0.45,
    ),
    "customer_cancelled": CategoryParams(
        hard=0.92, self0=0.07, retry=(0.30, 0.30, 0.30, 0.30, 0.30),
        lift_self=1.3, lift_retry=1.0, churn0=0.15, esc0=0.20,
    ),
    "mandate_failed": CategoryParams(  # pre-debit notification needs ~24h: retry_now fails
        hard=0.35, self0=0.18, retry=(0.05, 0.35, 0.35, 0.35, 0.32),
        lift_self=1.3, lift_retry=1.3, churn0=0.015, esc0=0.45, payday_boost=1.3,
    ),
}

# Global multiplier on retry success, tuned by `calibrate_retry_scale` so that the Razorpay
# default schedule recovers ~60% under `calibrated`. See docs/EVALUATION.md.
RETRY_SCALE: dict[str, float] = {"calibrated": 1.3727, "misspecified": 1.3727, "null_uplift": 1.3727, "drifted": 1.3727}
ATTEMPT_DECAY: dict[str, float] = {"calibrated": 0.85, "misspecified": 0.70, "null_uplift": 0.85, "drifted": 0.85}

# Merchant-specific departures from the taxonomy for the `drifted` variant (ADR-014).
DRIFT: dict[tuple[str, str], dict[str, object]] = {
    # bank errors at FitPulse are really liquidity problems: IF dynamics, no outage clock
    ("m_fitpulse", "bank_technical"): {"like": "insufficient_funds"},
    # FitPulse customers resent reminders about money: reminder backfires and drives churn
    ("m_fitpulse", "insufficient_funds"): {"lift_self": 0.8, "lift_retry": 0.7, "churn_mult": 4.0},
    # ScaleOps pays invoices on a weekly AP run: early retries fail, day 7 works, payday irrelevant
    ("m_scaleops", "insufficient_funds"): {"retry": (0.05, 0.05, 0.05, 0.08, 0.60), "payday_boost": 1.0},
    # ScaleOps mandates need the AP team to re-approve: retries fail, a human fixes it
    ("m_scaleops", "mandate_failed"): {"hard": 0.90, "esc0": 0.70, "self0": 0.10},
}
BANK_RETRY_OK, BANK_RETRY_DOWN = 0.75, 0.06
ESCALATION_RETRY_DAY = 2.0
"""A human who takes over a retry-fixable failure also re-attempts the charge once, at day 2."""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Components:
    """Per-row ingredients of P(recover | plan). All arrays have length n."""

    hard: np.ndarray
    p_self: np.ndarray
    p_seen: np.ndarray
    lift_self: np.ndarray
    lift_retry: np.ndarray
    churn: np.ndarray
    p_esc: np.ndarray
    attempt: np.ndarray
    base_retry: np.ndarray  # (n, 5) at RETRY_T
    payday_boost: np.ndarray
    days_to_payday: np.ndarray
    is_bank: np.ndarray
    is_fixable: np.ndarray  # human escalation includes one manual retry on these
    outage_hours: np.ndarray
    liq: np.ndarray
    decay: float
    retry_scale: float

    def retry_p(self, t: float, k: int) -> np.ndarray:
        """P(the k-th retry of the plan, executed at day ``t``, succeeds | retries can succeed).

        Attempts beyond the per-failure budget (``MAX_RETRIES`` minus prior attempts) never run,
        so their success probability is zero (ADR-006 attempt cap).
        """
        base = _interp_rows(t, self.base_retry)
        bank = np.where(24.0 * t + 0.5 >= self.outage_hours, BANK_RETRY_OK, BANK_RETRY_DOWN)
        base = np.where(self.is_bank, bank, base)
        boost = np.where(self.days_to_payday <= t, self.payday_boost, 1.0)
        decay = self.decay ** (self.attempt - 1 + k - 1)
        allowed = (k <= MAX_RETRIES + 1 - self.attempt).astype(float)
        return np.clip(base * self.liq * boost * decay * self.retry_scale * allowed, 0.0, P_MAX)


def _interp_rows(t: float, table: np.ndarray) -> np.ndarray:
    """Row-wise linear interpolation of ``table`` (n, len(RETRY_T)) at time ``t``; flat beyond 7d."""
    if t >= RETRY_T[-1]:
        return table[:, -1]
    if t <= RETRY_T[0]:
        return table[:, 0]
    i = int(np.searchsorted(RETRY_T, t))
    if RETRY_T[i] == t:
        return table[:, i]
    w = (t - RETRY_T[i - 1]) / (RETRY_T[i] - RETRY_T[i - 1])
    return table[:, i - 1] * (1 - w) + table[:, i] * w


class OutcomeModel:
    """True outcome process for one simulator variant."""

    def __init__(
        self,
        variant: SimVariant,
        retry_scale: float | None = None,
        overrides: dict[str, float] | None = None,
    ) -> None:
        """``overrides`` are sensitivity knobs (see ``SENSITIVITY_KEYS``); unknown keys raise."""
        self.variant = variant
        self.overrides = dict(overrides or {})
        bad = set(self.overrides) - set(SENSITIVITY_KEYS)
        if bad:
            raise ValueError(f"unknown sensitivity overrides: {sorted(bad)}")
        base_scale = RETRY_SCALE[variant] if retry_scale is None else retry_scale
        self.retry_scale = base_scale * self.overrides.get("retry_scale_mult", 1.0)
        self.decay = self.overrides.get("decay", ATTEMPT_DECAY[variant])
        self.params = dict(CALIBRATED)
        esc_level = self.overrides.get("esc_level")
        if esc_level is not None:
            self.params = {
                c: (dataclasses.replace(p, esc0=esc_level) if c in RETRY_FIXABLE else p)
                for c, p in self.params.items()
            }
        if variant == "misspecified":
            prng = np.random.default_rng(7)
            perturbed: dict[str, CategoryParams] = {}
            for cat, p in self.params.items():
                m = prng.uniform(0.8, 1.2, size=3)
                perturbed[cat] = CategoryParams(
                    hard=p.hard, self0=p.self0 * m[0],
                    retry=tuple(float(r * m[1]) for r in p.retry),  # type: ignore[arg-type]
                    lift_self=p.lift_self, lift_retry=p.lift_retry, churn0=p.churn0,
                    esc0=p.esc0 * m[2], payday_boost=p.payday_boost,
                )
            self.params = perturbed

    # ---- ingredients -----------------------------------------------------------------------
    def _per_cat(self, cat: np.ndarray, field: str) -> np.ndarray:
        table = {c: getattr(p, field) for c, p in self.params.items()}
        if field == "retry":
            return np.array([table[c] for c in cat], dtype=float)
        return np.array([table[c] for c in cat], dtype=float)

    def components(self, obs: pd.DataFrame, hidden: pd.DataFrame) -> Components:
        """Compute all per-row ingredients from observable + hidden state."""
        v = self.variant
        cat = obs["failure_category"].to_numpy().astype(str)
        is_b2b = (obs["segment"].to_numpy() == "b2b")
        amount = obs["amount"].to_numpy(dtype=float)
        contacts7 = obs["contacts_last_7d"].to_numpy(dtype=float)
        tenure = obs["customer_tenure_months"].to_numpy(dtype=float)
        risk = obs["risk_score"].to_numpy(dtype=float)
        attempt = obs["attempt_number"].to_numpy(dtype=float)
        z_liq = hidden["z_liquidity"].to_numpy()
        z_eng = hidden["z_engagement"].to_numpy()
        churn_intent = hidden["churn_intent"].to_numpy()

        money = np.isin(cat, MONEY_CATEGORIES)
        liq = np.exp(np.where(money, 0.45, 0.10) * z_liq)

        if v == "misspecified":
            amount_eff = np.where(
                is_b2b,
                np.clip(1 + 0.15 * np.log(amount / 5000.0), 0.5, 1.8),
                np.where(amount > 2000, 0.6, 1.0),
            )
        else:
            amount_eff = np.where(is_b2b, np.clip(1 + 0.08 * np.log(amount / 5000.0), 0.7, 1.5), 1.0)
        p_self = (
            self._per_cat(cat, "self0")
            * np.exp(0.35 * z_eng)
            * np.where(is_b2b, 1.3, 1.0)
            * (1 + 0.01 * np.minimum(tenure, 24))
            * amount_eff
        )
        p_seen = _sigmoid(0.4 + 0.9 * z_eng - 0.5 * contacts7)

        lift_self0 = self._per_cat(cat, "lift_self")
        lift_retry0 = self._per_cat(cat, "lift_retry")
        if v == "misspecified":
            hetero = np.clip(1 + 0.8 * z_eng, -0.5, 2.5)  # backfires on disengaged customers
            lift_self = 1 + (lift_self0 - 1) * hetero
            lift_retry = 1 + (lift_retry0 - 1) * hetero
            churn = (
                self._per_cat(cat, "churn0")
                * (0.5 + 1.5 * churn_intent)
                * np.where(contacts7 >= 2, 3.0, 1.0)
            )
            days_to_payday = np.where(
                hidden["hidden_segment"].to_numpy(),
                hidden["days_to_payday_hidden"].to_numpy(),
                obs["days_to_payday"].to_numpy(),
            )
        else:
            fatigue = np.maximum(0.0, 1 - 0.3 * contacts7)
            lift_self = 1 + (lift_self0 - 1) * fatigue
            lift_retry = 1 + (lift_retry0 - 1) * fatigue
            churn = self._per_cat(cat, "churn0") * (0.5 + 1.5 * churn_intent) * (1 + 0.6 * contacts7)
            days_to_payday = obs["days_to_payday"].to_numpy()

        # sensitivity knobs (identity by default)
        p_self = p_self * self.overrides.get("self_mult", 1.0)
        m = self.overrides.get("msg_lift_mult", 1.0)
        lift_self = 1 + (lift_self - 1) * m
        lift_retry = 1 + (lift_retry - 1) * m
        churn = churn * self.overrides.get("churn_mult", 1.0)
        payday_boost = 1 + (self._per_cat(cat, "payday_boost") - 1) * self.overrides.get(
            "payday_mult", 1.0
        )

        p_esc = (
            self._per_cat(cat, "esc0")
            * np.where(is_b2b, 1.15, 1.0)
            * np.where(cat == "risk_declined", 1 - 0.3 * risk / 100.0, 1.0)
        )
        hard = self._per_cat(cat, "hard")
        base_retry = self._per_cat(cat, "retry")

        if v == "null_uplift":
            hard = np.ones_like(hard)
            lift_self = np.ones_like(lift_self)
            lift_retry = np.ones_like(lift_retry)
            p_esc = np.zeros_like(p_esc)

        is_bank = cat == "bank_technical"
        if v == "drifted":
            merchant = obs["merchant_id"].to_numpy().astype(str)
            fatigue = np.maximum(0.0, 1 - 0.3 * contacts7)
            for (mid, cat_name), spec in DRIFT.items():
                m = (merchant == mid) & (cat == cat_name)
                if not m.any():
                    continue
                old = self.params[cat_name]
                if "like" in spec:
                    tgt = self.params[str(spec["like"])]
                    hard[m] = tgt.hard
                    p_self[m] = p_self[m] * (tgt.self0 / old.self0)
                    base_retry[m] = np.array(tgt.retry)
                    payday_boost[m] = tgt.payday_boost
                    lift_self[m] = 1 + (tgt.lift_self - 1) * fatigue[m]
                    lift_retry[m] = 1 + (tgt.lift_retry - 1) * fatigue[m]
                    p_esc[m] = p_esc[m] * (tgt.esc0 / old.esc0)
                    is_bank[m] = False
                    continue
                if "hard" in spec:
                    hard[m] = float(spec["hard"])
                if "self0" in spec:
                    p_self[m] = p_self[m] * (float(spec["self0"]) / old.self0)
                if "retry" in spec:
                    base_retry[m] = np.array(spec["retry"], dtype=float)
                if "payday_boost" in spec:
                    payday_boost[m] = float(spec["payday_boost"])
                if "lift_self" in spec:
                    lift_self[m] = 1 + (float(spec["lift_self"]) - 1) * fatigue[m]
                if "lift_retry" in spec:
                    lift_retry[m] = 1 + (float(spec["lift_retry"]) - 1) * fatigue[m]
                if "churn_mult" in spec:
                    churn[m] = churn[m] * float(spec["churn_mult"])
                if "esc0" in spec:
                    p_esc[m] = p_esc[m] * (float(spec["esc0"]) / old.esc0)

        return Components(
            hard=np.clip(hard, 0, 1),
            p_self=np.clip(p_self, 0, P_MAX),
            p_seen=p_seen,
            lift_self=lift_self,
            lift_retry=lift_retry,
            churn=np.clip(churn, 0, 0.9),
            p_esc=np.clip(p_esc, 0, P_MAX),
            attempt=attempt,
            base_retry=base_retry,
            payday_boost=payday_boost,
            days_to_payday=days_to_payday.astype(float),
            is_bank=is_bank,
            is_fixable=np.isin(cat, RETRY_FIXABLE),
            outage_hours=hidden["outage_hours"].to_numpy(),
            liq=liq,
            decay=self.decay,
            retry_scale=self.retry_scale,
        )

    # ---- analytic probability ----------------------------------------------------------------
    def probability(self, comp: Components, plan: Plan) -> np.ndarray:
        """Exact P(recover | x, hidden, plan) integrating over the CRN draws."""

        def branch(seen: bool) -> np.ndarray:
            ls = comp.lift_self if seen else 1.0
            lr = comp.lift_retry if seen else 1.0
            p_self = np.clip(comp.p_self * ls, 0, P_MAX)
            q_fail = np.ones_like(p_self)
            for k, t in enumerate(plan.retry_days, start=1):
                q_fail = q_fail * (1 - np.clip(comp.retry_p(t, k) * lr, 0, P_MAX))
            if plan.escalate:  # human also re-attempts once on retry-fixable categories
                q_fail = q_fail * (1 - comp.is_fixable * comp.retry_p(ESCALATION_RETRY_DAY, 1))
            retry_none = comp.hard + (1 - comp.hard) * q_fail
            esc = comp.p_esc if plan.escalate else 0.0
            churn = comp.churn if (plan.message and seen) else 0.0
            p_no = churn + (1 - churn) * (1 - p_self) * (1 - esc) * retry_none
            return 1 - p_no

        if not plan.message:
            return branch(False)
        return comp.p_seen * branch(True) + (1 - comp.p_seen) * branch(False)

    # ---- realisation with common random numbers -----------------------------------------------
    def realize(
        self, comp: Components, plan: Plan, u: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Realised (recovered, churned, message_seen) for ``plan`` using the stored uniforms."""
        seen = plan.message & (u["u_msg"].to_numpy() < comp.p_seen)
        p_self = np.clip(comp.p_self * np.where(seen, comp.lift_self, 1.0), 0, P_MAX)
        self_ok = u["u_self"].to_numpy() < p_self
        churned = seen & (u["u_churn"].to_numpy() < comp.churn)
        hard_fail = u["u_hard"].to_numpy() < comp.hard
        retry_ok = np.zeros(len(seen), dtype=bool)
        for k, t in enumerate(plan.retry_days, start=1):
            p = np.clip(comp.retry_p(t, k) * np.where(seen, comp.lift_retry, 1.0), 0, P_MAX)
            retry_ok |= u[f"u_att{k}"].to_numpy() < p
        if plan.escalate:
            p = comp.is_fixable * comp.retry_p(ESCALATION_RETRY_DAY, 1)
            retry_ok |= u["u_att1"].to_numpy() < p
        retry_ok &= ~hard_fail
        esc_ok = plan.escalate & (u["u_esc"].to_numpy() < comp.p_esc)
        recovered = ~churned & (self_ok | retry_ok | esc_ok)
        return recovered, churned, seen

    # ---- counterfactual table --------------------------------------------------------------
    def plans(self) -> dict[str, Plan]:
        """Every action the evaluator can score: 7 primitives + the Razorpay default schedule."""
        out = {primitive_name(a, d): plan_for(a, d) for a, d in PRIMITIVE_ACTIONS}
        out["razorpay_default"] = RAZORPAY_DEFAULT_PLAN  # alias of retry_delayed_1 (ADR-006)
        out["razorpay_t123"] = RAZORPAY_T123_PLAN  # literal T+1/T+2/T+3 spacing, sensitivity only
        return out

    def counterfactual_table(self, obs: pd.DataFrame, hidden: pd.DataFrame) -> pd.DataFrame:
        """``y_<action>`` realised outcome, ``p_<action>`` true probability, ``churn_<action>``."""
        comp = self.components(obs, hidden)
        cols: dict[str, np.ndarray] = {}
        for name, plan in self.plans().items():
            y, churned, _ = self.realize(comp, plan, hidden)
            cols[f"y_{name}"] = y
            cols[f"p_{name}"] = self.probability(comp, plan)
            cols[f"churn_{name}"] = churned
        return pd.DataFrame(cols)


def calibrate_retry_scale(
    obs: pd.DataFrame,
    hidden: pd.DataFrame,
    target: float = 0.60,
    variant: SimVariant = "calibrated",
    lo: float = 0.2,
    hi: float = 4.0,
    tol: float = 1e-3,
) -> float:
    """Bisect the global retry multiplier so that the Razorpay default schedule hits ``target``."""
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        model = OutcomeModel(variant, retry_scale=mid)
        rate = float(model.probability(model.components(obs, hidden), RAZORPAY_DEFAULT_PLAN).mean())
        if abs(rate - target) < tol:
            return mid
        if rate < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
