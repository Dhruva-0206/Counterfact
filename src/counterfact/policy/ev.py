"""Net expected value per arm and the argmax policy with a first-class ``no_action``.

    net_ev[arm] = uplift[arm] * amount - cost[arm] - fatigue_penalty(contacts_last_7d)

Costs come from the merchant configuration (:class:`counterfact.sim.schema.Merchant`):

* retries: ``retry_cost`` per attempt times the attempts in the schedule (3 under ADR-006)
* reminder: ``message_cost`` + retry costs + contact-fatigue penalty (ADR-010)
* escalation: ``ops_cost``

``no_action`` wins ties and wins whenever the best net EV is below ``min_ev_threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from counterfact.config import (
    ARMS,
    ESCALATE_HUMAN,
    NO_ACTION,
    REMIND_AND_RETRY,
    RETRY_DELAYED,
    RETRY_NOW,
    primitive_name,
)
from counterfact.models.uplift import TLearner, UpliftPrediction
from counterfact.sim.schema import MAX_RETRIES, Merchant

MERCHANT_COLS = (
    "merchant_id", "min_ev_threshold", "escalation_amount_threshold", "message_cost",
    "retry_cost", "ops_cost", "fatigue_rate", "ltv_cycles", "kill_switch",
)


def merchant_table(merchants: pd.DataFrame | list[Merchant]) -> pd.DataFrame:
    """Merchant config indexed by ``merchant_id``."""
    if isinstance(merchants, pd.DataFrame):
        t = merchants
    else:
        t = pd.DataFrame([m.model_dump() for m in merchants])
    return t.set_index("merchant_id")[list(MERCHANT_COLS[1:])]


def fatigue_penalty(
    contacts_last_7d: np.ndarray, amount: np.ndarray, fatigue_rate: np.ndarray, ltv_cycles: np.ndarray
) -> np.ndarray:
    """Expected LTV put at risk by one more contact (ADR-010)."""
    return fatigue_rate * contacts_last_7d * amount * ltv_cycles


def arm_costs(df: pd.DataFrame, merchants: pd.DataFrame, n_retries: int = MAX_RETRIES) -> np.ndarray:
    """``(n, 5)`` rupee cost of executing each arm for each row."""
    m = merchants.loc[df["merchant_id"].to_numpy()]
    amount = df["amount"].to_numpy(dtype=float)
    contacts = df["contacts_last_7d"].to_numpy(dtype=float)
    retry = m["retry_cost"].to_numpy(dtype=float) * n_retries
    cost = np.zeros((len(df), len(ARMS)))
    cost[:, RETRY_NOW] = retry
    cost[:, RETRY_DELAYED] = retry
    cost[:, REMIND_AND_RETRY] = (
        m["message_cost"].to_numpy(dtype=float)
        + retry
        + fatigue_penalty(
            contacts, amount, m["fatigue_rate"].to_numpy(dtype=float), m["ltv_cycles"].to_numpy(dtype=float)
        )
    )
    cost[:, ESCALATE_HUMAN] = m["ops_cost"].to_numpy(dtype=float)
    return cost


def net_ev(uplift: np.ndarray, amount: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """``(n, 5)`` net expected value; ``no_action`` is exactly zero by construction."""
    ev = uplift * amount[:, None] - cost
    ev[:, NO_ACTION] = 0.0
    return ev


def choose(ev: np.ndarray, min_ev_threshold: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Argmax with ``no_action`` winning ties and any case below the merchant threshold."""
    best = ev.argmax(axis=1)
    best_val = ev[np.arange(len(ev)), best]
    below = best_val < np.asarray(min_ev_threshold, dtype=float)
    tie = best_val <= ev[:, NO_ACTION] + tol
    return np.where(below | tie, NO_ACTION, best)


@dataclass
class Decision:
    """Vectorised policy output aligned to the input rows."""

    arm: np.ndarray
    delay_days: np.ndarray
    best_delay: np.ndarray  # (n,) delay the model prefers if retry_delayed is executed
    uplift: np.ndarray  # (n, 5)
    net_ev: np.ndarray  # (n, 5)
    cost: np.ndarray  # (n, 5)
    base: np.ndarray  # (n,) P(recover | no_action)
    threshold: np.ndarray  # (n,)

    def action_names(self) -> np.ndarray:
        return np.array(
            [primitive_name(int(a), int(d)) for a, d in zip(self.arm, self.delay_days, strict=True)],
            dtype=object,
        )

    def frame(self) -> pd.DataFrame:
        out = pd.DataFrame({"arm": self.arm, "delay_days": self.delay_days, "threshold": self.threshold})
        out["action_name"] = self.action_names()
        out["p_no_action_hat"] = self.base
        for i, name in enumerate(ARMS):
            out[f"uplift_{name}"] = self.uplift[:, i]
        for i, name in enumerate(ARMS):
            out[f"net_ev_{name}"] = self.net_ev[:, i]
        return out


class MLPolicy:
    """Uplift model + merchant economics -> arm per row (before guardrails)."""

    def __init__(
        self,
        model: TLearner,
        merchants: pd.DataFrame,
        estimate: str = "mean",
        z: float = 1.0,
    ) -> None:
        self.model = model
        self.merchants = merchant_table(merchants) if "merchant_id" in merchants.columns else merchants
        self.estimate = estimate
        self.z = z

    def decide(self, df: pd.DataFrame, X: pd.DataFrame, pred: UpliftPrediction | None = None) -> Decision:
        """Three estimators (ADR-007):

        * ``mean``  rank and gate on the point estimate;
        * ``lcb``   rank and gate on the ensemble lower confidence bound (mean - z * std);
        * ``gated`` act only if some arm's LCB net EV clears the merchant threshold, then pick the
          arm by mean net EV ("gate on confidence, rank on expectation").
        """
        amount = df["amount"].to_numpy(dtype=float)
        cost = arm_costs(df, self.merchants)
        thr = self.merchants.loc[df["merchant_id"].to_numpy(), "min_ev_threshold"].to_numpy(dtype=float)
        if self.estimate == "gated":
            pred = pred or self.model.predict(X, estimate="mean")
            lcb = self.model.predict(X, estimate="lcb", z=self.z)
            ev = net_ev(pred.uplift, amount, cost)
            ev_lcb = net_ev(lcb.uplift, amount, cost)
            arm = choose(ev, thr)
            confident = ev_lcb.max(axis=1) >= thr
            arm = np.where(confident, arm, NO_ACTION)
        else:
            pred = pred or self.model.predict(X, estimate=self.estimate, z=self.z)
            ev = net_ev(pred.uplift, amount, cost)
            arm = choose(ev, thr)
        delay = np.where(arm == RETRY_DELAYED, pred.best_delay, 0)
        return Decision(arm, delay, pred.best_delay, pred.uplift, ev, cost, pred.base, thr)
