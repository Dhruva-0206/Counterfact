"""Epsilon-uniform exploration policy that produced the logged data.

With probability ``EPSILON`` a primitive action is drawn uniformly from the 7 primitives; otherwise
the rule-table heuristic is followed. The propensity of the logged action is recorded on every
row so that IPS-style training weights and off-policy estimators are exact.

The logging phase is deliberately *not* guardrailed: an incumbent that retries expired cards is
what teaches the models that retrying expired cards does not work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recoverai.config import PRIMITIVE_ACTIONS, PRIMITIVE_INDEX, primitive_name
from recoverai.policy.baselines import heuristic_actions

EPSILON = 0.5
N_PRIMITIVES = len(PRIMITIVE_ACTIONS)


def log_actions(obs: pd.DataFrame, cf: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Choose a logged action per row and look up its realised outcome in the CRN table.

    Returns the logged columns only: ``arm, delay_days, action_idx, action_name, propensity,
    recovered, recovered_amount, contacted, churned, escalated``.
    """
    n = len(obs)
    h_arm, h_delay = heuristic_actions(obs)
    h_idx = np.array([PRIMITIVE_INDEX[(int(a), int(d))] for a, d in zip(h_arm, h_delay, strict=True)])
    explore = rng.random(n) < EPSILON
    idx = np.where(explore, rng.integers(0, N_PRIMITIVES, n), h_idx)
    propensity = EPSILON / N_PRIMITIVES + (1 - EPSILON) * (idx == h_idx)

    arms = np.array([PRIMITIVE_ACTIONS[i][0] for i in idx])
    delays = np.array([PRIMITIVE_ACTIONS[i][1] for i in idx])
    names = np.array([primitive_name(int(a), int(d)) for a, d in zip(arms, delays, strict=True)])

    y_cols = np.stack([cf[f"y_{primitive_name(a, d)}"].to_numpy() for a, d in PRIMITIVE_ACTIONS], axis=1)
    churn_cols = np.stack(
        [cf[f"churn_{primitive_name(a, d)}"].to_numpy() for a, d in PRIMITIVE_ACTIONS], axis=1
    )
    recovered = y_cols[np.arange(n), idx].astype(bool)
    churned = churn_cols[np.arange(n), idx].astype(bool)
    amount = obs["amount"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "arm": arms,
            "delay_days": delays,
            "action_idx": idx,
            "action_name": names,
            "propensity": propensity,
            "recovered": recovered,
            "recovered_amount": np.where(recovered, amount, 0.0),
            "contacted": arms == 3,
            "churned": churned,
            "escalated": arms == 4,
        }
    )
