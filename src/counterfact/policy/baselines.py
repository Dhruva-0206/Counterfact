"""Baseline policies: ``no_action``, ``razorpay_default`` and a rule-table ``heuristic``.

A policy for evaluation purposes is a function ``DataFrame -> np.ndarray[str]`` returning, per
row, the name of an action column in the counterfactual table (``y_<name>``). Primitive action
names come from :func:`counterfact.config.primitive_name`; ``razorpay_default`` is the T+1/T+2/T+3
retry schedule and is not one of the five arms.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from counterfact.config import (
    ESCALATE_HUMAN,
    NO_ACTION,
    REMIND_AND_RETRY,
    RETRY_DELAYED,
    RETRY_NOW,
    primitive_name,
)

Policy = Callable[[pd.DataFrame], np.ndarray]

HEURISTIC_TABLE: dict[str, tuple[int, int]] = {
    "insufficient_funds": (REMIND_AND_RETRY, 0),
    "bank_technical": (RETRY_DELAYED, 1),
    "gateway_5xx": (RETRY_NOW, 0),
    "auth_failed": (REMIND_AND_RETRY, 0),
    "card_expired": (ESCALATE_HUMAN, 0),
    "risk_declined": (ESCALATE_HUMAN, 0),
    "customer_cancelled": (NO_ACTION, 0),
    "mandate_failed": (RETRY_DELAYED, 3),
}
"""What an experienced ops team would do per failure category, ignoring amount and history."""


def heuristic_actions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Rule-table arm and delay per row."""
    cats = df["failure_category"].to_numpy().astype(str)
    arms = np.array([HEURISTIC_TABLE[c][0] for c in cats], dtype=int)
    delays = np.array([HEURISTIC_TABLE[c][1] for c in cats], dtype=int)
    return arms, delays


def action_names(arms: np.ndarray, delays: np.ndarray) -> np.ndarray:
    """Vectorised ``primitive_name`` over parallel arm / delay arrays."""
    return np.array(
        [primitive_name(int(a), int(d)) for a, d in zip(arms, delays, strict=True)], dtype=object
    )


def no_action_policy(df: pd.DataFrame) -> np.ndarray:
    return np.full(len(df), primitive_name(NO_ACTION), dtype=object)


def razorpay_default_policy(df: pd.DataFrame) -> np.ndarray:
    return np.full(len(df), "razorpay_default", dtype=object)


def heuristic_policy(df: pd.DataFrame) -> np.ndarray:
    return action_names(*heuristic_actions(df))


BASELINES: dict[str, Policy] = {
    "no_action": no_action_policy,
    "razorpay_default": razorpay_default_policy,
    "heuristic": heuristic_policy,
}
