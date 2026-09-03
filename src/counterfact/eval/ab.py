"""Randomized A/B evaluation on a held-out split with bootstrap confidence intervals.

Two numbers are reported for every policy against the control (``razorpay_default``):

* **A/B estimate**: units are randomized to policies (stratified by merchant x failure category);
  each unit's outcome comes from the counterfactual table for the action its policy chose. The
  difference in mean recovered rupees is what a real experiment would measure, with an
  independent-groups bootstrap CI.
* **Paired exact difference**: because the simulator stores every unit's outcome under every
  action, the same difference can be computed on all holdout units with no assignment noise.
  Reported as a reference; the A/B number is the honest one.

Reads ``counterfactuals.parquet``: this module is the evaluator, never a trainer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from counterfact.config import MERCHANT_IDS
from counterfact.eval.report import PER, policy_metrics, realised

STRATA = ("merchant_id", "failure_category")


def stratified_split(
    df: pd.DataFrame, holdout_frac: float, seed: int, strata: tuple[str, ...] = STRATA
) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of (train, holdout), stratified by ``strata``."""
    rng = np.random.default_rng(seed)
    hold = np.zeros(len(df), dtype=bool)
    keys = df[list(strata)].astype(str).agg("|".join, axis=1).to_numpy()
    for k in np.unique(keys):
        pos = np.flatnonzero(keys == k)
        rng.shuffle(pos)
        hold[pos[: int(round(holdout_frac * len(pos)))]] = True
    return np.flatnonzero(~hold), np.flatnonzero(hold)


def stratified_assignment(
    df: pd.DataFrame, n_groups: int, seed: int, strata: tuple[str, ...] = STRATA
) -> np.ndarray:
    """Balanced random assignment of rows to ``n_groups`` within each stratum."""
    rng = np.random.default_rng(seed)
    keys = df[list(strata)].astype(str).agg("|".join, axis=1).to_numpy()
    group = np.zeros(len(df), dtype=int)
    for k in np.unique(keys):
        pos = np.flatnonzero(keys == k)
        rng.shuffle(pos)
        group[pos] = np.arange(len(pos)) % n_groups
    return group


@dataclass
class ABRow:
    policy: str
    n: int
    metrics: dict[str, float]
    incr_vs_control_per_1k: float
    ci_low: float
    ci_high: float
    paired_exact_per_1k: float
    paired_recovery_delta: float


def bootstrap_diff(
    treat: np.ndarray, control: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Mean difference and percentile 95% CI for two independent groups."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        t = treat[rng.integers(0, len(treat), len(treat))]
        c = control[rng.integers(0, len(control), len(control))]
        diffs[b] = t.mean() - c.mean()
    return float(treat.mean() - control.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def ab_test(
    df: pd.DataFrame,
    cf: pd.DataFrame,
    policies: dict[str, np.ndarray],
    control: str = "razorpay_default",
    seed: int = 42,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """Randomized A/B of every policy vs ``control`` on ``df`` (already the holdout)."""
    names = list(policies)
    if control not in names:
        raise ValueError(f"control {control!r} must be one of the policies")
    group = stratified_assignment(df, len(names), seed)
    amount = df["amount"].to_numpy(dtype=float)
    value = {name: realised(cf, policies[name]) * amount for name in names}  # rupees per unit
    ctrl_mask = group == names.index(control)
    ctrl_val = value[control][ctrl_mask]
    rows = []
    for i, name in enumerate(names):
        mask = group == i
        m = policy_metrics(df[mask], cf[mask], policies[name][mask])
        est, lo, hi = bootstrap_diff(value[name][mask], ctrl_val, n_boot, seed + i)
        paired = float((value[name] - value[control]).mean() * PER)
        paired_rec = float((realised(cf, policies[name]) - realised(cf, policies[control])).mean())
        rows.append(
            {
                "policy": name,
                "n": int(mask.sum()),
                "incr_vs_control_per_1k": est * PER,
                "ci_low": lo * PER,
                "ci_high": hi * PER,
                "paired_exact_per_1k": paired,
                "paired_recovery_delta": paired_rec,
                **m,
            }
        )
    return pd.DataFrame(rows)


def paired_table(
    df: pd.DataFrame, cf: pd.DataFrame, policies: dict[str, np.ndarray], control: str = "razorpay_default"
) -> pd.DataFrame:
    """Exact paired comparison of every policy vs ``control`` on all rows (no assignment noise)."""
    amount = df["amount"].to_numpy(dtype=float)
    base = realised(cf, policies[control])
    rows = []
    for name, actions in policies.items():
        y = realised(cf, actions)
        m = policy_metrics(df, cf, actions)
        rows.append(
            {
                "policy": name,
                "n": len(df),
                "paired_exact_per_1k": float(((y - base) * amount).mean() * PER),
                "paired_recovery_delta": float((y - base).mean()),
                **m,
            }
        )
    return pd.DataFrame(rows)


def per_merchant(
    df: pd.DataFrame, cf: pd.DataFrame, policies: dict[str, np.ndarray], control: str = "razorpay_default"
) -> pd.DataFrame:
    """Paired exact incremental rupees per 1k and recovery per merchant and policy."""
    amount = df["amount"].to_numpy(dtype=float)
    rows = []
    for mid in MERCHANT_IDS:
        mask = (df["merchant_id"] == mid).to_numpy()
        if not mask.any():
            continue
        base = realised(cf[mask], policies[control][mask])
        for name, actions in policies.items():
            y = realised(cf[mask], actions[mask])
            rows.append(
                {
                    "merchant_id": mid,
                    "policy": name,
                    "n": int(mask.sum()),
                    "recovery_rate": float(y.mean()),
                    "incr_vs_control_per_1k": float(((y - base) * amount[mask]).mean() * PER),
                    "contacts_per_1k": float((actions[mask] == "remind_and_retry").mean() * PER),
                    "escalations_per_1k": float((actions[mask] == "escalate_human").mean() * PER),
                    "no_action_share": float((actions[mask] == "no_action").mean()),
                }
            )
    return pd.DataFrame(rows)
