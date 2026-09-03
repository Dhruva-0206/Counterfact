"""Off-policy evaluation: IPS, self-normalized IPS and doubly-robust estimates.

Answers "what would policy pi have earned?" from logged data alone, i.e. without exposing a
customer to pi. Requires the logging policy's propensities (stored per row) and, for DR/DM, a
reward model ``q_hat(x, a)`` (here: the uplift models' per-action recovery probabilities, trained
on the training split; OPE runs on the logged holdout rows, so the reward model never saw them).

For a deterministic target policy pi(x) and logged tuples (x_i, a_i, r_i, p_i):

    w_i   = 1[pi(x_i) = a_i] / p_i
    IPS   = mean(w_i r_i)
    SNIPS = sum(w_i r_i) / sum(w_i)
    DM    = mean(q_hat(x_i, pi(x_i)))
    DR    = mean(q_hat(x_i, pi(x_i)) + w_i (r_i - q_hat(x_i, a_i)))

All estimators are compared with the paired-exact truth from the counterfactual table in
``scripts/ope.py``; this module itself never reads it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OPEEstimate:
    policy: str
    n: int
    match_rate: float  # share of logged rows whose logged action equals the target action
    ess: float  # Kish effective sample size of the importance weights
    ips: float
    snips: float
    dm: float
    dr: float
    ips_se: float
    snips_se: float
    dr_se: float


def importance_weights(target: np.ndarray, logged: np.ndarray, propensity: np.ndarray) -> np.ndarray:
    """``1[target == logged] / propensity`` per row."""
    return (np.asarray(target) == np.asarray(logged)).astype(float) / np.asarray(propensity, dtype=float)


def ips(w: np.ndarray, r: np.ndarray) -> float:
    return float(np.mean(w * r))


def snips(w: np.ndarray, r: np.ndarray) -> float:
    s = float(np.sum(w))
    return float(np.sum(w * r) / s) if s > 0 else float("nan")


def direct_method(q_target: np.ndarray) -> float:
    return float(np.mean(q_target))


def doubly_robust(w: np.ndarray, r: np.ndarray, q_logged: np.ndarray, q_target: np.ndarray) -> float:
    return float(np.mean(q_target + w * (r - q_logged)))


def effective_sample_size(w: np.ndarray) -> float:
    s = float(np.sum(w))
    return float(s * s / np.sum(w * w)) if s > 0 else 0.0


def _bootstrap_se(fn, *arrays: np.ndarray, n_boot: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = fn(*[a[idx] for a in arrays])
    return float(np.std(vals, ddof=1))


def estimate(
    policy: str,
    target_actions: np.ndarray,
    logged_actions: np.ndarray,
    propensity: np.ndarray,
    reward: np.ndarray,
    q_hat: pd.DataFrame,
    n_boot: int = 200,
    seed: int = 0,
) -> OPEEstimate:
    """All estimators for one deterministic target policy.

    ``q_hat`` holds one column per action name with the reward model's prediction of the reward
    (already on the reward scale, e.g. P(recover) * amount) for every row.
    """
    w = importance_weights(target_actions, logged_actions, propensity)
    r = np.asarray(reward, dtype=float)
    q_t = q_hat.to_numpy()[np.arange(len(q_hat)), q_hat.columns.get_indexer(target_actions)]
    q_l = q_hat.to_numpy()[np.arange(len(q_hat)), q_hat.columns.get_indexer(logged_actions)]
    return OPEEstimate(
        policy=policy,
        n=len(r),
        match_rate=float(np.mean(w > 0)),
        ess=effective_sample_size(w),
        ips=ips(w, r),
        snips=snips(w, r),
        dm=direct_method(q_t),
        dr=doubly_robust(w, r, q_l, q_t),
        ips_se=_bootstrap_se(ips, w, r, n_boot=n_boot, seed=seed),
        snips_se=_bootstrap_se(snips, w, r, n_boot=n_boot, seed=seed),
        dr_se=_bootstrap_se(doubly_robust, w, r, q_l, q_t, n_boot=n_boot, seed=seed),
    )


def estimates_frame(results: list[OPEEstimate]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])
