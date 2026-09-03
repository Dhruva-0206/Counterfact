"""A/B machinery on a toy world with a known answer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from counterfact.eval.ab import ab_test, bootstrap_diff, stratified_assignment, stratified_split


def toy(n: int = 6_000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "merchant_id": rng.choice(["a", "b"], n),
            "failure_category": rng.choice(["x", "y", "z"], n),
            "amount": np.full(n, 100.0),
        }
    )
    # control recovers 50%, treatment recovers 60%, no_action 20%
    cf = pd.DataFrame(
        {
            "y_no_action": rng.random(n) < 0.20,
            "y_razorpay_default": rng.random(n) < 0.50,
            "y_treat": rng.random(n) < 0.60,
        }
    )
    return df, cf


def test_split_and_assignment_are_stratified_and_seeded() -> None:
    df, _ = toy()
    tr, ho = stratified_split(df, 0.4, 1)
    assert len(tr) + len(ho) == len(df) and abs(len(ho) / len(df) - 0.4) < 0.01
    tr2, _ = stratified_split(df, 0.4, 1)
    assert np.array_equal(tr, tr2)
    g = stratified_assignment(df, 3, 1)
    shares = pd.crosstab(df["merchant_id"] + df["failure_category"], g, normalize="index")
    assert (abs(shares - 1 / 3) < 0.02).all().all()


def test_ab_recovers_known_lift() -> None:
    df, cf = toy()
    pols = {
        "razorpay_default": np.full(len(df), "razorpay_default", dtype=object),
        "treat": np.full(len(df), "treat", dtype=object),
        "no_action": np.full(len(df), "no_action", dtype=object),
    }
    res = ab_test(df, cf, pols, seed=3, n_boot=300).set_index("policy")
    # true lift: 0.10 * Rs 100 * 1000 = Rs 10,000 per 1k
    assert 6_000 < res.loc["treat", "incr_vs_control_per_1k"] < 14_000
    assert res.loc["treat", "ci_low"] < 10_000 < res.loc["treat", "ci_high"]
    assert 8_000 < res.loc["treat", "paired_exact_per_1k"] < 12_000
    assert res.loc["no_action", "incr_vs_control_per_1k"] < -20_000
    assert res.loc["razorpay_default", "incr_vs_control_per_1k"] == 0.0


def test_bootstrap_ci_contains_truth() -> None:
    rng = np.random.default_rng(0)
    t = rng.normal(1.0, 1.0, 2000)
    c = rng.normal(0.0, 1.0, 2000)
    est, lo, hi = bootstrap_diff(t, c, 500, 0)
    assert lo < 1.0 < hi and abs(est - 1.0) < 0.15
