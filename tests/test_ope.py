"""OPE estimators on a toy bandit with a known answer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from counterfact.eval.ope import (
    direct_method,
    doubly_robust,
    effective_sample_size,
    estimate,
    importance_weights,
    ips,
    snips,
)

ACTIONS = np.array(["a", "b", "c"])
TRUE_MEAN = {"a": 1.0, "b": 3.0, "c": 5.0}  # E[r | action]


def toy(n: int = 40_000, seed: int = 0, p: tuple[float, ...] = (0.6, 0.3, 0.1)):
    rng = np.random.default_rng(seed)
    logged = rng.choice(ACTIONS, size=n, p=p)
    prop = np.array([p[list(ACTIONS).index(a)] for a in logged])
    r = np.array([TRUE_MEAN[a] for a in logged]) + rng.normal(0, 1, n)
    return logged, prop, r


def test_ips_and_snips_recover_known_value() -> None:
    logged, prop, r = toy()
    target = np.full(len(logged), "c")
    w = importance_weights(target, logged, prop)
    assert abs(ips(w, r) - 5.0) < 0.15
    assert abs(snips(w, r) - 5.0) < 0.15
    assert 0 < effective_sample_size(w) < len(logged)


def test_dr_is_unbiased_with_a_biased_reward_model() -> None:
    logged, prop, r = toy()
    target = np.full(len(logged), "b")
    q = pd.DataFrame({a: np.full(len(logged), TRUE_MEAN[a] + 1.5) for a in ACTIONS})  # biased by +1.5
    w = importance_weights(target, logged, prop)
    q_t = q["b"].to_numpy()
    q_l = q.to_numpy()[np.arange(len(q)), q.columns.get_indexer(logged)]
    assert abs(direct_method(q_t) - 3.0) == pytest.approx(1.5)  # DM inherits the bias
    assert abs(doubly_robust(w, r, q_l, q_t) - 3.0) < 0.1  # DR corrects it


def test_dr_has_lower_variance_than_ips_with_a_good_model() -> None:
    logged, prop, r = toy(n=20_000)
    target = np.full(len(logged), "c")
    q = pd.DataFrame({a: np.full(len(logged), TRUE_MEAN[a]) for a in ACTIONS})
    est = estimate("always_c", target, logged, prop, r, q, n_boot=100)
    assert est.dr_se < est.ips_se
    assert abs(est.dr - 5.0) < 0.1 and abs(est.ips - 5.0) < 0.3
    assert est.match_rate == pytest.approx(0.1, abs=0.01)


def test_context_dependent_policy() -> None:
    """Target picks 'a' on even rows and 'c' on odd rows: truth = 3.0."""
    logged, prop, r = toy()
    target = np.where(np.arange(len(logged)) % 2 == 0, "a", "c")
    q = pd.DataFrame({a: np.full(len(logged), TRUE_MEAN[a]) for a in ACTIONS})
    est = estimate("alt", target, logged, prop, r, q, n_boot=50)
    for v in (est.ips, est.snips, est.dr, est.dm):
        assert abs(v - 3.0) < 0.15
