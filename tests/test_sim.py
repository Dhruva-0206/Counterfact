"""Simulator invariants: seeded reproducibility, failure mix, coherent counterfactuals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from counterfact.config import FAILURE_MIX, PRIMITIVE_ACTIONS, primitive_name
from counterfact.sim.generator import generate_customers, generate_failures
from counterfact.sim.logging_policy import EPSILON, N_PRIMITIVES, log_actions
from counterfact.sim.outcome_model import OutcomeModel
from counterfact.sim.schema import MAX_RETRIES, RAZORPAY_DEFAULT_PLAN, plan_for

N = 4_000


@pytest.fixture(scope="module")
def population() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(123)
    customers = generate_customers(1_000, rng)
    return generate_failures(customers, N, rng, "calibrated")


def test_seeded_reproducibility() -> None:
    a = generate_failures(generate_customers(500, np.random.default_rng(1)), 800, np.random.default_rng(1))
    b = generate_failures(generate_customers(500, np.random.default_rng(1)), 800, np.random.default_rng(1))
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])


def test_failure_mix_matches_spec(population) -> None:
    obs, _ = population
    share = obs["failure_category"].value_counts(normalize=True)
    for cat, p in FAILURE_MIX.items():
        assert abs(share[cat] - p) < 0.03, (cat, share[cat], p)


def test_instrument_consistency(population) -> None:
    obs, _ = population
    assert (obs.loc[obs.failure_category == "card_expired", "payment_method"] == "card").all()
    assert obs.loc[obs.failure_category == "mandate_failed", "payment_method"].isin(
        ["upi_autopay", "emandate"]
    ).all()
    assert obs.loc[obs.payment_method != "card", "card_expiry_days"].isna().all()
    assert (obs.loc[obs.failure_category == "card_expired", "card_expiry_days"] < 0).all()


def test_amount_and_ticket_range(population) -> None:
    obs, _ = population
    assert obs["amount"].min() >= 299
    assert (obs["plan_amount"].between(299, 15_000)).all()
    assert obs["days_to_payday"].between(0, 31).all()


def test_probabilities_bounded_and_coherent(population) -> None:
    obs, hidden = population
    model = OutcomeModel("calibrated")
    cf = model.counterfactual_table(obs, hidden)
    for a, d in PRIMITIVE_ACTIONS:
        p = cf[f"p_{primitive_name(a, d)}"]
        assert p.between(0, 1).all()
    # realised outcomes are Bernoulli(p): the mean must be close to the mean probability
    for name in ("no_action", "retry_now", "razorpay_default"):
        assert abs(cf[f"y_{name}"].mean() - cf[f"p_{name}"].mean()) < 0.03
    # ADR-006: the Razorpay default schedule IS retry_delayed(1) under the equalized budget
    np.testing.assert_allclose(cf["p_razorpay_default"], cf["p_retry_delayed_1"], atol=1e-12)
    assert (cf["y_razorpay_default"] == cf["y_retry_delayed_1"]).all()
    # any retry arm weakly dominates doing nothing, in probability
    for name in ("retry_now", "retry_delayed_1", "retry_delayed_3", "retry_delayed_7"):
        assert (cf[f"p_{name}"] >= cf["p_no_action"] - 1e-12).all()
    # attempt cap: with two prior attempts only one retry remains, so p(retry_now) sits
    # strictly below the first-attempt level for the same category (gateway_5xx is the cleanest)
    g5 = obs.failure_category == "gateway_5xx"
    first, third = g5 & (obs.attempt_number == 1), g5 & (obs.attempt_number == 3)
    assert cf.loc[first, "p_retry_now"].mean() > cf.loc[third, "p_retry_now"].mean() + 0.05
    # retrying an expired card is worthless; escalation is not
    ce = obs.failure_category == "card_expired"
    # (three attempts against the 3% issuer-updater share buys at most ~2-3 points)
    assert cf.loc[ce, "p_retry_now"].mean() - cf.loc[ce, "p_no_action"].mean() < 0.03
    assert cf.loc[ce, "p_escalate_human"].mean() > cf.loc[ce, "p_no_action"].mean() + 0.2
    # a human taking over a retry-fixable failure also re-attempts once, so escalation is never
    # far below a single automated retry on gateway errors, yet still below the full 3-attempt plan
    g5 = obs.failure_category == "gateway_5xx"
    assert cf.loc[g5, "p_escalate_human"].mean() > 0.7
    assert cf.loc[g5, "p_escalate_human"].mean() < cf.loc[g5, "p_retry_now"].mean()


def test_null_uplift_has_no_treatment_effect(population) -> None:
    obs, hidden = population
    cf = OutcomeModel("null_uplift").counterfactual_table(obs, hidden)
    base = cf["p_no_action"]
    for name in ("retry_now", "retry_delayed_1", "retry_delayed_7", "escalate_human", "razorpay_default"):
        np.testing.assert_allclose(cf[f"p_{name}"], base, atol=1e-12)
    # messages can only hurt (churn), never help
    assert (cf["p_remind_and_retry"] <= base + 1e-12).all()


def test_plans_are_bounded_and_budget_equalized() -> None:
    assert plan_for(0).n_retries == 0
    assert all(plan_for(a, d).n_retries <= MAX_RETRIES for a, d in PRIMITIVE_ACTIONS)
    # every retry arm gets the same three-attempt budget, two days apart (ADR-006)
    assert plan_for(1).retry_days == (0.0, 2.0, 4.0)
    assert plan_for(2, 3).retry_days == (3.0, 5.0, 7.0)
    assert plan_for(3).retry_days == (1.0, 3.0, 5.0) and plan_for(3).message
    assert RAZORPAY_DEFAULT_PLAN == plan_for(2, 1)
    assert not plan_for(4).retry_days and plan_for(4).escalate


def test_logging_policy_propensities(population) -> None:
    obs, hidden = population
    cf = OutcomeModel("calibrated").counterfactual_table(obs, hidden)
    logged = log_actions(obs, cf, np.random.default_rng(0))
    assert set(logged["propensity"].round(6)) == {
        round(EPSILON / N_PRIMITIVES, 6),
        round(EPSILON / N_PRIMITIVES + (1 - EPSILON), 6),
    }
    # every primitive has support
    assert logged["action_idx"].nunique() == N_PRIMITIVES
    # logged outcome equals the counterfactual column of the logged action
    for i in range(0, N, 97):
        name = logged.loc[i, "action_name"]
        assert bool(logged.loc[i, "recovered"]) == bool(cf.loc[i, f"y_{name}"])
