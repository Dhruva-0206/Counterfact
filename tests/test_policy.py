"""Net-EV policy semantics, feature builder leakage checks, and the T-learner contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from counterfact.config import (
    ARMS,
    ESCALATE_HUMAN,
    NO_ACTION,
    REMIND_AND_RETRY,
    RETRY_DELAYED,
    RETRY_NOW,
)
from counterfact.features import build as fb
from counterfact.models.uplift import TLearner, UpliftConfig
from counterfact.policy.ev import (
    MLPolicy,
    arm_costs,
    choose,
    fatigue_penalty,
    merchant_table,
    net_ev,
)
from counterfact.sim.generator import generate_customers, generate_failures, merchants_frame
from counterfact.sim.logging_policy import log_actions
from counterfact.sim.outcome_model import OutcomeModel


@pytest.fixture(scope="module")
def logged() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    obs, hidden = generate_failures(generate_customers(800, rng), 3_000, rng)
    cf = OutcomeModel("calibrated").counterfactual_table(obs, hidden)
    return pd.concat([obs, log_actions(obs, cf, np.random.default_rng(6))], axis=1)


# ---- ev.py -----------------------------------------------------------------------------------
def test_no_action_wins_ties_and_threshold() -> None:
    ev = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 5.0, 3.0, 0.0, 0.0], [0.0, 50.0, 60.0, 0.0, 0.0],
                   [0.0, -5.0, -1.0, -2.0, -3.0]])
    arm = choose(ev, np.array([10.0, 10.0, 10.0, 10.0]))
    assert arm.tolist() == [NO_ACTION, NO_ACTION, RETRY_DELAYED, NO_ACTION]


def test_net_ev_no_action_is_exactly_zero_and_costs_enter() -> None:
    uplift = np.array([[0.0, 0.1, 0.2, 0.3, 0.4]])
    cost = np.array([[0.0, 1.5, 1.5, 60.0, 150.0]])
    ev = net_ev(uplift, np.array([1000.0]), cost)
    assert ev[0, NO_ACTION] == 0.0
    np.testing.assert_allclose(ev[0, 1:], [98.5, 198.5, 240.0, 250.0])


def test_arm_costs_and_fatigue(logged: pd.DataFrame) -> None:
    m = merchant_table(merchants_frame())
    df = logged.head(5).copy()
    df["contacts_last_7d"] = [0, 1, 2, 3, 0]
    c = arm_costs(df, m)
    assert (c[:, NO_ACTION] == 0).all()
    assert (c[:, RETRY_NOW] == c[:, RETRY_DELAYED]).all() and (c[:, RETRY_NOW] > 0).all()
    assert (c[:, ESCALATE_HUMAN] >= 120).all()
    # reminder cost grows with contacts_last_7d through the fatigue penalty
    fat = fatigue_penalty(df["contacts_last_7d"].to_numpy(float), df["amount"].to_numpy(float),
                          np.full(5, 0.01), np.full(5, 3.0))
    assert fat[0] == 0 and fat[3] == pytest.approx(0.09 * df["amount"].iloc[3])
    assert c[3, REMIND_AND_RETRY] > c[0, REMIND_AND_RETRY]


# ---- features/build.py -----------------------------------------------------------------------
def test_feature_matrix_excludes_logged_and_id_columns(logged: pd.DataFrame) -> None:
    X = fb.build_features(logged)
    assert not set(X.columns) & set(fb.LOGGED_COLS) and not set(X.columns) & set(fb.ID_COLS)
    assert X.shape[0] == len(logged) and list(X.columns) == list(fb.FEATURES)
    # categorical codes are fixed by vocabulary, not by data order
    assert list(X["failure_category"].cat.categories)[0] == "insufficient_funds"


def test_no_feature_correlates_with_outcome(logged: pd.DataFrame) -> None:
    rep = fb.leakage_report(logged, fb.build_features(logged))
    assert (rep["abs_corr"] < 0.99).all(), rep.head()


def test_training_never_opens_counterfactuals(monkeypatch, tmp_path: Path, logged: pd.DataFrame) -> None:
    """Dynamic guard: any parquet read whose path mentions counterfactuals raises during training."""
    real = pd.read_parquet

    def guarded(path, *a, **k):
        if "counterfactual" in str(path):
            raise PermissionError("training tried to read the counterfactual table")
        return real(path, *a, **k)

    monkeypatch.setattr(pd, "read_parquet", guarded)
    d = tmp_path / "calibrated"
    d.mkdir()
    logged.to_parquet(d / "failures.parquet", index=False)
    pd.DataFrame({"event_id": logged["event_id"]}).to_parquet(d / "counterfactuals.parquet", index=False)
    from counterfact.config import Settings
    s = Settings(data_dir=tmp_path)
    df = fb.load_training_frame(s, "calibrated")
    X = fb.build_features(df)
    TLearner(UpliftConfig(n_estimators=20, min_child_samples=10)).fit(
        X, df["action_idx"].to_numpy(), df["recovered"].to_numpy(), df["propensity"].to_numpy()
    )
    with pytest.raises(PermissionError):
        pd.read_parquet(d / "counterfactuals.parquet")


def test_features_hash_is_stable_and_ignores_logged_columns(logged: pd.DataFrame) -> None:
    r = logged.iloc[0]
    h1 = fb.features_hash(r)
    r2 = r.copy()
    r2["recovered"] = not r2["recovered"]
    assert fb.features_hash(r2) == h1 and len(h1) == 16


# ---- models/uplift.py --------------------------------------------------------------------------
def test_tlearner_contract(logged: pd.DataFrame) -> None:
    X = fb.build_features(logged)
    m = TLearner(UpliftConfig(n_estimators=30, min_child_samples=10, n_ensemble=2)).fit(
        X, logged["action_idx"].to_numpy(), logged["recovered"].to_numpy(), logged["propensity"].to_numpy()
    )
    up = m.predict_uplift(X)
    assert up.shape == (len(X), len(ARMS)) and (up[:, NO_ACTION] == 0).all()
    pred = m.predict(X)
    assert pred.primitive_mean.shape == (len(X), 7) and set(pred.best_delay) <= {1, 3, 7}
    lcb = m.predict_uplift(X, estimate="lcb", z=1.0)
    assert (lcb[:, 1:] <= up[:, 1:] + 1e-12).all()
    policy = MLPolicy(m, merchants_frame())
    dec = policy.decide(logged, X)
    assert set(dec.arm) <= set(range(5)) and (dec.delay_days[dec.arm != RETRY_DELAYED] == 0).all()


def test_gated_estimator_abstains_more_than_mean(logged: pd.DataFrame) -> None:
    X = fb.build_features(logged)
    m = TLearner(UpliftConfig(n_estimators=30, min_child_samples=10, n_ensemble=4)).fit(
        X, logged["action_idx"].to_numpy(), logged["recovered"].to_numpy(), logged["propensity"].to_numpy()
    )
    mean_dec = MLPolicy(m, merchants_frame(), estimate="mean").decide(logged, X)
    gated_dec = MLPolicy(m, merchants_frame(), estimate="gated", z=3.0).decide(logged, X)
    assert (gated_dec.arm == NO_ACTION).mean() > (mean_dec.arm == NO_ACTION).mean()
    # when the gate opens, the arm is the same one the point estimate would pick
    acted = gated_dec.arm != NO_ACTION
    assert (gated_dec.arm[acted] == mean_dec.arm[acted]).all()
