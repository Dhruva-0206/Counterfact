"""T-learner uplift model: one LightGBM classifier per primitive action, IPS-weighted.

``predict_uplift(X) -> (n, 5)`` returns, per arm, the estimated incremental recovery probability
over ``no_action``. For ``retry_delayed`` the three delays are scored separately and the best
one is reported together with its delay. An optional bootstrap ensemble supports a
lower-confidence-bound (LCB) estimate, the documented fallback if abstention under
``null_uplift`` is too low with point estimates (ADR-007).

This module trains on logged data only. It never imports the simulator's outcome process and
never reads the counterfactual table (enforced by tests).
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from counterfact.config import (
    ARMS,
    NO_ACTION,
    PRIMITIVE_ACTIONS,
    PRIMITIVE_INDEX,
    RETRY_DELAYED,
    RETRY_DELAYS,
)

N_PRIMITIVES = len(PRIMITIVE_ACTIONS)
DELAY_PRIMITIVES = {d: PRIMITIVE_INDEX[(RETRY_DELAYED, d)] for d in RETRY_DELAYS}


@dataclass
class UpliftConfig:
    """LightGBM hyper-parameters and ensemble settings. Deliberately regularized."""

    n_estimators: int = 300
    learning_rate: float = 0.04
    num_leaves: int = 15
    min_child_samples: int = 80
    reg_lambda: float = 5.0
    colsample_bytree: float = 0.8
    subsample: float = 0.8
    subsample_freq: int = 1
    n_ensemble: int = 1
    seed: int = 42
    extra: dict = field(default_factory=dict)

    def lgb_params(self, seed: int) -> dict:
        p = {
            "objective": "binary",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "reg_lambda": self.reg_lambda,
            "colsample_bytree": self.colsample_bytree,
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "random_state": seed,
            "verbose": -1,
            "n_jobs": 4,
        }
        p.update(self.extra)
        return p


@dataclass
class UpliftPrediction:
    """All quantities the policy needs, aligned to the input rows."""

    primitive_mean: np.ndarray  # (n, 7) P(recover | primitive action)
    primitive_std: np.ndarray  # (n, 7) ensemble std (zeros for a single model)
    uplift: np.ndarray  # (n, 5) per arm, best delay used for retry_delayed
    best_delay: np.ndarray  # (n,) delay in days chosen for retry_delayed
    base: np.ndarray  # (n,) P(recover | no_action)


class TLearner:
    """One model per primitive action; uplift = mu_a(x) - mu_0(x)."""

    def __init__(self, cfg: UpliftConfig | None = None) -> None:
        self.cfg = cfg or UpliftConfig()
        self.models: dict[int, list[lgb.LGBMClassifier]] = {}
        self.feature_names: list[str] = []
        self.train_sizes: dict[int, int] = {}

    # ---- fit ---------------------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        action_idx: np.ndarray,
        y: np.ndarray,
        propensity: np.ndarray,
    ) -> TLearner:
        """Fit per-action models on logged rows, weighting each row by 1 / propensity."""
        self.feature_names = list(X.columns)
        y = np.asarray(y).astype(int)
        w_all = 1.0 / np.asarray(propensity, dtype=float)
        rng = np.random.default_rng(self.cfg.seed)
        for a in range(N_PRIMITIVES):
            mask = np.asarray(action_idx) == a
            Xa, ya, wa = X[mask], y[mask], w_all[mask]
            wa = wa / wa.mean()
            self.train_sizes[a] = int(mask.sum())
            members: list[lgb.LGBMClassifier] = []
            for e in range(self.cfg.n_ensemble):
                if self.cfg.n_ensemble > 1:
                    idx = rng.integers(0, len(Xa), len(Xa))
                    Xe, ye, we = Xa.iloc[idx], ya[idx], wa[idx]
                else:
                    Xe, ye, we = Xa, ya, wa
                m = lgb.LGBMClassifier(**self.cfg.lgb_params(self.cfg.seed + 97 * e + a))
                m.fit(Xe, ye, sample_weight=we, categorical_feature="auto")
                members.append(m)
            self.models[a] = members
        return self

    # ---- predict -------------------------------------------------------------------------------
    def predict_primitives(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(mean, std) of P(recover) per primitive action, shape (n, 7) each."""
        X = X[self.feature_names]
        n = len(X)
        mean = np.zeros((n, N_PRIMITIVES))
        std = np.zeros((n, N_PRIMITIVES))
        for a, members in self.models.items():
            preds = np.stack([m.predict_proba(X)[:, 1] for m in members], axis=1)
            mean[:, a] = preds.mean(axis=1)
            std[:, a] = preds.std(axis=1) if preds.shape[1] > 1 else 0.0
        return mean, std

    def predict(self, X: pd.DataFrame, estimate: str = "mean", z: float = 1.0) -> UpliftPrediction:
        """Uplift per arm. ``estimate="lcb"`` uses mean - z * std of the ensemble uplift."""
        mean, std = self.predict_primitives(X)
        base = mean[:, PRIMITIVE_INDEX[(NO_ACTION, 0)]]
        up_prim = mean - base[:, None]
        if estimate == "lcb":
            # uplift std: ensemble std of the difference approximated by sum of variances
            base_var = std[:, PRIMITIVE_INDEX[(NO_ACTION, 0)]] ** 2
            up_prim = up_prim - z * np.sqrt(std**2 + base_var[:, None])
        elif estimate != "mean":
            raise ValueError("estimate must be 'mean' or 'lcb'")
        uplift = np.zeros((len(X), len(ARMS)))
        best_delay = np.zeros(len(X), dtype=int)
        for (arm, _delay), idx in PRIMITIVE_INDEX.items():
            if arm == RETRY_DELAYED:
                continue
            uplift[:, arm] = up_prim[:, idx]
        delay_cols = np.stack([up_prim[:, DELAY_PRIMITIVES[d]] for d in RETRY_DELAYS], axis=1)
        pick = delay_cols.argmax(axis=1)
        uplift[:, RETRY_DELAYED] = delay_cols[np.arange(len(X)), pick]
        best_delay = np.array(RETRY_DELAYS)[pick]
        uplift[:, NO_ACTION] = 0.0
        return UpliftPrediction(mean, std, uplift, best_delay, base)

    def predict_uplift(self, X: pd.DataFrame, estimate: str = "mean", z: float = 1.0) -> np.ndarray:
        """``(n, 5)`` uplift matrix; the brief's interface."""
        return self.predict(X, estimate=estimate, z=z).uplift

    # ---- persistence ---------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"cfg": asdict(self.cfg), "models": self.models,
                         "feature_names": self.feature_names, "train_sizes": self.train_sizes}, f)

    @classmethod
    def load(cls, path: Path) -> TLearner:
        with open(path, "rb") as f:
            blob = pickle.load(f)  # noqa: S301 - our own artefact
        obj = cls(UpliftConfig(**blob["cfg"]))
        obj.models = blob["models"]
        obj.feature_names = blob["feature_names"]
        obj.train_sizes = blob["train_sizes"]
        return obj
