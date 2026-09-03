"""Decision-time feature matrix. Leakage-checked.

Rules enforced here and by ``tests/test_no_counterfactual_leak.py`` / ``tests/test_features.py``:

* Only columns known at the moment ``payment.failed`` arrives are features. Logged action /
  outcome columns and identifiers are dropped by name; anything not in ``FEATURES`` is ignored.
* Categorical codes come from fixed vocabularies in :mod:`counterfact.config`, never from data,
  so the code of ``"card_expired"`` is the same at training time and inside the agent.
* ``load_training_frame`` reads ``failures.parquet`` only; the hidden outcome table is off limits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from counterfact.config import CATEGORICAL_VOCAB, Settings, SimVariant

ID_COLS: tuple[str, ...] = ("event_id", "customer_id", "subscription_id", "failed_at")
LOGGED_COLS: tuple[str, ...] = (
    "arm", "delay_days", "action_idx", "action_name", "propensity",
    "recovered", "recovered_amount", "contacted", "churned", "escalated",
)
CATEGORICAL: tuple[str, ...] = tuple(CATEGORICAL_VOCAB)
NUMERIC: tuple[str, ...] = (
    "amount", "plan_amount", "seats", "attempt_number",
    "hour_ist", "dow", "day_of_month", "days_to_payday",
    "customer_tenure_months", "subscription_age_cycles",
    "prior_failures_90d", "prior_recoveries_90d", "prior_recovery_rate",
    "last_success_days_ago", "contacts_last_24h", "contacts_last_7d",
    "risk_score", "card_expiry_days",
)
FEATURES: tuple[str, ...] = CATEGORICAL + NUMERIC


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the feature matrix ``X`` (same row order as ``df``).

    Raises if any logged / outcome column is requested as a feature; missing categorical levels
    become NaN (LightGBM handles them) rather than silently growing the vocabulary.
    """
    forbidden = set(FEATURES) & set(LOGGED_COLS)
    if forbidden:
        raise ValueError(f"logged columns cannot be features: {sorted(forbidden)}")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    X = pd.DataFrame(index=df.index)
    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c].astype(str), categories=list(CATEGORICAL_VOCAB[c]))
    for c in NUMERIC:
        X[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return X


def load_training_frame(settings: Settings, variant: SimVariant) -> pd.DataFrame:
    """Logged failures for training. Never touches the hidden outcome table."""
    path = settings.variant_dir(variant) / "failures.parquet"
    if "counterfactual" in path.name:
        raise PermissionError("training code must not read the counterfactual table")
    return pd.read_parquet(path)


def features_hash(row: pd.Series | dict) -> str:
    """Stable sha256 of the decision-time features of one event, for the audit trail."""
    items = {k: (None if _is_nan(v) else v) for k, v in dict(row).items() if k in FEATURES}
    payload = json.dumps(items, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _is_nan(v: object) -> bool:
    try:
        return bool(np.isnan(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def leakage_report(df: pd.DataFrame, X: pd.DataFrame, target: str = "recovered") -> pd.DataFrame:
    """|corr| of every feature (categoricals one-hot) with the logged outcome, descending."""
    y = df[target].astype(float)
    rows = []
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            for level in X[c].cat.categories:
                ind = (X[c] == level).astype(float)
                if ind.std() > 0:
                    rows.append((f"{c}={level}", abs(np.corrcoef(ind, y)[0, 1])))
        else:
            v = X[c].fillna(X[c].median())
            if v.std() > 0:
                rows.append((c, abs(np.corrcoef(v, y)[0, 1])))
    return pd.DataFrame(rows, columns=["feature", "abs_corr"]).sort_values(
        "abs_corr", ascending=False
    ).reset_index(drop=True)


def save_feature_list(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"categorical": CATEGORICAL, "numeric": NUMERIC}, indent=2))
