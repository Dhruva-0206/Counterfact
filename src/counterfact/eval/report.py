"""Tables and figures for the evaluator. Reads the counterfactual table as ground truth.

This module is the *oracle side* of the project: it may read ``counterfactuals.parquet`` because
it measures policies; it never trains anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from counterfact.config import SIM_VARIANTS, Settings, SimVariant, get_settings
from counterfact.policy.baselines import BASELINES, Policy

PER = 1_000


def load_variant(settings: Settings, variant: SimVariant) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ``(failures, counterfactuals)`` for a variant."""
    d = settings.variant_dir(variant)
    return pd.read_parquet(d / "failures.parquet"), pd.read_parquet(d / "counterfactuals.parquet")


def realised(cf: pd.DataFrame, actions: np.ndarray, prefix: str = "y_") -> np.ndarray:
    """Pick the counterfactual column named by ``actions`` for every row."""
    names = pd.unique(actions)
    out = np.zeros(len(actions), dtype=float)
    for name in names:
        mask = actions == name
        out[mask] = cf.loc[mask, f"{prefix}{name}"].to_numpy(dtype=float)
    return out


def policy_metrics(df: pd.DataFrame, cf: pd.DataFrame, actions: np.ndarray) -> dict[str, float]:
    """Raw recovery rate, rupees recovered, contacts, escalations, wasted contacts per 1,000."""
    y = realised(cf, actions)
    amount = df["amount"].to_numpy(dtype=float)
    y0 = cf["y_no_action"].to_numpy(dtype=float)
    contacted = actions == "remind_and_retry"
    escalated = actions == "escalate_human"
    intervened = actions != "no_action"
    n = len(df)
    return {
        "recovery_rate": float(y.mean()),
        "rs_at_risk_per_1k": float(amount.sum() / n * PER),
        "rs_recovered_per_1k": float((y * amount).sum() / n * PER),
        "rs_incremental_vs_no_action_per_1k": float(((y - y0) * amount).sum() / n * PER),
        "contacts_per_1k": float(contacted.sum() / n * PER),
        "escalations_per_1k": float(escalated.sum() / n * PER),
        "wasted_contacts_per_1k": float((contacted & (y0 > 0)).sum() / n * PER),
        "wasted_interventions_per_1k": float((intervened & (y0 > 0)).sum() / n * PER),
        "no_action_share": float((~intervened).mean()),
    }


def baseline_table(
    settings: Settings | None = None,
    policies: dict[str, Policy] | None = None,
    variants: tuple[SimVariant, ...] = SIM_VARIANTS,
) -> pd.DataFrame:
    """Metrics for each policy under each simulator variant (Checkpoint 1 table)."""
    settings = settings or get_settings()
    policies = policies or BASELINES
    rows = []
    for v in variants:
        df, cf = load_variant(settings, v)
        for name, pol in policies.items():
            m = policy_metrics(df, cf, pol(df))
            rows.append({"variant": v, "policy": name, **m})
    return pd.DataFrame(rows)


def format_table(t: pd.DataFrame) -> str:
    """Compact markdown rendering of :func:`baseline_table` output."""
    cols = {
        "variant": "variant",
        "policy": "policy",
        "recovery_rate": "raw recovery",
        "rs_recovered_per_1k": "Rs recovered /1k",
        "rs_incremental_vs_no_action_per_1k": "Rs incr. vs no_action /1k",
        "contacts_per_1k": "contacts /1k",
        "wasted_contacts_per_1k": "wasted contacts /1k",
        "escalations_per_1k": "escalations /1k",
    }
    out = t[list(cols)].rename(columns=cols).copy()
    out["raw recovery"] = out["raw recovery"].map(lambda x: f"{x:.1%}")
    for c in list(cols.values())[3:]:
        out[c] = out[c].map(lambda x: f"{x:,.0f}")
    return to_markdown(out)


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal GitHub-markdown table renderer (no ``tabulate`` dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def write_table(t: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(path, index=False)
