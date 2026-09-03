"""Checkpoint 1: baseline metrics per simulator variant, from the generated data.

Usage::

    python scripts/checkpoint1.py                 # baselines x variants (after `make data`)
    python scripts/checkpoint1.py --by-category   # true P(recover) by category and action
"""

from __future__ import annotations

import argparse

import pandas as pd

from recoverai.config import FAILURE_CATEGORIES, get_settings
from recoverai.eval.report import (
    baseline_table,
    format_table,
    load_variant,
    to_markdown,
    write_table,
)

ACTIONS = (
    "no_action", "retry_now", "retry_delayed_1", "retry_delayed_7",
    "remind_and_retry", "escalate_human", "razorpay_default",
)


def by_category(variant: str) -> pd.DataFrame:
    """Mean true recovery probability per failure category and action."""
    settings = get_settings()
    df, cf = load_variant(settings, variant)
    g = df["failure_category"]
    share = g.value_counts(normalize=True)
    rows = []
    for cat in FAILURE_CATEGORIES:
        row = {"category": cat, "share": f"{share[cat]:.0%}"}
        row.update({a: f"{cf.loc[g == cat, f'p_{a}'].mean():.2f}" for a in ACTIONS})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--by-category", action="store_true")
    ap.add_argument("--variant", default="calibrated")
    args = ap.parse_args()
    settings = get_settings()
    if args.by_category:
        print(to_markdown(by_category(args.variant)))
        return
    t = baseline_table(settings)
    print(format_table(t))
    out = settings.reports_dir / "tables" / "checkpoint1_baselines.csv"
    write_table(t, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
