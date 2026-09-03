"""Checkpoint 2 evaluation: ML policy vs Razorpay default per variant (A/B + paired exact).

Usage::

    python scripts/evaluate.py --all-variants [--estimate mean|lcb --z 1.0] [--n-boot 1000]

Writes ``reports/tables/ab_*.csv`` and ``reports/figures/ab_<variant>.png`` and prints:
headline (paired exact, Rs incremental vs razorpay_default first, vs no_action second,
abstention under every variant), the two-arm randomized A/B with CIs, arm mix, guardrail
rejections and per-merchant tables.
"""

from __future__ import annotations

import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from counterfact.config import POLICY_ESTIMATE, POLICY_Z, SIM_VARIANTS, get_settings  # noqa: E402
from counterfact.eval.pipeline import (  # noqa: E402
    ab_table,
    evaluate_variant,
    fmt,
    headline_table,
    write_tables,
)
from counterfact.eval.report import to_markdown  # noqa: E402


def plot_paired(paired: pd.DataFrame, ab: pd.DataFrame, variant: str, path) -> None:
    order = ["ml_policy", "heuristic", "razorpay_t123", "oracle"]
    t = paired[paired["policy"].isin(order)].set_index("policy").loc[order].reset_index()
    lakh = 1e5
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = list(range(len(t)))
    colors = ["#2A9D8F" if p == "ml_policy" else "#8D99AE" for p in t["policy"]]
    ax.barh(y, t["paired_exact_per_1k"] / lakh, color=colors)
    ml = ab[ab["policy"] == "ml_policy"]
    if len(ml):
        i = int(t.index[t["policy"] == "ml_policy"][0])
        est = ml["incr_vs_control_per_1k"].iloc[0] / lakh
        ax.errorbar([est], [i], xerr=[[est - ml["ci_low"].iloc[0] / lakh], [ml["ci_high"].iloc[0] / lakh - est]],
                    fmt="o", color="#111", capsize=4, label="two-arm randomized A/B, 95% CI")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(["ML policy (gated z=2)", "rule table (guardrailed)", "literal T+1/T+2/T+3", "oracle (true probs)"])
    ax.invert_yaxis()
    ax.axvline(0, color="#888", lw=1)
    ax.set_xlabel("Rs lakh incremental vs razorpay_default per 1,000 failures (paired exact)")
    ax.set_title(f"Holdout, variant = {variant}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default=None, choices=SIM_VARIANTS)
    ap.add_argument("--all-variants", action="store_true")
    ap.add_argument("--estimate", default=POLICY_ESTIMATE, choices=["mean", "lcb", "gated"])
    ap.add_argument("--z", type=float, default=POLICY_Z)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    settings = get_settings()
    variants = list(SIM_VARIANTS) if args.all_variants else [args.variant or settings.sim_variant]

    keys = ("ab", "paired", "per_merchant", "arm_mix", "guardrails", "abstention")
    all_tables: dict[str, list[pd.DataFrame]] = {k: [] for k in keys}
    for v in variants:
        tables = evaluate_variant(settings, v, args.estimate, args.z, args.n_boot)
        for k, t in tables.items():
            all_tables[k].append(t)
        plot_paired(tables["paired"], tables["ab"], v, settings.reports_dir / "figures" / f"ab_{v}.png")
    merged = {k: pd.concat(ts, ignore_index=True) for k, ts in all_tables.items()}
    write_tables(merged, settings.reports_dir / "tables", "ab")

    print(f"## headline (paired exact on holdout; estimate={args.estimate}, z={args.z})")
    print(to_markdown(fmt(headline_table(merged["paired"]))))
    print("\n## two-arm randomized A/B: ml_policy vs razorpay_default (95% bootstrap CI)")
    print(to_markdown(fmt(ab_table(merged["ab"]))))
    print("\n## arm mix (ml_policy)")
    print(to_markdown(fmt(merged["arm_mix"])))
    print("\n## guardrail rejections per 1k (ml_policy)")
    print(to_markdown(fmt(merged["guardrails"])))
    print("\n## abstentions (ml_policy): was no_action right?")
    ab_t = merged["abstention"].copy()
    for c in ("abstention_share", "self_recovered", "razorpay_would_recover"):
        ab_t[c] = ab_t[c].map(lambda x: f"{x:.1%}")
    print(to_markdown(fmt(ab_t)))
    pm = merged["per_merchant"][merged["per_merchant"]["policy"].isin(["ml_policy", "heuristic"])]
    print("\n## per merchant (paired exact, vs razorpay_default)")
    print(to_markdown(fmt(pm)))
    print(f"\nwrote {settings.reports_dir / 'tables'} and {settings.reports_dir / 'figures'}")


if __name__ == "__main__":
    main()
