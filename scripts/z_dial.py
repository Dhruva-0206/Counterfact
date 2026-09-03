"""The conservatism dial: calibrated lift vs null-uplift abstention as the gate z varies.

Usage::  python scripts/z_dial.py   (after `make train`)

Writes ``reports/tables/z_dial.csv`` and ``reports/figures/z_dial.png``.
"""

from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from counterfact.config import POLICY_Z, SIM_VARIANTS, get_settings  # noqa: E402
from counterfact.eval.pipeline import evaluate_variant  # noqa: E402
from counterfact.eval.report import to_markdown  # noqa: E402

GRID = [("mean", 0.0), ("gated", 1.0), ("gated", 1.5), ("gated", 2.0), ("gated", 2.5), ("gated", 3.0)]


def main() -> None:
    settings = get_settings()
    rows = []
    for estimate, z in GRID:
        rec: dict[str, object] = {"estimate": estimate, "z": z}
        for variant in SIM_VARIANTS:
            p = evaluate_variant(settings, variant, estimate, z, n_boot=50)["paired"].set_index("policy")
            rec[f"{variant}_incr_per_1k"] = float(p.loc["ml_policy", "paired_exact_per_1k"])
            rec[f"{variant}_abstention"] = float(p.loc["ml_policy", "no_action_share"])
            rec[f"{variant}_contacts_per_1k"] = float(p.loc["ml_policy", "contacts_per_1k"])
        rows.append(rec)
        print(f"  {estimate} z={z}: calibrated {rec['calibrated_incr_per_1k']:,.0f}  "
              f"null abstention {rec['null_uplift_abstention']:.1%}")
    t = pd.DataFrame(rows)
    out = settings.reports_dir / "tables" / "z_dial.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)

    fig, ax1 = plt.subplots(figsize=(7, 3.8))
    ax1.plot(t["z"], t["calibrated_incr_per_1k"] / 1e3, "o-", color="#2A9D8F", label="calibrated lift vs Razorpay default (Rs k / 1k)")
    ax1.plot(t["z"], t["misspecified_incr_per_1k"] / 1e3, "s--", color="#264653", label="misspecified lift (Rs k / 1k)")
    ax1.set_xlabel("confidence gate z (0 = point estimate)")
    ax1.set_ylabel("Rs thousand per 1,000 failures")
    ax2 = ax1.twinx()
    ax2.plot(t["z"], t["null_uplift_abstention"] * 100, "^-", color="#E76F51", label="null-uplift abstention (%)")
    ax2.axhline(80, color="#E76F51", lw=0.8, ls=":")
    ax2.set_ylabel("abstention under null uplift (%)")
    ax1.axvline(POLICY_Z, color="#888", lw=0.8, ls="--")
    ax1.text(POLICY_Z + 0.03, ax1.get_ylim()[0], "shipped", fontsize=8, color="#555")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
    ax1.set_title("Conservatism dial: gate on the ensemble lower bound, rank on the mean")
    fig.tight_layout()
    fig.savefig(settings.reports_dir / "figures" / "z_dial.png", dpi=150)
    show = t.copy()
    for c in show.columns:
        if c.endswith("abstention"):
            show[c] = show[c].map(lambda x: f"{x:.1%}")
        elif c.endswith("per_1k"):
            show[c] = show[c].map(lambda x: f"{x:,.0f}")
    print(to_markdown(show))


if __name__ == "__main__":
    main()
