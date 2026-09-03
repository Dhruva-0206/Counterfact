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

    # two panels, one axis each (no dual y-axis): lift on the left, abstention on the right
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.plot(t["z"], t["calibrated_incr_per_1k"] / 1e5, "o-", color="#2a78d6", lw=2, ms=6, label="calibrated")
    ax1.plot(t["z"], t["misspecified_incr_per_1k"] / 1e5, "s-", color="#eb6834", lw=2, ms=6, label="misspecified")
    if "drifted_incr_per_1k" in t:
        ax1.plot(t["z"], t["drifted_incr_per_1k"] / 1e5, "^-", color="#1baf7a", lw=2, ms=6, label="drifted")
    ax1.set_xlabel("confidence gate z (0 = point estimate)")
    ax1.set_ylabel("Rs lakh incremental vs Razorpay default per 1,000")
    ax1.set_title("Lift when uplift is real", fontsize=10)
    ax1.legend(fontsize=8, frameon=False)
    ax2.plot(t["z"], t["null_uplift_abstention"] * 100, "o-", color="#2a78d6", lw=2, ms=6)
    ax2.axhline(80, color="#52514e", lw=1, ls=":")
    ax2.text(0.05, 81, "80% bar", fontsize=8, color="#52514e")
    ax2.set_xlabel("confidence gate z (0 = point estimate)")
    ax2.set_ylabel("abstention under null uplift (%)")
    ax2.set_title("Abstention when nothing works", fontsize=10)
    for ax in (ax1, ax2):
        ax.axvline(POLICY_Z, color="#52514e", lw=0.8, ls="--")
        ax.text(POLICY_Z + 0.05, ax.get_ylim()[0], "shipped", fontsize=8, color="#52514e")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    fig.suptitle("Conservatism dial: gate on the ensemble lower bound, rank on the mean", fontsize=11)
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
