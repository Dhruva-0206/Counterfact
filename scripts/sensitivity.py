"""Sensitivity of the headline to every simulator assumption that can move it (ADR-008).

For each setting the calibrated world is regenerated with the override, the models are retrained
on its logged data, and the paired-exact headline (Rs incremental vs the control per 1k) is
recomputed for ``ml_policy``, the guardrailed ``heuristic`` and the ``oracle``. The last column
states whether the qualitative ranking ``oracle >= ml_policy > razorpay_default`` holds.

Usage::

    python scripts/sensitivity.py [--n-ensemble 10] [--quick]

Writes ``reports/tables/sensitivity.csv`` and prints a markdown table. ~1 min per setting.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from counterfact.config import POLICY_ESTIMATE, POLICY_N_ENSEMBLE, POLICY_Z, ROOT, get_settings
from counterfact.eval.pipeline import evaluate_variant, train_variant
from counterfact.eval.report import to_markdown
from counterfact.sim.generator import build_dataset

# (setting, value, simulator overrides, policy-side kwargs)
GRID: list[tuple[str, str, dict[str, float], dict[str, object]]] = [
    ("default", "-", {}, {}),
    ("control", "razorpay_t123 (literal T+1/T+2/T+3)", {}, {"control": "razorpay_t123"}),
    ("esc_level", "0.20", {"esc_level": 0.20}, {}),
    ("esc_level", "0.35", {"esc_level": 0.35}, {}),
    ("esc_level", "0.60", {"esc_level": 0.60}, {}),
    ("retry_scale_mult", "0.8", {"retry_scale_mult": 0.8}, {}),
    ("retry_scale_mult", "1.2", {"retry_scale_mult": 1.2}, {}),
    ("self_mult", "0.7", {"self_mult": 0.7}, {}),
    ("self_mult", "1.3", {"self_mult": 1.3}, {}),
    ("msg_lift_mult", "0.5", {"msg_lift_mult": 0.5}, {}),
    ("msg_lift_mult", "1.5", {"msg_lift_mult": 1.5}, {}),
    ("churn_mult", "0.5", {"churn_mult": 0.5}, {}),
    ("churn_mult", "2.0", {"churn_mult": 2.0}, {}),
    ("decay", "0.70", {"decay": 0.70}, {}),
    ("decay", "0.95", {"decay": 0.95}, {}),
    ("payday_mult", "0.0", {"payday_mult": 0.0}, {}),
    ("payday_mult", "2.0", {"payday_mult": 2.0}, {}),
    ("ops_cost_mult", "0.5", {}, {"ops_cost_mult": 0.5}),
    ("ops_cost_mult", "2.0", {}, {"ops_cost_mult": 2.0}),
]


def run_setting(name: str, overrides: dict[str, float], policy_kw: dict[str, object],
                n_ensemble: int, n_boot: int) -> dict[str, object]:
    base = get_settings()
    settings = base.model_copy(update={"data_dir": ROOT / "data" / "sensitivity" / name})
    build_dataset(settings, "calibrated", overrides=overrides)
    model = train_variant(settings, "calibrated", n_ensemble, base.seed, verbose=False)
    tables = evaluate_variant(settings, "calibrated", POLICY_ESTIMATE, POLICY_Z, n_boot, model=model,
                              control=str(policy_kw.get("control", "razorpay_default")),
                              ops_cost_mult=float(policy_kw.get("ops_cost_mult", 1.0)))
    p = tables["paired"].set_index("policy")
    ctrl = str(policy_kw.get("control", "razorpay_default"))
    ml, heur, orc = (p.loc[k, "paired_exact_per_1k"] for k in ("ml_policy", "heuristic", "oracle"))
    return {
        "control_recovery": float(p.loc[ctrl, "recovery_rate"]),
        "ml_incr_per_1k": float(ml),
        "heuristic_incr_per_1k": float(heur),
        "oracle_incr_per_1k": float(orc),
        "ml_recovery_delta": float(p.loc["ml_policy", "paired_recovery_delta"]),
        "ml_abstention": float(p.loc["ml_policy", "no_action_share"]),
        "ml_contacts_per_1k": float(p.loc["ml_policy", "contacts_per_1k"]),
        "heuristic_contacts_per_1k": float(p.loc["heuristic", "contacts_per_1k"]),
        "ranking_holds": bool(orc >= ml > 0),
        "ml_vs_heuristic": "ml > heuristic" if ml > heur else "heuristic > ml",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-ensemble", type=int, default=POLICY_N_ENSEMBLE)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--quick", action="store_true", help="only default + esc_level rows")
    args = ap.parse_args()
    grid = [g for g in GRID if args.quick is False or g[0] in ("default", "esc_level")]
    rows = []
    for setting, value, overrides, policy_kw in grid:
        t0 = time.time()
        name = f"{setting}_{value.split(' ')[0]}".replace(".", "p")
        r = run_setting(name, overrides, policy_kw, args.n_ensemble, args.n_boot)
        rows.append({"setting": setting, "value": value, **r})
        print(f"  {setting}={value}: ml {r['ml_incr_per_1k']:,.0f}  heuristic {r['heuristic_incr_per_1k']:,.0f}  "
              f"oracle {r['oracle_incr_per_1k']:,.0f}  abstention {r['ml_abstention']:.1%}  ({time.time() - t0:.0f}s)")
    t = pd.DataFrame(rows)
    out = get_settings().reports_dir / "tables" / "sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    show = t.copy()
    for c in ("ml_incr_per_1k", "heuristic_incr_per_1k", "oracle_incr_per_1k", "ml_contacts_per_1k", "heuristic_contacts_per_1k"):
        show[c] = show[c].map(lambda x: f"{x:,.0f}")
    show["control_recovery"] = show["control_recovery"].map(lambda x: f"{x:.1%}")
    show["ml_recovery_delta"] = show["ml_recovery_delta"].map(lambda x: f"{x:+.1%}")
    show["ml_abstention"] = show["ml_abstention"].map(lambda x: f"{x:.1%}")
    print("\n" + to_markdown(show))
    print(f"\nranking oracle >= ml > control holds in {int(t['ranking_holds'].sum())}/{len(t)} settings; wrote {out}")


if __name__ == "__main__":
    main()
