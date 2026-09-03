"""Checkpoint 3: off-policy estimates vs paired-exact truth vs randomized A/B, per policy.

Usage::  python scripts/ope.py [--variant calibrated | --all-variants] [--n-boot 200]

For each variant and each of ``ml_policy``, ``heuristic`` (guardrailed) and ``razorpay_default``,
IPS / SNIPS / DM / DR are computed on the logged holdout rows only, then compared with:

* the paired-exact truth from the counterfactual table (same rows), and
* a two-arm randomized A/B of that policy vs ``razorpay_default`` (difference with 95% CI).

Reward = recovered rupees (per 1,000 failures) and, separately, recovery rate.
Writes ``reports/tables/ope.csv`` and prints the Checkpoint 3 table.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from counterfact.config import SIM_VARIANTS, get_settings
from counterfact.eval.ab import ab_test
from counterfact.eval.ope import estimate
from counterfact.eval.pipeline import ope_inputs
from counterfact.eval.report import PER, realised, to_markdown


def run_variant(variant: str, n_boot: int) -> pd.DataFrame:
    settings = get_settings()
    df, cf, targets, q_hat = ope_inputs(settings, variant)
    amount = df["amount"].to_numpy(dtype=float)
    logged = df["action_name"].to_numpy().astype(str)
    prop = df["propensity"].to_numpy(dtype=float)
    rows = []
    truth_ctrl = realised(cf, targets["razorpay_default"]) * amount
    for scale_name, reward, q in (
        ("rupees", df["recovered"].to_numpy(dtype=float) * amount, q_hat.mul(amount, axis=0)),
        ("recovery", df["recovered"].to_numpy(dtype=float), q_hat),
    ):
        mult = PER if scale_name == "rupees" else 1.0
        ests = {}
        for name, target in targets.items():
            ests[name] = estimate(name, target.astype(str), logged, prop, reward, q, n_boot=n_boot, seed=settings.seed)
        for name, target in targets.items():
            e = ests[name]
            y = realised(cf, target)
            truth = float((y * amount).mean() if scale_name == "rupees" else y.mean())
            # randomized two-arm A/B of this policy vs razorpay_default (difference, 95% CI)
            ab = ab_test(df, cf, {name: target, "razorpay_default": targets["razorpay_default"]},
                         control="razorpay_default", seed=settings.seed, n_boot=max(100, n_boot))
            r = ab.set_index("policy").loc[name]
            if scale_name == "rupees":
                ab_est, lo, hi = float(r["incr_vs_control_per_1k"]), float(r["ci_low"]), float(r["ci_high"])
            else:  # recovery-rate difference from the same assignment
                ab_est = lo = hi = float("nan")
            dr_diff = (e.dr - ests["razorpay_default"].dr) * mult
            rows.append({
                "variant": variant, "reward": scale_name, "policy": name,
                "n_logged": e.n, "match_rate": e.match_rate, "ess": e.ess,
                "truth": truth * mult,
                "ips": e.ips * mult, "snips": e.snips * mult, "dm": e.dm * mult, "dr": e.dr * mult,
                "ips_se": e.ips_se * mult, "snips_se": e.snips_se * mult, "dr_se": e.dr_se * mult,
                "ips_err_pct": (e.ips * mult - truth * mult) / (truth * mult) * 100,
                "snips_err_pct": (e.snips * mult - truth * mult) / (truth * mult) * 100,
                "dr_err_pct": (e.dr * mult - truth * mult) / (truth * mult) * 100,
                "truth_diff_vs_razorpay": float(((y * amount if scale_name == "rupees" else y) - (truth_ctrl if scale_name == "rupees" else realised(cf, targets["razorpay_default"]))).mean() * mult),
                "dr_diff_vs_razorpay": dr_diff,
                "ab_diff": ab_est, "ab_ci_low": lo, "ab_ci_high": hi,
                "dr_within_ab_ci": bool(lo <= dr_diff <= hi) if scale_name == "rupees" else None,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default=None, choices=SIM_VARIANTS)
    ap.add_argument("--all-variants", action="store_true")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    settings = get_settings()
    variants = list(SIM_VARIANTS) if args.all_variants else [args.variant or settings.sim_variant]
    t = pd.concat([run_variant(v, args.n_boot) for v in variants], ignore_index=True)
    out = settings.reports_dir / "tables" / "ope.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)

    rup = t[t["reward"] == "rupees"].copy()
    show = rup[["variant", "policy", "match_rate", "ess", "truth", "ips", "snips", "dm", "dr",
                "ips_err_pct", "snips_err_pct", "dr_err_pct", "truth_diff_vs_razorpay", "dr_diff_vs_razorpay",
                "ab_diff", "ab_ci_low", "ab_ci_high", "dr_within_ab_ci"]].copy()
    for c in ("truth", "ips", "snips", "dm", "dr", "truth_diff_vs_razorpay", "dr_diff_vs_razorpay", "ab_diff", "ab_ci_low", "ab_ci_high"):
        show[c] = show[c].map(lambda x: f"{x:,.0f}")
    for c in ("ips_err_pct", "snips_err_pct", "dr_err_pct"):
        show[c] = show[c].map(lambda x: f"{x:+.1f}%")
    show["match_rate"] = show["match_rate"].map(lambda x: f"{x:.1%}")
    show["ess"] = show["ess"].map(lambda x: f"{x:,.0f}")
    print("## OPE vs truth, reward = recovered rupees per 1,000 failures (logged holdout rows)")
    print(to_markdown(show))
    rec = t[t["reward"] == "recovery"].copy()
    show2 = rec[["variant", "policy", "truth", "ips", "snips", "dm", "dr", "dr_se"]].copy()
    for c in ("truth", "ips", "snips", "dm", "dr", "dr_se"):
        show2[c] = show2[c].map(lambda x: f"{x:.3f}")
    print("\n## OPE vs truth, reward = recovery rate")
    print(to_markdown(show2))
    ok = rup["dr_within_ab_ci"].fillna(False)
    print(f"\nDR difference vs razorpay_default inside the A/B CI in {int(ok.sum())}/{len(ok)} policy-variant cells; wrote {out}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
