"""Generate the seeded datasets: ``data/<variant>/{failures,counterfactuals,merchants}.parquet``.

Usage::

    python scripts/make_data.py --all-variants
    python scripts/make_data.py --variant calibrated --n 50000 --seed 42
    python scripts/make_data.py --calibrate        # re-derive the retry scale and print it
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from recoverai.config import SIM_VARIANTS, get_settings
from recoverai.sim.generator import build_dataset, generate_customers, generate_failures
from recoverai.sim.outcome_model import RETRY_SCALE, calibrate_retry_scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default=None, choices=SIM_VARIANTS)
    ap.add_argument("--all-variants", action="store_true")
    ap.add_argument("--n", type=int, default=None, help="number of failed payments")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--calibrate", action="store_true", help="print the calibrated retry scale")
    ap.add_argument("--target", type=float, default=0.60, help="target Razorpay-default recovery")
    args = ap.parse_args()

    settings = get_settings()
    if args.n is not None:
        settings = settings.model_copy(update={"n_failures": args.n})
    if args.seed is not None:
        settings = settings.model_copy(update={"seed": args.seed})

    if args.calibrate:
        rng = np.random.default_rng(settings.seed)
        customers = generate_customers(settings.n_customers, rng)
        obs, hidden = generate_failures(customers, settings.n_failures, rng, "calibrated")
        scale = calibrate_retry_scale(obs, hidden, target=args.target)
        print(f"calibrated retry_scale = {scale:.4f}  (current constant {RETRY_SCALE['calibrated']})")
        return

    variants = list(SIM_VARIANTS) if args.all_variants else [args.variant or settings.sim_variant]
    for v in variants:
        t0 = time.time()
        out = build_dataset(settings, v)
        print(f"[{v}] wrote {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
