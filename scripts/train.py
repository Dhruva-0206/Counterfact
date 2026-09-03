"""Fit the uplift models per simulator variant on logged data only.

Usage::

    python scripts/train.py --all-variants            # 10-member bootstrap ensembles (default)
    python scripts/train.py --variant calibrated --n-ensemble 1

Writes ``data/<variant>/models/uplift.pkl`` and ``split.json`` (holdout event ids). Prints the
per-action training sizes and holdout AUC per action as a sanity check; the counterfactual table
is never opened here.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from counterfact.config import POLICY_N_ENSEMBLE, SIM_VARIANTS, get_settings
from counterfact.eval.pipeline import train_variant


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default=None, choices=SIM_VARIANTS)
    ap.add_argument("--all-variants", action="store_true")
    ap.add_argument("--n-ensemble", type=int, default=POLICY_N_ENSEMBLE)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    settings = get_settings()
    seed = args.seed if args.seed is not None else settings.seed
    variants = list(SIM_VARIANTS) if args.all_variants else [args.variant or settings.sim_variant]
    for v in variants:
        t0 = time.time()
        print(f"[{v}] training 7 x {args.n_ensemble} models ...")
        train_variant(settings, v, args.n_ensemble, seed)
        print(f"[{v}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
