"""End-to-end pipeline for one simulator variant: train -> policy -> guardrails -> A/B.

Shared by ``scripts/train.py``, ``scripts/evaluate.py``, ``scripts/sensitivity.py`` and
``scripts/z_dial.py``. Evaluator side (``evaluate_variant``) reads the counterfactual table;
``train_variant`` reads logged data only.

Evaluation design (ADR-013):

* **Two-arm randomized A/B** on the holdout: ``ml_policy`` vs the control, stratified by
  merchant x failure category, bootstrap CI on the rupee difference. This is what a production
  experiment would measure.
* **Paired exact** comparison of every policy (ML, guardrailed heuristic, raw heuristic,
  no_action, oracle) against the control on all holdout rows, using the stored counterfactuals.
  No assignment noise; this is the simulator's ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from counterfact.config import (
    POLICY_ESTIMATE,
    POLICY_Z,
    PRIMITIVE_ACTIONS,
    Settings,
    SimVariant,
    primitive_name,
)
from counterfact.eval.ab import ab_test, paired_table, per_merchant, stratified_split
from counterfact.eval.report import PER, load_variant, realised
from counterfact.features.build import build_features, leakage_report, load_training_frame
from counterfact.models.uplift import TLearner, UpliftConfig
from counterfact.policy.baselines import (
    heuristic_actions,
    no_action_policy,
    razorpay_default_policy,
)
from counterfact.policy.ev import Decision, MLPolicy
from counterfact.policy.guardrails import apply_guardrails_frame, guardrailed_actions
from counterfact.sim.schema import Merchant

CONTROL = "razorpay_default"
HOLDOUT_FRAC = 0.4  # 20k of 50k


# ---- training (logged data only) --------------------------------------------------------------
def train_variant(
    settings: Settings, variant: SimVariant, n_ensemble: int, seed: int, verbose: bool = True
) -> TLearner:
    """Fit the T-learner on the stratified training split; write model + holdout ids."""
    from sklearn.metrics import roc_auc_score

    df = load_training_frame(settings, variant)
    train_pos, hold_pos = stratified_split(df, HOLDOUT_FRAC, seed)
    X = build_features(df)
    rep = leakage_report(df.iloc[train_pos], X.iloc[train_pos])
    if (rep["abs_corr"] > 0.99).any():
        raise RuntimeError(f"leakage suspected: {rep.head(3).to_dict('records')}")
    model = TLearner(UpliftConfig(n_ensemble=n_ensemble, seed=seed)).fit(
        X.iloc[train_pos],
        df["action_idx"].to_numpy()[train_pos],
        df["recovered"].to_numpy()[train_pos],
        df["propensity"].to_numpy()[train_pos],
    )
    out_dir = settings.variant_dir(variant) / "models"
    model.save(out_dir / "uplift.pkl")
    (out_dir / "split.json").write_text(
        json.dumps({"seed": seed, "holdout_frac": HOLDOUT_FRAC,
                    "holdout_event_ids": df["event_id"].iloc[hold_pos].tolist()})
    )
    if verbose:
        mean, _ = model.predict_primitives(X.iloc[hold_pos])
        hold = df.iloc[hold_pos]
        for a, (arm, d) in enumerate(PRIMITIVE_ACTIONS):
            mask = (hold["action_idx"] == a).to_numpy()
            y = hold["recovered"].to_numpy()[mask]
            auc = roc_auc_score(y, mean[mask, a]) if 0 < y.mean() < 1 else float("nan")
            print(f"{primitive_name(arm, d):>18s}  train n={model.train_sizes[a]:5d}  holdout AUC={auc:.3f}")
        print(f"  top feature/outcome |corr| on train: {rep.iloc[0]['feature']} = {rep.iloc[0]['abs_corr']:.3f}")
    return model


# ---- evaluation (reads counterfactuals) --------------------------------------------------------
def load_merchants(settings: Settings, variant: SimVariant, ops_cost_mult: float = 1.0) -> dict[str, Merchant]:
    t = pd.read_parquet(settings.variant_dir(variant) / "merchants.parquet")
    out = {}
    for r in t.to_dict("records"):
        r["ops_cost"] = float(r["ops_cost"]) * ops_cost_mult
        out[r["merchant_id"]] = Merchant(**r)
    return out


def holdout_frame(settings: Settings, variant: SimVariant) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Holdout rows (and aligned counterfactuals) from the split written by ``train_variant``."""
    df, cf = load_variant(settings, variant)
    split = json.loads((settings.variant_dir(variant) / "models" / "split.json").read_text())
    mask = df["event_id"].isin(set(split["holdout_event_ids"])).to_numpy()
    return df[mask].reset_index(drop=True), cf[mask].reset_index(drop=True)


@dataclass
class PolicyRun:
    actions: np.ndarray  # final action names after guardrails
    decision: Decision  # pre-guardrail policy output
    guard: pd.DataFrame  # guardrail frame (arm, overridden, rejection codes, ...)


def run_ml_policy(
    df: pd.DataFrame,
    model: TLearner,
    merchants: dict[str, Merchant],
    estimate: str = POLICY_ESTIMATE,
    z: float = POLICY_Z,
) -> PolicyRun:
    """Decide with the uplift model, then apply guardrails row by row."""
    X = build_features(df)
    policy = MLPolicy(model, pd.DataFrame([m.model_dump() for m in merchants.values()]), estimate, z)
    dec = policy.decide(df, X)
    guard, _ = apply_guardrails_frame(df, merchants, dec.net_ev, dec.arm, dec.best_delay)
    return PolicyRun(guard["action_name"].to_numpy(dtype=object), dec, guard)


def all_policies(
    df: pd.DataFrame, cf: pd.DataFrame, run: PolicyRun, merchants: dict[str, Merchant]
) -> dict[str, np.ndarray]:
    h_arm, h_delay = heuristic_actions(df)
    raw = np.array([primitive_name(int(a), int(d)) for a, d in zip(h_arm, h_delay, strict=True)], dtype=object)
    return {
        "ml_policy": run.actions,
        CONTROL: razorpay_default_policy(df),
        "razorpay_t123": np.full(len(df), "razorpay_t123", dtype=object),
        "heuristic": guardrailed_actions(df, merchants, h_arm, h_delay),
        "heuristic_raw": raw,
        "no_action": no_action_policy(df),
        "oracle": oracle_actions(df, cf, run),
    }


def evaluate_variant(
    settings: Settings,
    variant: SimVariant,
    estimate: str = POLICY_ESTIMATE,
    z: float = POLICY_Z,
    n_boot: int = 1000,
    model: TLearner | None = None,
    control: str = CONTROL,
    ops_cost_mult: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """A/B + paired + per-merchant + arm-mix + guardrail tables for one variant (holdout split)."""
    df, cf = holdout_frame(settings, variant)
    merchants = load_merchants(settings, variant, ops_cost_mult)
    model = model or TLearner.load(settings.variant_dir(variant) / "models" / "uplift.pkl")
    run = run_ml_policy(df, model, merchants, estimate, z)
    policies = all_policies(df, cf, run, merchants)

    ab = ab_test(df, cf, {"ml_policy": policies["ml_policy"], control: policies[control]},
                 control=control, seed=settings.seed, n_boot=n_boot)
    ab.insert(0, "variant", variant)
    paired = paired_table(df, cf, policies, control=control)
    paired.insert(0, "variant", variant)
    merch = per_merchant(df, cf, policies, control=control)
    merch.insert(0, "variant", variant)

    arm_mix = (
        pd.Series(run.actions).value_counts(normalize=True).rename("share").rename_axis("action").reset_index()
    )
    arm_mix.insert(0, "variant", variant)
    guard_summary = (
        run.guard["rejection_codes"].str.split("|").explode().replace("", np.nan).dropna()
        .value_counts().rename("count").rename_axis("code").reset_index()
    )
    guard_summary.insert(0, "variant", variant)
    guard_summary["per_1k"] = guard_summary["count"] / len(df) * PER
    abst = abstention_table(df, cf, run.actions)
    abst.insert(0, "variant", variant)
    return {"ab": ab, "paired": paired, "per_merchant": merch, "arm_mix": arm_mix,
            "guardrails": guard_summary, "abstention": abst}


def abstention_table(df: pd.DataFrame, cf: pd.DataFrame, actions: np.ndarray) -> pd.DataFrame:
    """What happened on the rows where the policy chose ``no_action``.

    ``self_recovered`` is the "no_action was right" rate; ``razorpay_would_recover`` is what the
    incumbent schedule would have recovered on the same rows; ``foregone_per_1k`` is the paired
    rupee difference on those rows, spread over all rows (so it is comparable to the headline).
    """
    mask = actions == "no_action"
    amount = df["amount"].to_numpy(dtype=float)
    y0 = cf["y_no_action"].to_numpy(dtype=float)
    yr = cf["y_razorpay_default"].to_numpy(dtype=float)
    n = int(mask.sum())
    row = {
        "abstained": n,
        "abstention_share": float(mask.mean()),
        "self_recovered": float(y0[mask].mean()) if n else float("nan"),
        "razorpay_would_recover": float(yr[mask].mean()) if n else float("nan"),
        "mean_amount_abstained": float(amount[mask].mean()) if n else float("nan"),
        "foregone_per_1k": float(((yr - y0) * amount)[mask].sum() / len(df) * PER),
    }
    return pd.DataFrame([row])


def ope_inputs(
    settings: Settings, variant: SimVariant, estimate: str = POLICY_ESTIMATE, z: float = POLICY_Z
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    """Holdout logged rows, aligned counterfactuals, target actions per policy, reward model.

    The reward model ``q_hat`` is the uplift ensemble's mean P(recover) per primitive action,
    multiplied by amount (reward scale = rupees). Trained on the training split only.
    """
    df, cf = holdout_frame(settings, variant)
    merchants = load_merchants(settings, variant)
    model = TLearner.load(settings.variant_dir(variant) / "models" / "uplift.pkl")
    run = run_ml_policy(df, model, merchants, estimate, z)
    policies = all_policies(df, cf, run, merchants)
    targets = {
        "ml_policy": policies["ml_policy"],
        "heuristic": policies["heuristic"],
        "razorpay_default": np.full(len(df), "retry_delayed_1", dtype=object),  # alias (ADR-006)
    }
    mean, _ = model.predict_primitives(build_features(df))
    names = [primitive_name(a, d) for a, d in PRIMITIVE_ACTIONS]
    q_hat = pd.DataFrame(mean, columns=names)
    return df, cf, targets, q_hat


def oracle_actions(df: pd.DataFrame, cf: pd.DataFrame, run: PolicyRun) -> np.ndarray:
    """Best net-EV action per row using the TRUE probabilities (same costs, no guardrails)."""
    amount = df["amount"].to_numpy(dtype=float)
    names = ["no_action", "retry_now", "retry_delayed_1", "retry_delayed_3", "retry_delayed_7",
             "remind_and_retry", "escalate_human"]
    arm_of = {"no_action": 0, "retry_now": 1, "retry_delayed_1": 2, "retry_delayed_3": 2,
              "retry_delayed_7": 2, "remind_and_retry": 3, "escalate_human": 4}
    base = cf["p_no_action"].to_numpy()
    ev = np.stack(
        [(cf[f"p_{n}"].to_numpy() - base) * amount - run.decision.cost[:, arm_of[n]] for n in names], axis=1
    )
    ev[:, 0] = 0.0
    best = ev.argmax(axis=1)
    return np.array([names[b] for b in best], dtype=object)


# ---- tables ---------------------------------------------------------------------------------------
HEADLINE_COLS = {
    "variant": "variant",
    "policy": "policy",
    "paired_exact_per_1k": "Rs incr vs razorpay_default /1k",
    "rs_incremental_vs_no_action_per_1k": "Rs incr vs no_action /1k",
    "paired_recovery_delta": "recovery delta vs razorpay",
    "recovery_rate": "raw recovery",
    "no_action_share": "abstention",
    "contacts_per_1k": "contacts /1k",
    "wasted_contacts_per_1k": "wasted contacts /1k",
    "escalations_per_1k": "escalations /1k",
}


def headline_table(paired_all: pd.DataFrame) -> pd.DataFrame:
    """Checkpoint 2 headline (paired exact): incremental vs razorpay_default first."""
    return paired_all[list(HEADLINE_COLS)].rename(columns=HEADLINE_COLS)


def ab_table(ab_all: pd.DataFrame) -> pd.DataFrame:
    """Two-arm randomized A/B rows (ml_policy only): rupee deltas with 95% CIs."""
    t = ab_all[ab_all["policy"] == "ml_policy"]
    return t[["variant", "n", "incr_vs_control_per_1k", "ci_low", "ci_high"]].rename(
        columns={"n": "n per arm", "incr_vs_control_per_1k": "Rs incr vs razorpay_default /1k (A/B)",
                 "ci_low": "CI low", "ci_high": "CI high"}
    )


PCT_COLS = ("raw recovery", "abstention", "recovery delta vs razorpay", "share", "no_action_share", "recovery_rate")


def fmt(t: pd.DataFrame) -> pd.DataFrame:
    out = t.copy()
    for c in out.columns:
        if c in PCT_COLS:
            spec = "{:+.1%}" if "delta" in c else "{:.1%}"
            out[c] = out[c].map(spec.format)
        elif out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda x: f"{x:,.0f}")
    return out


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out_dir / f"{prefix}_{name}.csv", index=False)


_ = realised
