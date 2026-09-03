"""Seeded synthetic population: merchants, customers and failed subscription payments.

Two tables come out of :func:`generate_failures`:

* ``obs``    - decision-time observable columns only (what the agent and the models may see)
* ``hidden`` - latent customer state, hidden per-event shocks and the common random numbers
               used to realise coherent counterfactual outcomes. Written to
               ``counterfactuals.parquet`` and never read by ``features/`` or ``models/``.

Nothing here decides outcomes; see :mod:`counterfact.sim.outcome_model`.
"""

from __future__ import annotations

import calendar
import json
from pathlib import Path

import numpy as np
import pandas as pd

from counterfact import config
from counterfact.config import FAILURE_CATEGORIES, FAILURE_MIX, FAILURE_SOURCE, Settings, SimVariant
from counterfact.sim.schema import Merchant

MERCHANTS: tuple[Merchant, ...] = (
    Merchant(merchant_id="m_streambox", name="StreamBox OTT", segment="b2c", plan_amount=299,
             min_ev_threshold=5, escalation_amount_threshold=5_000, ops_cost=120),
    Merchant(merchant_id="m_fitpulse", name="FitPulse", segment="b2c", plan_amount=999,
             min_ev_threshold=10, escalation_amount_threshold=5_000, ops_cost=150),
    Merchant(merchant_id="m_learnloop", name="LearnLoop", segment="b2c", plan_amount=1_499,
             min_ev_threshold=10, escalation_amount_threshold=10_000, ops_cost=150),
    Merchant(merchant_id="m_clouddesk", name="CloudDesk", segment="b2b", plan_amount=4_999,
             min_ev_threshold=25, escalation_amount_threshold=40_000, ops_cost=250),
    Merchant(merchant_id="m_scaleops", name="ScaleOps", segment="b2b", plan_amount=15_000,
             min_ev_threshold=50, escalation_amount_threshold=100_000, ops_cost=400),
)
CUSTOMER_SHARE = np.array([0.36, 0.22, 0.18, 0.14, 0.10])
BANKS = config.BANKS
BANK_WEIGHTS = np.array([0.22, 0.20, 0.18, 0.12, 0.08, 0.05, 0.05, 0.10])
METHODS_B2C = (("card", 0.45), ("upi_autopay", 0.40), ("emandate", 0.05), ("wallet", 0.10))
METHODS_B2B = (("card", 0.35), ("upi_autopay", 0.15), ("emandate", 0.45), ("wallet", 0.05))
WINDOW_START = pd.Timestamp("2025-01-01")
WINDOW_DAYS = 540

UNIFORM_COLS = ("u_hard", "u_self", "u_msg", "u_churn", "u_esc", "u_att1", "u_att2", "u_att3")


def merchants_frame() -> pd.DataFrame:
    """Merchant configuration as a DataFrame (5 rows)."""
    return pd.DataFrame([m.model_dump() for m in MERCHANTS])


def _sample_methods(rng: np.random.Generator, segment: np.ndarray) -> np.ndarray:
    out = np.empty(len(segment), dtype=object)
    for seg, table in (("b2c", METHODS_B2C), ("b2b", METHODS_B2B)):
        mask = segment == seg
        names = [n for n, _ in table]
        probs = np.array([p for _, p in table])
        out[mask] = rng.choice(names, size=mask.sum(), p=probs)
    return out


def generate_customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Customers with observable attributes and hidden latent state."""
    m_idx = rng.choice(len(MERCHANTS), size=n, p=CUSTOMER_SHARE)
    segment = np.array([MERCHANTS[i].segment for i in m_idx], dtype=object)
    tenure = np.clip(rng.exponential(12, n).astype(int) + 1, 1, 60)
    z_liq = rng.normal(0, 1, n) + np.where(segment == "b2b", 0.3, 0.0)
    z_eng = rng.normal(0.3 * (np.log1p(tenure) / np.log(61) - 0.5), 1.0, n)
    churn_intent = rng.beta(2, 5, n)
    hidden_segment = rng.random(n) < 0.4
    return pd.DataFrame(
        {
            "customer_id": [f"cust_{i:05d}" for i in range(n)],
            "merchant_id": [MERCHANTS[i].merchant_id for i in m_idx],
            "segment": segment,
            "payment_method": _sample_methods(rng, segment),
            "bank_code": rng.choice(BANKS, size=n, p=BANK_WEIGHTS),
            "customer_tenure_months": tenure,
            "z_liquidity": z_liq,
            "z_engagement": z_eng,
            "churn_intent": churn_intent,
            "hidden_segment": hidden_segment,
        }
    )


def _days_to_payday(ts: pd.Series, payday: int = 1) -> np.ndarray:
    """Days until the next occurrence of ``payday`` (day of month); 0 if today."""
    day = ts.dt.day.to_numpy()
    month = ts.dt.month.to_numpy()
    year = ts.dt.year.to_numpy()
    dim = np.array([calendar.monthrange(y, m)[1] for y, m in zip(year, month, strict=True)])
    return np.where(day <= payday, payday - day, dim - day + payday)


def generate_failures(
    customers: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    variant: SimVariant = "calibrated",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``n`` failed payments drawn over the customer base.

    Returns ``(obs, hidden)`` aligned by row and by ``event_id``.
    """
    merch = {m.merchant_id: m for m in MERCHANTS}
    # customers with low liquidity fail more often
    w = np.exp(-0.6 * customers["z_liquidity"].to_numpy())
    c_idx = rng.choice(len(customers), size=n, p=w / w.sum())
    cust = customers.iloc[c_idx].reset_index(drop=True)
    segment = cust["segment"].to_numpy()
    is_b2b = segment == "b2b"
    plan_amount = cust["merchant_id"].map(lambda m: merch[m].plan_amount).to_numpy(dtype=float)

    # --- timing -----------------------------------------------------------------------------
    offset_days = rng.integers(0, WINDOW_DAYS, n)
    early = rng.random(n) < 0.6  # autopay debits are attempted early morning IST
    hour = np.where(early, rng.integers(1, 9, n), rng.integers(9, 23, n))
    minute = rng.integers(0, 60, n)
    failed_at = pd.Series(
        WINDOW_START
        + pd.to_timedelta(offset_days, unit="D")
        + pd.to_timedelta(hour * 60 + minute, unit="m")
    )

    # --- failure category and instrument -----------------------------------------------------
    cats = np.array(FAILURE_CATEGORIES)
    probs = np.array([FAILURE_MIX[c] for c in cats])
    category = rng.choice(cats, size=n, p=probs)
    method = cust["payment_method"].to_numpy().astype(object).copy()
    method[(category == "card_expired") & (method != "card")] = "card"
    mandate_ok = np.isin(method, ["upi_autopay", "emandate"])
    method[(category == "mandate_failed") & ~mandate_ok] = "upi_autopay"

    # --- amount --------------------------------------------------------------------------------
    seats = np.where(is_b2b, np.clip(1 + rng.poisson(1.5, n), 1, 6), 1)
    cycle_choice = rng.random(n)
    cycle = np.full(n, "monthly", dtype=object)
    cycle[(~is_b2b) & (cycle_choice > 0.80)] = "quarterly"
    cycle[(~is_b2b) & (cycle_choice > 0.95)] = "annual"
    cycle[is_b2b & (cycle_choice > 0.85)] = "quarterly"
    cycle_mult = np.select(
        [cycle == "monthly", cycle == "quarterly", cycle == "annual"], [1.0, 2.7, 9.6], 1.0
    )
    amount = np.round(plan_amount * seats * cycle_mult, 0)

    # --- history and contact load --------------------------------------------------------------
    z_liq = cust["z_liquidity"].to_numpy()
    z_eng = cust["z_engagement"].to_numpy()
    tenure = cust["customer_tenure_months"].to_numpy()
    prior_fail = rng.poisson(np.exp(0.2 - 0.5 * z_liq), n)
    p_rec = 1 / (1 + np.exp(-(0.3 + 0.5 * z_eng)))
    prior_rec = rng.binomial(prior_fail, p_rec)
    prior_rate = np.where(prior_fail > 0, prior_rec / np.maximum(prior_fail, 1), 0.5)
    contacts_24h = (rng.random(n) < 0.15).astype(int)
    contacts_7d = np.minimum(contacts_24h + rng.poisson(0.3 + 0.3 * prior_fail, n), 6)
    attempt = rng.choice([1, 2, 3], size=n, p=[0.80, 0.15, 0.05])
    sub_age = np.minimum(tenure, 1 + rng.poisson(6, n))
    last_success = np.clip(rng.gamma(2.0, 20.0, n).astype(int), 0, 365)
    risk = np.where(
        category == "risk_declined", rng.beta(6, 3, n) * 100, rng.beta(2, 8, n) * 100
    )
    card_expiry = np.full(n, np.nan)
    is_card = method == "card"
    expired = is_card & (category == "card_expired")
    valid = is_card & (category != "card_expired")
    card_expiry[expired] = -rng.integers(1, 60, n)[expired]
    card_expiry[valid] = rng.integers(30, 900, n)[valid]

    sub_ids = [
        f"sub_{c[5:]}_{s}" for c, s in zip(cust["customer_id"], sub_age, strict=True)
    ]
    obs = pd.DataFrame(
        {
            "event_id": [f"evt_{i:06d}" for i in range(n)],
            "customer_id": cust["customer_id"].to_numpy(),
            "merchant_id": cust["merchant_id"].to_numpy(),
            "subscription_id": sub_ids,
            "segment": segment,
            "amount": amount,
            "plan_amount": plan_amount,
            "plan_cycle": cycle,
            "seats": seats,
            "failure_category": category,
            "failure_source": [FAILURE_SOURCE[c] for c in category],
            "payment_method": method,
            "bank_code": cust["bank_code"].to_numpy(),
            "attempt_number": attempt,
            "failed_at": failed_at,
            "hour_ist": hour,
            "dow": failed_at.dt.dayofweek.to_numpy(),
            "day_of_month": failed_at.dt.day.to_numpy(),
            "days_to_payday": _days_to_payday(failed_at, 1),
            "customer_tenure_months": tenure,
            "subscription_age_cycles": sub_age,
            "prior_failures_90d": prior_fail,
            "prior_recoveries_90d": prior_rec,
            "prior_recovery_rate": prior_rate,
            "last_success_days_ago": last_success,
            "contacts_last_24h": contacts_24h,
            "contacts_last_7d": contacts_7d,
            "risk_score": np.round(risk, 1),
            "card_expiry_days": card_expiry,
        }
    )

    # --- hidden state and common random numbers -------------------------------------------------
    outage = rng.lognormal(np.log(4.0), 1.0, n)
    if variant == "misspecified":
        # outages cluster by (bank, calendar day): whole cohorts fail together, unobserved
        key = (
            pd.Series(cust["bank_code"].to_numpy()).astype(str)
            + "|"
            + failed_at.dt.date.astype(str)
        )
        codes, uniq = pd.factorize(key)
        cluster_draw = rng.lognormal(np.log(4.0), 1.3, len(uniq))
        outage = cluster_draw[codes]
    hidden = pd.DataFrame(
        {
            "event_id": obs["event_id"].to_numpy(),
            "z_liquidity": z_liq,
            "z_engagement": z_eng,
            "churn_intent": cust["churn_intent"].to_numpy(),
            "hidden_segment": cust["hidden_segment"].to_numpy(),
            "outage_hours": outage,
            "days_to_payday_hidden": _days_to_payday(failed_at, 7),
        }
    )
    for col in UNIFORM_COLS:
        hidden[col] = rng.random(n)
    return obs, hidden


def build_dataset(
    settings: Settings,
    variant: SimVariant,
    out_dir: Path | None = None,
    overrides: dict[str, float] | None = None,
) -> Path:
    """Generate, label with the true outcome process, log a randomized policy, write parquet.

    Writes ``failures.parquet`` (observable + logged action/outcome), ``counterfactuals.parquet``
    (hidden state, uniforms, outcome under every action), ``merchants.parquet`` and ``meta.json``.
    ``overrides`` are simulator sensitivity knobs (see ``outcome_model.SENSITIVITY_KEYS``).
    """
    from counterfact.sim.logging_policy import EPSILON, log_actions
    from counterfact.sim.outcome_model import OutcomeModel

    out_dir = out_dir or settings.variant_dir(variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Same population and features for every variant: only the hidden outcome process differs.
    rng = np.random.default_rng(settings.seed)
    customers = generate_customers(settings.n_customers, rng)
    obs, hidden = generate_failures(customers, settings.n_failures, rng, variant)

    model = OutcomeModel(variant, overrides=overrides)
    cf = model.counterfactual_table(obs, hidden)
    logged = log_actions(obs, cf, np.random.default_rng(settings.seed + 1))
    failures = pd.concat([obs, logged], axis=1)
    counterfactuals = pd.concat([hidden, cf], axis=1)

    failures.to_parquet(out_dir / "failures.parquet", index=False)
    counterfactuals.to_parquet(out_dir / "counterfactuals.parquet", index=False)
    merchants_frame().to_parquet(out_dir / "merchants.parquet", index=False)
    meta = {
        "variant": variant,
        "seed": settings.seed,
        "n_failures": int(len(failures)),
        "n_customers": int(len(customers)),
        "retry_scale": model.retry_scale,
        "overrides": dict(overrides or {}),
        "arms": list(config.ARMS),
        "logging_epsilon": EPSILON,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir
