"""Settings, seeds, paths and the fixed action / failure vocabularies.

Everything reproducible starts here: one seed, one variant name, one data directory.
No external settings library; `.env` is parsed by hand to keep the dependency set small.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

SimVariant = Literal["calibrated", "misspecified", "null_uplift", "drifted"]
SIM_VARIANTS: tuple[SimVariant, ...] = ("calibrated", "misspecified", "null_uplift", "drifted")

# ---- Action space: exactly five arms. Do not expand. ---------------------------------------
ARMS: tuple[str, ...] = (
    "no_action",
    "retry_now",
    "retry_delayed",
    "remind_and_retry",
    "escalate_human",
)
ARM_ID: dict[str, int] = {name: i for i, name in enumerate(ARMS)}
NO_ACTION, RETRY_NOW, RETRY_DELAYED, REMIND_AND_RETRY, ESCALATE_HUMAN = range(5)
RETRY_DELAYS: tuple[int, ...] = (1, 3, 7)

# Primitive (arm, delay) actions used by the logging policy and OPE; delay is 0 unless arm 2.
PRIMITIVE_ACTIONS: tuple[tuple[int, int], ...] = (
    (NO_ACTION, 0),
    (RETRY_NOW, 0),
    (RETRY_DELAYED, 1),
    (RETRY_DELAYED, 3),
    (RETRY_DELAYED, 7),
    (REMIND_AND_RETRY, 0),
    (ESCALATE_HUMAN, 0),
)
PRIMITIVE_INDEX: dict[tuple[int, int], int] = {a: i for i, a in enumerate(PRIMITIVE_ACTIONS)}


def primitive_name(arm: int, delay: int = 0) -> str:
    """Human-readable primitive action name, e.g. ``retry_delayed_3``."""
    return f"{ARMS[arm]}_{delay}" if arm == RETRY_DELAYED else ARMS[arm]


# ---- Failure taxonomy (Razorpay-style) ------------------------------------------------------
FAILURE_CATEGORIES: tuple[str, ...] = (
    "insufficient_funds",
    "bank_technical",
    "gateway_5xx",
    "auth_failed",
    "card_expired",
    "risk_declined",
    "customer_cancelled",
    "mandate_failed",
)
FAILURE_MIX: dict[str, float] = {
    "insufficient_funds": 0.30,
    "bank_technical": 0.15,
    "gateway_5xx": 0.10,
    "auth_failed": 0.15,
    "card_expired": 0.10,
    "risk_declined": 0.08,
    "customer_cancelled": 0.07,
    "mandate_failed": 0.05,
}
FAILURE_SOURCE: dict[str, str] = {
    "insufficient_funds": "customer",
    "bank_technical": "bank",
    "gateway_5xx": "gateway",
    "auth_failed": "customer",
    "card_expired": "customer",
    "risk_declined": "razorpay",
    "customer_cancelled": "business",
    "mandate_failed": "bank",
}
PAYMENT_METHODS: tuple[str, ...] = ("card", "upi_autopay", "emandate", "wallet")
FAILURE_SOURCES: tuple[str, ...] = ("customer", "bank", "gateway", "business", "razorpay")
MERCHANT_IDS: tuple[str, ...] = (
    "m_streambox", "m_fitpulse", "m_learnloop", "m_clouddesk", "m_scaleops",
)
SEGMENTS: tuple[str, ...] = ("b2c", "b2b")
PLAN_CYCLES: tuple[str, ...] = ("monthly", "quarterly", "annual")
BANKS: tuple[str, ...] = ("HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "YES", "IDFC", "OTHER")

# Fixed vocabularies for categorical features: category codes must be identical at training
# time and at decision time, so they are never inferred from data.
CATEGORICAL_VOCAB: dict[str, tuple[str, ...]] = {
    "merchant_id": MERCHANT_IDS,
    "segment": SEGMENTS,
    "plan_cycle": PLAN_CYCLES,
    "failure_category": FAILURE_CATEGORIES,
    "failure_source": FAILURE_SOURCES,
    "payment_method": PAYMENT_METHODS,
    "bank_code": BANKS,
}

# Shipped policy estimator (ADR-007): gate on the ensemble lower confidence bound, rank on the
# mean. z=2 reaches >= 80% abstention under null_uplift with 10-member bootstrap ensembles.
POLICY_ESTIMATE = "gated"
POLICY_Z = 2.0
POLICY_N_ENSEMBLE = 10

# Guardrail constants (policy/guardrails.py). Quiet hours are IST, inclusive start, exclusive end.
MAX_RETRIES_PER_FAILURE = 3
MAX_CONTACTS_24H = 1
MAX_CONTACTS_7D = 3
QUIET_HOURS_START = 21
QUIET_HOURS_END = 8


def _load_dotenv(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines from a .env file; never overrides real environment."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        out[key.strip()] = value
    return out


class Settings(BaseModel):
    """Runtime settings. Construct via :func:`get_settings` to read env / .env."""

    seed: int = 42
    sim_variant: SimVariant = "calibrated"
    n_failures: int = 50_000
    n_customers: int = 8_000
    outcome_window_days: int = 14
    executor: Literal["mock", "razorpay"] = "mock"
    executor_failure_rate: float = Field(0.0, ge=0.0, le=1.0)
    razorpay_key_id: str | None = None
    razorpay_key_secret: str | None = None
    razorpay_webhook_secret: str | None = None
    anthropic_api_key: str | None = None
    data_dir: Path = DATA_DIR
    reports_dir: Path = REPORTS_DIR

    def variant_dir(self, variant: SimVariant | None = None) -> Path:
        """Directory holding generated data for a simulator variant."""
        return self.data_dir / (variant or self.sim_variant)


def get_settings(env_file: Path | None = ROOT / ".env") -> Settings:
    """Build :class:`Settings` from ``COUNTERFACT_*`` / credential env vars with .env fallback."""
    env = {**_load_dotenv(env_file), **os.environ} if env_file else dict(os.environ)
    kw: dict[str, object] = {}

    def pick(key: str, field: str) -> None:
        if key in env and env[key] != "":
            kw[field] = env[key]

    pick("COUNTERFACT_SEED", "seed")
    pick("COUNTERFACT_SIM_VARIANT", "sim_variant")
    pick("COUNTERFACT_N_FAILURES", "n_failures")
    pick("COUNTERFACT_N_CUSTOMERS", "n_customers")
    pick("COUNTERFACT_EXECUTOR", "executor")
    pick("COUNTERFACT_EXECUTOR_FAILURE_RATE", "executor_failure_rate")
    pick("RAZORPAY_KEY_ID", "razorpay_key_id")
    pick("RAZORPAY_KEY_SECRET", "razorpay_key_secret")
    pick("RAZORPAY_WEBHOOK_SECRET", "razorpay_webhook_secret")
    pick("ANTHROPIC_API_KEY", "anthropic_api_key")
    pick("COUNTERFACT_DATA_DIR", "data_dir")
    pick("COUNTERFACT_REPORTS_DIR", "reports_dir")
    return Settings(**kw)  # type: ignore[arg-type]
