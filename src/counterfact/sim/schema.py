"""Typed records shared by the simulator, the agent and the evaluator.

The parquet tables are the bulk representation; these models are the per-record contract used
by the agent loop, the API and the tests. Field names match the parquet columns exactly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from counterfact.config import (
    ARMS,
    ESCALATE_HUMAN,
    NO_ACTION,
    REMIND_AND_RETRY,
    RETRY_DELAYED,
    RETRY_DELAYS,
    RETRY_NOW,
)

Segment = Literal["b2c", "b2b"]
PlanCycle = Literal["monthly", "quarterly", "annual"]


class Merchant(BaseModel):
    """A subscription merchant and its policy configuration (all rupee amounts)."""

    merchant_id: str
    name: str
    segment: Segment
    plan_amount: float
    min_ev_threshold: float = 10.0
    escalation_amount_threshold: float = 10_000.0
    message_cost: float = 0.85
    retry_cost: float = 0.5
    ops_cost: float = 150.0
    fatigue_rate: float = 0.01
    ltv_cycles: float = 3.0
    kill_switch: bool = False


class Customer(BaseModel):
    """Observable customer attributes at decision time."""

    customer_id: str
    merchant_id: str
    segment: Segment
    payment_method: str
    bank_code: str
    customer_tenure_months: int
    prior_failures_90d: int
    prior_recoveries_90d: int


class Context(BaseModel):
    """Decision-time context for one ``payment.failed`` event. Only pre-decision information."""

    event_id: str
    customer_id: str
    merchant_id: str
    subscription_id: str
    segment: Segment
    amount: float = Field(gt=0)
    plan_amount: float
    plan_cycle: PlanCycle
    seats: int = 1
    failure_category: str
    failure_source: str
    payment_method: str
    bank_code: str
    attempt_number: int = Field(ge=1, le=3)
    failed_at: datetime
    hour_ist: int = Field(ge=0, le=23)
    dow: int = Field(ge=0, le=6)
    day_of_month: int = Field(ge=1, le=31)
    days_to_payday: int = Field(ge=0, le=31)
    customer_tenure_months: int
    subscription_age_cycles: int
    prior_failures_90d: int
    prior_recoveries_90d: int
    prior_recovery_rate: float
    last_success_days_ago: int
    contacts_last_24h: int
    contacts_last_7d: int
    risk_score: float = Field(ge=0, le=100)
    card_expiry_days: float | None = None


class Plan(BaseModel):
    """What an arm actually does over the outcome window. Bounded by construction."""

    retry_days: tuple[float, ...] = ()
    message: bool = False
    escalate: bool = False

    @property
    def n_retries(self) -> int:
        return len(self.retry_days)


class Intervention(BaseModel):
    """A chosen action: arm id plus the retry delay parameter (0 unless ``retry_delayed``)."""

    arm: int = Field(ge=0, le=4)
    delay_days: int = 0

    @field_validator("delay_days")
    @classmethod
    def _delay_ok(cls, v: int, info) -> int:  # noqa: ANN001
        arm = info.data.get("arm")
        if arm == RETRY_DELAYED and v not in RETRY_DELAYS:
            raise ValueError(f"retry_delayed needs delay_days in {RETRY_DELAYS}, got {v}")
        if arm != RETRY_DELAYED and v != 0:
            raise ValueError("delay_days must be 0 unless arm is retry_delayed")
        return v

    @property
    def name(self) -> str:
        return f"retry_delayed_{self.delay_days}" if self.arm == RETRY_DELAYED else ARMS[self.arm]

    def plan(self) -> Plan:
        return plan_for(self.arm, self.delay_days)


class Outcome(BaseModel):
    """Realised outcome of one intervention over the 14-day window."""

    event_id: str
    recovered: bool
    recovered_amount: float
    contacted: bool
    churned: bool
    escalated: bool


def plan_for(arm: int, delay_days: int = 0) -> Plan:
    """Map an arm (and delay parameter) to its bounded execution plan."""
    if arm == NO_ACTION:
        return Plan()
    if arm == RETRY_NOW:
        return Plan(retry_days=(0.0,))
    if arm == RETRY_DELAYED:
        if delay_days not in RETRY_DELAYS:
            raise ValueError(f"delay_days must be one of {RETRY_DELAYS}")
        return Plan(retry_days=(float(delay_days),))
    if arm == REMIND_AND_RETRY:
        return Plan(retry_days=(1.0,), message=True)
    if arm == ESCALATE_HUMAN:
        return Plan(escalate=True)
    raise ValueError(f"unknown arm {arm}")


RAZORPAY_DEFAULT_PLAN = Plan(retry_days=(1.0, 2.0, 3.0))
"""Razorpay's default behaviour on a failed subscription charge: retries at T+1, T+2, T+3."""
