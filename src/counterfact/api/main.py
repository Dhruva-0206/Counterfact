"""FastAPI surface: ``POST /webhook/payment_failed``, ``GET /decisions``, ``GET /metrics``.

Run with ``uv run uvicorn counterfact.api.main:app --reload``. The agent is built once at startup
from settings (variant, executor, audit directory). A Razorpay webhook is verified with the
configured webhook secret when the ``X-Razorpay-Signature`` header is present; the simulator's
event schema (``Context``) is accepted directly for demos, since test mode cannot emit the
failure taxonomy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from counterfact.agent.audit import AuditStore
from counterfact.agent.executor import FailureInjector, build_executor, verify_webhook
from counterfact.agent.explain import ClaudeExplainer, TemplateExplainer
from counterfact.agent.loop import Agent, context_from_row
from counterfact.config import ROOT, Settings, get_settings
from counterfact.eval.pipeline import load_merchants
from counterfact.models.uplift import TLearner

INJECT_FLAG = ROOT / "data" / "inject_5xx.flag"


def build_agent(settings: Settings, audit_dir: Path | None = None) -> Agent:
    """Assemble model + merchants + executor + audit store from settings."""
    variant = settings.sim_variant
    model = TLearner.load(settings.variant_dir(variant) / "models" / "uplift.pkl")
    merchants = load_merchants(settings, variant)
    store = AuditStore(audit_dir or settings.reports_dir / "audit" / "api")
    injector = FailureInjector(flag_path=INJECT_FLAG)
    executor = build_executor(settings, store, injector=injector)
    return Agent(model, merchants, executor, store)


def create_app(settings: Settings | None = None, agent: Agent | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Counterfact", version="0.1.0")
    state: dict[str, Any] = {"agent": agent, "settings": settings}

    def get_agent() -> Agent:
        if state["agent"] is None:
            state["agent"] = build_agent(settings)
        return state["agent"]

    @app.post("/webhook/payment_failed")
    async def payment_failed(request: Request) -> dict[str, Any]:
        body = await request.body()
        sig = request.headers.get("X-Razorpay-Signature")
        if sig is not None:
            if not settings.razorpay_webhook_secret or not verify_webhook(body, sig, settings.razorpay_webhook_secret):
                raise HTTPException(status_code=401, detail="invalid webhook signature")
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail="invalid JSON") from e
        if isinstance(payload.get("payload"), dict) and "entity" in payload:  # real Razorpay webhook
            event_name = str(payload.get("event", ""))
            if event_name == "payment_link.paid":
                return record_link_payment(get_agent(), payload)
            if event_name not in RAZORPAY_EVENTS_HANDLED:
                return {"ignored": event_name}
            event = razorpay_event_to_context(payload)
        else:
            event = payload.get("event", payload)  # accept either a bare Context or {"event": Context}
        try:
            ctx = context_from_row(event)
        except Exception as e:  # noqa: BLE001 - validation error surfaced as 422
            raise HTTPException(status_code=422, detail=str(e)) from e
        row = handle_event(get_agent(), ctx)
        row["webhook_event"] = payload.get("event") if isinstance(payload, dict) else None
        return row

    @app.get("/decisions")
    def decisions(limit: int = 50) -> list[dict[str, Any]]:
        return get_agent().store.recent(limit=limit)

    @app.get("/decisions/{event_id}")
    def decision(event_id: str) -> dict[str, Any]:
        row = get_agent().store.get(event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="unknown event_id")
        return row

    @app.post("/decisions/{event_id}/explain")
    def explain(event_id: str) -> dict[str, Any]:
        agent = get_agent()
        row = agent.store.get(event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="unknown event_id")
        if row.get("explanation"):
            return {"event_id": event_id, "explanation": row["explanation"], "source": row.get("explanation_source")}
        explainer = state.setdefault("explainer", ClaudeExplainer() if settings.anthropic_api_key else TemplateExplainer())
        text, source = explainer.explain(row)
        agent.store.set_explanation(event_id, text, source)
        return {"event_id": event_id, "explanation": text, "source": source}

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return get_agent().store.metrics()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "variant": settings.sim_variant, "executor": settings.executor}

    app.state.counterfact = state
    return app


RAZORPAY_EVENTS_HANDLED = ("payment.failed", "subscription.pending", "subscription.halted")
"""Events that represent a failed subscription charge; charged/captured are acknowledged only."""

# Razorpay error_reason / error_code -> simulator failure taxonomy. Test mode cannot emit most of
# these (docs/ARCHITECTURE.md); unknown reasons fall back to bank_technical.
RAZORPAY_REASON_MAP: dict[str, str] = {
    "insufficient_funds": "insufficient_funds",
    "insufficient_account_balance": "insufficient_funds",
    "card_declined": "risk_declined",
    "risk_check_failed": "risk_declined",
    "expired_card": "card_expired",
    "card_expired": "card_expired",
    "authentication_failed": "auth_failed",
    "invalid_otp": "auth_failed",
    "otp_failed": "auth_failed",
    "gateway_technical_error": "gateway_5xx",
    "gateway_error": "gateway_5xx",
    "server_error": "gateway_5xx",
    "bank_technical_error": "bank_technical",
    "issuer_declined": "bank_technical",
    "payment_cancelled": "customer_cancelled",
    "mandate_failed": "mandate_failed",
    "mandate_not_active": "mandate_failed",
}


def razorpay_event_to_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Map a Razorpay webhook body to the agent's decision-time event schema.

    Only fields the webhook carries are taken from it (ids, amount, method, error reason, time);
    the behavioural history features default to a first failure with no recent contacts. Merchant
    id comes from ``notes.merchant_id`` on the subscription or payment, else the first merchant.
    """
    from datetime import UTC, datetime, timedelta

    from counterfact.config import FAILURE_SOURCE, MERCHANT_IDS

    body = payload.get("payload", {})
    pay = (body.get("payment") or {}).get("entity") or {}
    sub = (body.get("subscription") or {}).get("entity") or {}
    notes = {**(sub.get("notes") or {}), **(pay.get("notes") or {})}
    reason = str(pay.get("error_reason") or pay.get("error_code") or "").lower()
    category = RAZORPAY_REASON_MAP.get(reason, "bank_technical")
    if not pay and payload.get("event") in ("subscription.pending", "subscription.halted"):
        category = "bank_technical"
    method = str(pay.get("method") or "card")
    method = {"upi": "upi_autopay", "emandate": "emandate", "wallet": "wallet"}.get(method, "card")
    created = int(pay.get("created_at") or sub.get("created_at") or payload.get("created_at") or 0)
    failed_at = (datetime.fromtimestamp(created, tz=UTC) + timedelta(hours=5, minutes=30)).replace(tzinfo=None) if created else datetime.now()
    amount = float(pay.get("amount") or sub.get("plan_amount") or 0) / 100.0 or 299.0
    merchant_id = notes.get("merchant_id") or MERCHANT_IDS[0]
    attempt = 1 + int(sub.get("paid_count") or 0) * 0 + (1 if payload.get("event") == "subscription.halted" else 0)
    return {
        "event_id": str(pay.get("id") or f"{sub.get('id', 'sub')}:{payload.get('event')}:{created}"),
        "customer_id": str(pay.get("customer_id") or sub.get("customer_id") or "cust_unknown"),
        "merchant_id": merchant_id,
        "subscription_id": str(sub.get("id") or pay.get("subscription_id") or notes.get("subscription_id") or ""),
        "segment": "b2b" if merchant_id in ("m_clouddesk", "m_scaleops") else "b2c",
        "amount": amount,
        "plan_amount": amount,
        "plan_cycle": "monthly",
        "seats": 1,
        "failure_category": category,
        "failure_source": FAILURE_SOURCE[category],
        "payment_method": method,
        "bank_code": "OTHER",
        "attempt_number": min(attempt, 3),
        "failed_at": failed_at,
        "hour_ist": failed_at.hour,
        "dow": failed_at.weekday(),
        "day_of_month": failed_at.day,
        "days_to_payday": (32 - failed_at.day) % 31 if failed_at.day > 1 else 0,
        "customer_tenure_months": 1,
        "subscription_age_cycles": int(sub.get("paid_count") or 0) + 1,
        "prior_failures_90d": 0,
        "prior_recoveries_90d": 0,
        "prior_recovery_rate": 0.5,
        "last_success_days_ago": 30,
        "contacts_last_24h": 0,
        "contacts_last_7d": 0,
        "risk_score": 10.0,
        "card_expiry_days": (-1.0 if category == "card_expired" else 365.0) if method == "card" else None,
        "invoice_id": pay.get("invoice_id") or sub.get("current_invoice_id"),
        "token_id": pay.get("token_id"),
        "order_id": pay.get("order_id"),
    }


def record_link_payment(agent: Agent, payload: dict[str, Any]) -> dict[str, Any]:
    """``payment_link.paid``: the link's ``reference_id`` is our idempotency key -> recovery outcome."""
    body = payload.get("payload", {})
    link = (body.get("payment_link") or {}).get("entity") or {}
    pay = (body.get("payment") or {}).get("entity") or {}
    key = str(link.get("reference_id") or (pay.get("notes") or {}).get("idempotency_key") or "")
    row = agent.store.find_by_key(key) if key else None
    if row is None:
        return {"ignored": "payment_link.paid", "reason": "no decision with this reference_id", "reference_id": key}
    amount = float(pay.get("amount") or link.get("amount_paid") or link.get("amount") or 0) / 100.0
    agent.store.set_outcome(row["event_id"], {
        "recovered": True, "recovered_amount": amount, "source": "payment_link.paid",
        "payment_id": pay.get("id"), "plink_id": link.get("id"), "reference_id": key,
    })
    return {"outcome_recorded": row["event_id"], "reference_id": key, "recovered_amount": amount,
            "payment_id": pay.get("id"), "plink_id": link.get("id")}


def handle_event(agent: Agent, ctx) -> dict[str, Any]:
    """Process one event and return the audit row (shared by the API and tests)."""
    handled = agent.handle(ctx)
    row = dict(handled.row)
    row["guardrail_checks"] = [c for c in row["guardrail_checks"]]
    return row


app = create_app()
