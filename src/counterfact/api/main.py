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
from counterfact.agent.executor import (
    FailureInjector,
    MockExecutor,
    RazorpayExecutor,
    verify_webhook,
)
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
    if settings.executor == "razorpay":
        import razorpay  # type: ignore

        client = razorpay.Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))
        executor = RazorpayExecutor(store, client, injector=injector)
    else:
        executor = MockExecutor(store, failure_rate=settings.executor_failure_rate, seed=settings.seed, injector=injector)
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
        event = payload.get("event", payload)  # accept either a bare Context or {"event": Context}
        try:
            ctx = context_from_row(event)
        except Exception as e:  # noqa: BLE001 - validation error surfaced as 422
            raise HTTPException(status_code=422, detail=str(e)) from e
        return handle_event(get_agent(), ctx)

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


def handle_event(agent: Agent, ctx) -> dict[str, Any]:
    """Process one event and return the audit row (shared by the API and tests)."""
    handled = agent.handle(ctx)
    row = dict(handled.row)
    row["guardrail_checks"] = [c for c in row["guardrail_checks"]]
    return row


app = create_app()
