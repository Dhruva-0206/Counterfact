"""API handlers called directly (no HTTP client dependency): webhook, decisions, metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from counterfact.agent.audit import AuditStore
from counterfact.agent.executor import MockExecutor, verify_webhook
from counterfact.agent.loop import Agent, context_from_row
from counterfact.api.main import create_app, handle_event
from counterfact.config import Settings
from counterfact.features.build import build_features
from counterfact.models.uplift import TLearner, UpliftConfig
from counterfact.sim.generator import MERCHANTS, generate_customers, generate_failures
from counterfact.sim.logging_policy import log_actions
from counterfact.sim.outcome_model import OutcomeModel


@pytest.fixture(scope="module")
def agent_and_events(tmp_path_factory) -> tuple[Agent, pd.DataFrame]:
    rng = np.random.default_rng(21)
    obs, hidden = generate_failures(generate_customers(400, rng), 1_500, rng)
    cf = OutcomeModel("calibrated").counterfactual_table(obs, hidden)
    logged = pd.concat([obs, log_actions(obs, cf, np.random.default_rng(22))], axis=1)
    model = TLearner(UpliftConfig(n_estimators=30, min_child_samples=20)).fit(
        build_features(logged), logged["action_idx"].to_numpy(), logged["recovered"].to_numpy(), logged["propensity"].to_numpy()
    )
    store = AuditStore(tmp_path_factory.mktemp("audit"))
    agent = Agent(model, {m.merchant_id: m for m in MERCHANTS}, MockExecutor(store), store, estimate="mean")
    return agent, obs


def test_context_from_row_validates_and_drops_extras(agent_and_events) -> None:
    _, obs = agent_and_events
    rec = obs.iloc[0].to_dict()
    rec["something_post_failure"] = 1
    ctx = context_from_row(rec)
    assert ctx.event_id == rec["event_id"] and not hasattr(ctx, "something_post_failure")
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        context_from_row({**rec, "amount": -5})


def test_handle_event_returns_full_audit_row(agent_and_events) -> None:
    agent, obs = agent_and_events
    row = handle_event(agent, context_from_row(obs.iloc[1].to_dict()))
    assert row["event_id"] == obs.iloc[1]["event_id"]
    assert len(row["uplift"]) == 5 and len(row["net_ev"]) == 5 and row["executor_result"]["status"] in ("executed", "skipped")
    again = handle_event(agent, context_from_row(obs.iloc[1].to_dict()))
    assert again["idempotency_key"] == row["idempotency_key"]
    if row["chosen_arm"] != 0:
        assert again["executor_result"]["status"] == "duplicate"


def test_app_routes_and_metrics(agent_and_events, tmp_path: Path) -> None:
    agent, obs = agent_and_events
    app = create_app(Settings(data_dir=tmp_path), agent=agent)
    routes = {r.path for r in app.routes}
    assert {"/webhook/payment_failed", "/decisions", "/decisions/{event_id}", "/metrics", "/health"} <= routes
    handle_event(agent, context_from_row(obs.iloc[2].to_dict()))
    m = agent.store.metrics()
    assert m["decisions"] >= 1 and "executor" in m
    assert agent.store.recent(limit=1)[0]["event_id"]


def test_payment_link_paid_records_outcome(agent_and_events) -> None:
    from counterfact.api.main import record_link_payment

    agent, obs = agent_and_events
    row = handle_event(agent, context_from_row(obs.iloc[3].to_dict()))
    payload = {"entity": "event", "event": "payment_link.paid", "payload": {
        "payment_link": {"entity": {"id": "plink_1", "reference_id": row["idempotency_key"], "amount": 29900, "amount_paid": 29900, "status": "paid"}},
        "payment": {"entity": {"id": "pay_1", "amount": 29900, "status": "captured", "notes": {"idempotency_key": row["idempotency_key"]}}},
    }}
    out = record_link_payment(agent, payload)
    assert out["outcome_recorded"] == row["event_id"] and out["recovered_amount"] == 299.0
    assert agent.store.get(row["event_id"])["outcome"]["source"] == "payment_link.paid"
    assert record_link_payment(agent, {"payload": {"payment_link": {"entity": {"reference_id": "nope"}}}})["ignored"] == "payment_link.paid"


def test_webhook_signature_verification() -> None:
    import hashlib
    import hmac

    body = json.dumps({"event": "subscription.charged"}).encode()
    secret = "whsec_test"
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_webhook(body, sig, secret) is True
    assert verify_webhook(body, "deadbeef", secret) is False
