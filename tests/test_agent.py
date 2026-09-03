"""Agent loop, executors, audit store: idempotency and graceful 5xx handling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from counterfact.agent.audit import AuditStore
from counterfact.agent.executor import (
    ExecutionRequest,
    FailureInjector,
    MockExecutor,
    PermanentError,
    RazorpayExecutor,
    TransientError,
    make_idempotency_key,
)
from counterfact.agent.loop import Agent
from counterfact.config import RETRY_NOW
from counterfact.features.build import build_features
from counterfact.models.uplift import TLearner, UpliftConfig
from counterfact.sim.generator import MERCHANTS, generate_customers, generate_failures
from counterfact.sim.logging_policy import log_actions
from counterfact.sim.outcome_model import OutcomeModel


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(11)
    obs, hidden = generate_failures(generate_customers(600, rng), 2_500, rng)
    cf = OutcomeModel("calibrated").counterfactual_table(obs, hidden)
    logged = pd.concat([obs, log_actions(obs, cf, np.random.default_rng(12))], axis=1)
    X = build_features(logged)
    model = TLearner(UpliftConfig(n_estimators=40, min_child_samples=20, n_ensemble=2)).fit(
        X, logged["action_idx"].to_numpy(), logged["recovered"].to_numpy(), logged["propensity"].to_numpy()
    )
    return obs, cf, model


def make_agent(tmp_path: Path, model, **exec_kw) -> tuple[Agent, MockExecutor, AuditStore]:
    store = AuditStore(tmp_path / "audit")
    executor = MockExecutor(store, backoff_base=0.0, sleep=lambda s: None, **exec_kw)
    merchants = {m.merchant_id: m for m in MERCHANTS}
    return Agent(model, merchants, executor, store, estimate="mean"), executor, store


def req(key: str = "k1", arm: int = RETRY_NOW, event: str = "evt_1") -> ExecutionRequest:
    return ExecutionRequest(idempotency_key=key, event_id=event, subscription_id="sub_1", action_name="retry_now",
                            arm=arm, delay_days=0, amount=999.0, effective_retries=3)


# ---- executor / ledger ----------------------------------------------------------------------------
def test_same_key_executes_once(tmp_path: Path) -> None:
    store = AuditStore(tmp_path)
    ex = MockExecutor(store)
    r1 = ex.execute(req())
    r2 = ex.execute(req())
    r3 = ex.execute(req())
    assert r1.status == "executed" and r2.status == "duplicate" and r3.status == "duplicate"
    assert ex.charges == {"evt_1": 1}
    assert r2.provider_ref == r1.provider_ref


def test_transient_error_backs_off_then_queues_without_charging(tmp_path: Path) -> None:
    store = AuditStore(tmp_path)
    sleeps: list[float] = []
    inj = FailureInjector(count=3)
    ex = MockExecutor(store, injector=inj, max_api_retries=3, backoff_base=0.1, sleep=sleeps.append)
    r = ex.execute(req())
    assert r.status == "queued" and r.attempts == 3 and ex.charges == {}
    assert sleeps == [0.1, 0.2]  # exponential backoff between attempts
    assert store.lookup("k1")["status"] == "queued"
    # re-drive succeeds under the same key, charging exactly once
    r2 = ex.redrive(req())
    assert r2.status == "executed" and ex.charges == {"evt_1": 1}
    assert ex.redrive(req()).status == "duplicate"


def test_partial_outage_recovers_within_backoff(tmp_path: Path) -> None:
    store = AuditStore(tmp_path)
    inj = FailureInjector(count=2)  # fails twice, third attempt succeeds
    ex = MockExecutor(store, injector=inj, max_api_retries=3, sleep=lambda s: None)
    r = ex.execute(req())
    assert r.status == "executed" and r.attempts == 3 and ex.charges == {"evt_1": 1}


def test_flag_file_injection(tmp_path: Path) -> None:
    flag = tmp_path / "inject.flag"
    flag.write_text("1")
    inj = FailureInjector(flag_path=flag)
    assert inj.consume() is True and not flag.exists() and inj.consume() is False


def test_razorpay_executor_uses_sdk_shapes_and_is_idempotent(tmp_path: Path) -> None:
    calls: list[tuple] = []

    class FakePayment:
        def createRecurring(self, data):  # noqa: N802 - SDK name
            calls.append(("createRecurring", data))
            return {"razorpay_payment_id": "pay_test123"}

    class FakeInvoice:
        def notify_by(self, invoice_id, medium):
            calls.append(("notify_by", invoice_id, medium))
            return {}

    class FakeSub:
        def fetch(self, sid):
            calls.append(("fetch", sid))
            return {"status": "halted"}

    class FakeClient:
        payment, invoice, subscription = FakePayment(), FakeInvoice(), FakeSub()

    store = AuditStore(tmp_path)
    ex = RazorpayExecutor(store, FakeClient())
    r = ex.execute(ExecutionRequest("k9", "evt_9", "sub_9", "remind_and_retry", 3, 0, 1499.0, 3,
                                    payload={"invoice_id": "inv_1", "customer_id": "cust_1", "token_id": "tok_1"}))
    assert r.status == "executed" and r.provider_ref == "pay_test123"
    assert calls[0] == ("notify_by", "inv_1", "sms")
    body = calls[1][1]
    assert body["amount"] == 149900 and body["currency"] == "INR" and body["recurring"] == "1"
    assert body["notes"]["idempotency_key"] == "k9"
    assert ex.execute(ExecutionRequest("k9", "evt_9", "sub_9", "remind_and_retry", 3, 0, 1499.0, 3)).status == "duplicate"
    assert len(calls) == 2  # no second provider call


def test_razorpay_error_classification(tmp_path: Path) -> None:
    class Boom:
        def __init__(self, exc):
            self.exc = exc

        def fetch(self, sid):
            raise self.exc

    class ServerError(Exception):
        status_code = 503

    class BadRequestError(Exception):
        status_code = 400

    class FakeClient:
        def __init__(self, exc):
            self.subscription = Boom(exc)

    ex = RazorpayExecutor(AuditStore(tmp_path / "a"), FakeClient(ServerError("down")), sleep=lambda s: None)
    r = ex.execute(ExecutionRequest("k1", "e", "s", "escalate_human", 4, 0, 10.0, 0))
    assert r.status == "queued"
    ex2 = RazorpayExecutor(AuditStore(tmp_path / "b"), FakeClient(BadRequestError("bad")))
    r2 = ex2.execute(ExecutionRequest("k1", "e", "s", "escalate_human", 4, 0, 10.0, 0))
    assert r2.status == "failed"
    with pytest.raises(TransientError):
        RazorpayExecutor._wrap(lambda: (_ for _ in ()).throw(ServerError()))
    with pytest.raises(PermanentError):
        RazorpayExecutor._wrap(lambda: (_ for _ in ()).throw(BadRequestError()))


# ---- agent loop ---------------------------------------------------------------------------------
def test_batch_with_injected_5xx_continues_and_never_double_charges(tmp_path: Path, world) -> None:
    obs, cf, model = world
    agent, executor, store = make_agent(tmp_path, model)
    df = obs.head(120).reset_index(drop=True)
    handled = agent.handle_batch(df.iloc[:60])
    executor.injector = FailureInjector(count=3)
    handled += agent.handle_batch(df.iloc[60:])
    statuses = [h.result.status for h in handled]
    assert len(handled) == 120 and statuses.count("queued") == 1
    assert all(v == 1 for v in executor.charges.values())
    # every event has exactly one audit row with the required fields
    rows = store.all_decisions()
    assert len(rows) == 120 and len({r["event_id"] for r in rows}) == 120
    required = {"event_id", "idempotency_key", "features_hash", "uplift", "net_ev", "chosen_arm",
                "guardrail_checks", "reason", "executor_result"}
    assert required <= set(rows[0])
    assert len(rows[0]["uplift"]) == 5 and len(rows[0]["net_ev"]) == 5
    queued = [h for h in handled if h.result.status == "queued"][0]
    assert store.lookup(queued.row["idempotency_key"])["status"] == "queued"
    # re-drive executes the parked action exactly once
    redriven = agent.redrive_queued()
    assert len(redriven) == 1 and redriven[0][1].status in ("executed", "skipped")
    assert all(v == 1 for v in executor.charges.values())
    assert store.get(queued.row["event_id"])["executor_status"] in ("executed", "skipped")


def test_replaying_the_same_events_is_idempotent(tmp_path: Path, world) -> None:
    obs, cf, model = world
    agent, executor, store = make_agent(tmp_path, model)
    df = obs.head(40).reset_index(drop=True)
    first = agent.handle_batch(df)
    second = agent.handle_batch(df)  # webhook redelivery
    assert all(h.result.status != "duplicate" for h in first)
    acted = [h for h in second if h.guard.arm != 0]
    assert acted and all(h.result.status == "duplicate" for h in acted)
    assert all(v == 1 for v in executor.charges.values())
    keys = {h.row["idempotency_key"] for h in first}
    assert keys == {h.row["idempotency_key"] for h in second}


def test_idempotency_key_is_deterministic_and_action_specific() -> None:
    a = make_idempotency_key("evt_1", "retry_now", "abc")
    assert a == make_idempotency_key("evt_1", "retry_now", "abc")
    assert a != make_idempotency_key("evt_1", "retry_delayed_3", "abc")
    assert a != make_idempotency_key("evt_1", "retry_now", "abd")


def test_audit_jsonl_is_append_only_and_outcomes_are_recorded(tmp_path: Path, world) -> None:
    obs, cf, model = world
    agent, executor, store = make_agent(tmp_path, model)
    handled = agent.handle_batch(obs.head(5).reset_index(drop=True))
    agent.record_outcome(handled[0].row["event_id"], True, 999.0)
    lines = [json.loads(line) for line in store.jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 6 and lines[-1]["outcome"]["recovered"] is True
    assert store.get(handled[0].row["event_id"])["outcome"]["recovered_amount"] == 999.0
    m = store.metrics()
    assert m["decisions"] == 5 and m["outcomes_known"] == 1 and m["rs_recovered"] == 999.0
