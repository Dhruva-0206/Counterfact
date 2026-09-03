"""The agent loop: event -> decide -> guard -> execute -> audit (-> outcome when known).

``Agent.handle_batch`` decides for a whole frame at once (vectorised policy), then walks rows to
apply guardrails, execute through the idempotent executor and write one audit row per event.
A provider failure mid-batch is absorbed by the executor (backoff, then ``queued``) and the loop
continues; ``redrive_queued`` retries parked keys later under the same idempotency key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from counterfact.agent.audit import AuditStore
from counterfact.agent.executor import (
    BaseExecutor,
    ExecutionRequest,
    ExecutorResult,
    make_idempotency_key,
)
from counterfact.config import ARMS, NO_ACTION, POLICY_ESTIMATE, POLICY_Z
from counterfact.features.build import FEATURES, build_features, features_hash
from counterfact.models.uplift import TLearner
from counterfact.policy.ev import MLPolicy
from counterfact.policy.guardrails import GuardrailDecision, apply_guardrails
from counterfact.sim.schema import Context, Merchant

EVENT_COLUMNS = ("event_id", "subscription_id", "customer_id", "failed_at") + FEATURES


@dataclass
class HandledEvent:
    row: dict[str, Any]  # the audit row as written
    guard: GuardrailDecision
    result: ExecutorResult


def event_frame(events: list[Context] | list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    """Normalise incoming events (pydantic contexts, dicts, or a frame) to one DataFrame."""
    if isinstance(events, pd.DataFrame):
        return events.reset_index(drop=True)
    rows = [e.model_dump() if isinstance(e, Context) else dict(e) for e in events]
    return pd.DataFrame(rows)


class Agent:
    """Decide, guard, execute and audit ``payment.failed`` events."""

    def __init__(
        self,
        model: TLearner,
        merchants: dict[str, Merchant],
        executor: BaseExecutor,
        store: AuditStore,
        estimate: str = POLICY_ESTIMATE,
        z: float = POLICY_Z,
    ) -> None:
        self.model = model
        self.merchants = merchants
        self.executor = executor
        self.store = store
        self.policy = MLPolicy(model, pd.DataFrame([m.model_dump() for m in merchants.values()]), estimate, z)
        self._pending: dict[str, ExecutionRequest] = {}  # idempotency_key -> request (for re-drive)

    # ---- core -------------------------------------------------------------------------------------
    def handle_batch(self, events: list[Context] | list[dict[str, Any]] | pd.DataFrame) -> list[HandledEvent]:
        df = event_frame(events)
        X = build_features(df)
        dec = self.policy.decide(df, X)
        out: list[HandledEvent] = []
        records = df.to_dict("records")
        for i, rec in enumerate(records):
            merchant = self.merchants[rec["merchant_id"]]
            guard = apply_guardrails(rec, merchant, dec.net_ev[i], int(dec.arm[i]), int(dec.best_delay[i]))
            fhash = features_hash(rec)
            key = make_idempotency_key(rec["event_id"], guard.action_name, fhash)
            req = ExecutionRequest(
                idempotency_key=key, event_id=rec["event_id"], subscription_id=str(rec.get("subscription_id", "")),
                action_name=guard.action_name, arm=guard.arm, delay_days=guard.delay_days,
                amount=float(rec["amount"]), effective_retries=guard.effective_retries,
                message_send_at=guard.message_send_at,
                payload={k: rec.get(k) for k in ("customer_id", "invoice_id", "token_id", "order_id") if k in rec},
            )
            result = self.executor.execute(req)
            if result.status == "queued":
                self._pending[key] = req
            row = self._audit_row(rec, i, dec, guard, fhash, key, result)
            self.store.append_decision(row)
            out.append(HandledEvent(row, guard, result))
        return out

    def handle(self, event: Context | dict[str, Any]) -> HandledEvent:
        return self.handle_batch([event])[0]

    def record_outcome(self, event_id: str, recovered: bool, recovered_amount: float, **extra: Any) -> None:
        self.store.set_outcome(event_id, {"recovered": bool(recovered), "recovered_amount": float(recovered_amount), **extra})

    def redrive_queued(self) -> list[tuple[str, ExecutorResult]]:
        """Retry every queued execution under its original key; returns (event_id, result) pairs."""
        done = []
        for key, req in list(self._pending.items()):
            result = self.executor.redrive(req)
            if result.status != "queued":
                self._pending.pop(key, None)
            self.store.set_executor_result(req.event_id, result.to_dict())
            done.append((req.event_id, result))
        return done

    # ---- audit row -------------------------------------------------------------------------------
    def _audit_row(self, rec: dict[str, Any], i: int, dec, guard: GuardrailDecision, fhash: str,
                   key: str, result: ExecutorResult) -> dict[str, Any]:
        ev = dec.net_ev[i]
        up = dec.uplift[i]
        proposed = ARMS[int(dec.arm[i])]
        best_ev = float(ev[int(guard.arm)])
        if guard.arm == NO_ACTION and not guard.overridden:
            reason = (f"no_action: best net EV {float(np.max(ev)):.0f} did not clear the gate "
                      f"(threshold {float(dec.threshold[i]):.0f}) or no arm beat doing nothing")
        elif guard.overridden:
            codes = ",".join(c.code for c in guard.checks if not c.passed)
            reason = f"{guard.action_name} by guardrail ({codes}); policy proposed {proposed} (net EV {float(ev[int(dec.arm[i])]):.0f})"
        else:
            reason = (f"{guard.action_name}: highest net EV {best_ev:.0f} = uplift {float(up[int(guard.arm)]):.3f} "
                      f"x Rs {float(rec['amount']):,.0f} - cost {float(dec.cost[i][int(guard.arm)]):.0f}; "
                      f"P(recover | no_action) = {float(dec.base[i]):.2f}")
        return {
            "event_id": rec["event_id"],
            "idempotency_key": key,
            "features_hash": fhash,
            "merchant_id": rec["merchant_id"],
            "subscription_id": rec.get("subscription_id"),
            "amount": float(rec["amount"]),
            "failure_category": rec["failure_category"],
            "attempt_number": int(rec["attempt_number"]),
            "uplift": [round(float(v), 5) for v in up],
            "net_ev": [round(float(v), 2) for v in ev],
            "p_no_action_hat": round(float(dec.base[i]), 4),
            "threshold": float(dec.threshold[i]),
            "proposed_arm": int(dec.arm[i]),
            "chosen_arm": int(guard.arm),
            "action_name": guard.action_name,
            "delay_days": int(guard.delay_days),
            "overridden": bool(guard.overridden),
            "effective_retries": int(guard.effective_retries),
            "message_send_at": guard.message_send_at,
            "guardrail_checks": [c.__dict__ for c in guard.checks],
            "rejection_codes": "|".join(c.code for c in guard.checks if not c.passed),
            "reason": reason,
            "executor": self.executor.name,
            "executor_result": result.to_dict(),
            "outcome": None,
        }


def context_from_row(rec: dict[str, Any]) -> Context:
    """Build a validated ``Context`` from a raw event dict (drops unknown keys)."""
    keys = set(Context.model_fields)
    return Context(**{k: v for k, v in rec.items() if k in keys})
