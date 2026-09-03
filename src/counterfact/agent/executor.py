"""Executors: MockExecutor (default) and RazorpayExecutor (test mode). Idempotent by construction.

Contract (``BaseExecutor.execute``):

1. Reserve the idempotency key in the ledger. If it already exists, return ``duplicate`` and make
   **no** provider call. A replayed event therefore cannot charge twice.
2. Call the provider. A transient error (HTTP 5xx, connection reset, injected fault) is retried
   with exponential backoff up to ``max_api_retries`` times **under the same key**.
3. If the provider keeps failing, the key is parked as ``queued`` (no charge happened), the
   caller logs it and continues with the next event; ``redrive`` re-claims queued keys later.

``FailureInjector`` lets ``scripts/inject_failure.py`` force a 5xx on the next provider call from
outside the process (flag file), for the demo.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from counterfact.agent.audit import AuditStore
from counterfact.config import ESCALATE_HUMAN, NO_ACTION, REMIND_AND_RETRY, RETRY_DELAYED, RETRY_NOW
from counterfact.sim.schema import plan_for

RETRY_ARMS = (RETRY_NOW, RETRY_DELAYED, REMIND_AND_RETRY)


class TransientError(Exception):
    """Provider-side transient failure (5xx, timeout, connection reset)."""

    def __init__(self, status_code: int, detail: str = "") -> None:
        super().__init__(f"{status_code} {detail}".strip())
        self.status_code = status_code


class PermanentError(Exception):
    """Provider rejected the request for good (4xx other than 429/408)."""


@dataclass
class ExecutionRequest:
    idempotency_key: str
    event_id: str
    subscription_id: str
    action_name: str
    arm: int
    delay_days: int
    amount: float
    effective_retries: int
    message_send_at: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutorResult:
    status: str  # executed | queued | failed | skipped | duplicate
    attempts: int
    provider_ref: str | None = None
    error: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_idempotency_key(event_id: str, action_name: str, features_hash: str) -> str:
    """Deterministic key: same event + same decision on the same features -> same key."""
    return hashlib.sha256(f"{event_id}|{action_name}|{features_hash}".encode()).hexdigest()[:32]


class FailureInjector:
    """Forces the next N provider calls to fail with a 5xx. Programmatic and/or via a flag file."""

    def __init__(self, flag_path: Path | None = None, count: int = 0) -> None:
        self.flag_path = flag_path
        self._count = count
        self.fired = 0

    def arm(self, count: int = 1) -> None:
        self._count += count

    def _read_flag(self) -> int:
        if self.flag_path and self.flag_path.exists():
            try:
                n = int(self.flag_path.read_text().strip() or "1")
            except ValueError:
                n = 1
            self.flag_path.unlink(missing_ok=True)
            return n
        return 0

    def consume(self) -> bool:
        """True if this call should fail (and decrement the budget)."""
        self._count += self._read_flag()
        if self._count > 0:
            self._count -= 1
            self.fired += 1
            return True
        return False


class BaseExecutor:
    """Idempotent execution with backoff and queueing. Subclasses implement ``_call``."""

    name = "base"

    def __init__(
        self,
        ledger: AuditStore,
        max_api_retries: int = 3,
        backoff_base: float = 0.05,
        injector: FailureInjector | None = None,
        sleep=time.sleep,
    ) -> None:
        self.ledger = ledger
        self.max_api_retries = max_api_retries
        self.backoff_base = backoff_base
        self.injector = injector
        self._sleep = sleep
        self.log: list[dict[str, Any]] = []

    def execute(self, req: ExecutionRequest) -> ExecutorResult:
        if not self.ledger.reserve(req.idempotency_key, req.event_id, req.action_name):
            prior = self.ledger.lookup(req.idempotency_key) or {}
            return ExecutorResult(
                status="duplicate", attempts=0, provider_ref=(prior.get("result") or {}).get("provider_ref"),
                detail={"prior_status": prior.get("status")},
            )
        return self._run_reserved(req)

    def redrive(self, req: ExecutionRequest) -> ExecutorResult:
        """Retry a previously queued request under its original key."""
        if not self.ledger.claim_queued(req.idempotency_key):
            prior = self.ledger.lookup(req.idempotency_key) or {}
            return ExecutorResult(status="duplicate", attempts=0, detail={"prior_status": prior.get("status")})
        return self._run_reserved(req)

    def _run_reserved(self, req: ExecutionRequest) -> ExecutorResult:
        attempts = 0
        while True:
            attempts += 1
            try:
                if self.injector is not None and self.injector.consume():
                    raise TransientError(502, "injected provider failure")
                result = self._call(req)
                result.attempts = attempts
                self.ledger.finish(req.idempotency_key, result.status, attempts, result.to_dict())
                self.log.append({"key": req.idempotency_key, "attempt": attempts, "status": result.status})
                return result
            except TransientError as e:
                self.log.append({"key": req.idempotency_key, "attempt": attempts, "status": "transient", "error": str(e)})
                if attempts >= self.max_api_retries:
                    result = ExecutorResult(status="queued", attempts=attempts, error=str(e),
                                            detail={"reason": "provider unavailable after backoff; parked for re-drive"})
                    self.ledger.finish(req.idempotency_key, "queued", attempts, result.to_dict())
                    return result
                self._sleep(self.backoff_base * (2 ** (attempts - 1)))
            except PermanentError as e:
                result = ExecutorResult(status="failed", attempts=attempts, error=str(e))
                self.ledger.finish(req.idempotency_key, "failed", attempts, result.to_dict())
                return result

    def _call(self, req: ExecutionRequest) -> ExecutorResult:  # pragma: no cover - abstract
        raise NotImplementedError


class MockExecutor(BaseExecutor):
    """In-process provider stand-in. Counts charges per event so tests can assert no duplicates."""

    name = "mock"

    def __init__(self, ledger: AuditStore, failure_rate: float = 0.0, seed: int = 0, **kw: Any) -> None:
        super().__init__(ledger, **kw)
        self.failure_rate = failure_rate
        self.rng = np.random.default_rng(seed)
        self.charges: dict[str, int] = {}
        self.messages: dict[str, int] = {}
        self.escalations: dict[str, int] = {}

    def _call(self, req: ExecutionRequest) -> ExecutorResult:
        if self.failure_rate > 0 and self.rng.random() < self.failure_rate:
            raise TransientError(503, "simulated gateway timeout")
        ref = "mock_" + hashlib.sha1(req.idempotency_key.encode()).hexdigest()[:12]
        if req.arm == NO_ACTION:
            return ExecutorResult(status="skipped", attempts=0, detail={"note": "no_action: nothing executed"})
        if req.arm in RETRY_ARMS:
            self.charges[req.event_id] = self.charges.get(req.event_id, 0) + 1
            plan = plan_for(req.arm, req.delay_days)
            detail: dict[str, Any] = {
                "schedule_days": list(plan.retry_days[: req.effective_retries]),
                "attempts_scheduled": req.effective_retries,
            }
            if req.arm == REMIND_AND_RETRY:
                self.messages[req.event_id] = self.messages.get(req.event_id, 0) + 1
                detail["message_send_at"] = req.message_send_at or "now"
            return ExecutorResult(status="executed", attempts=0, provider_ref=ref, detail=detail)
        if req.arm == ESCALATE_HUMAN:
            self.escalations[req.event_id] = self.escalations.get(req.event_id, 0) + 1
            return ExecutorResult(status="executed", attempts=0, provider_ref="ops_" + ref[5:],
                                  detail={"queue": "merchant_ops", "sla_hours": 24})
        raise PermanentError(f"unknown arm {req.arm}")


class RazorpayExecutor(BaseExecutor):
    """Razorpay test-mode executor. Maps arms to real SDK calls; idempotency is ours, not Razorpay's.

    * retry_now / retry_delayed / remind_and_retry -> ``client.payment.createRecurring`` (charge
      the saved mandate/token again for the subscription's failed invoice); the schedule beyond
      the first attempt is recorded for the scheduler (out of scope for the demo).
    * remind_and_retry additionally -> ``client.invoice.notify_by(invoice_id, "sms")``.
    * escalate_human -> ``client.subscription.fetch`` snapshot attached to the ops ticket.
    * no_action -> nothing.

    Razorpay test mode cannot emit specific decline reasons; the failure taxonomy comes from the
    simulator while these calls hit real test-mode endpoints. Pass a fake client in tests.
    """

    name = "razorpay"

    def __init__(self, ledger: AuditStore, client: Any, **kw: Any) -> None:
        super().__init__(ledger, **kw)
        self.client = client

    @staticmethod
    def _wrap(fn, *args: Any, **kwargs: Any) -> Any:
        try:
            import razorpay.errors as rz_err  # type: ignore
        except ImportError:  # pragma: no cover
            rz_err = None
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - classify provider errors
            code = getattr(e, "status_code", None) or getattr(e, "code", None)
            name = type(e).__name__
            if rz_err is not None and isinstance(e, rz_err.ServerError | rz_err.GatewayError):
                raise TransientError(int(code or 502), name) from e
            if code and int(code) >= 500 or name in ("ServerError", "GatewayError", "ConnectionError", "Timeout"):
                raise TransientError(int(code or 502), name) from e
            raise PermanentError(f"{name}: {e}") from e

    def _call(self, req: ExecutionRequest) -> ExecutorResult:
        p = req.payload
        if req.arm == NO_ACTION:
            return ExecutorResult(status="skipped", attempts=0)
        if req.arm == ESCALATE_HUMAN:
            snap = self._wrap(self.client.subscription.fetch, req.subscription_id)
            return ExecutorResult(status="executed", attempts=0, provider_ref=req.subscription_id,
                                  detail={"queue": "merchant_ops", "subscription_status": snap.get("status")})
        detail: dict[str, Any] = {}
        if req.arm == REMIND_AND_RETRY and p.get("invoice_id"):
            self._wrap(self.client.invoice.notify_by, p["invoice_id"], "sms")
            detail["reminder"] = {"invoice_id": p["invoice_id"], "medium": "sms", "send_at": req.message_send_at}
        body = {
            "email": p.get("email", "customer@example.com"),
            "contact": p.get("contact", "9999999999"),
            "amount": int(round(req.amount * 100)),
            "currency": "INR",
            "order_id": p.get("order_id"),
            "customer_id": p.get("customer_id"),
            "token": p.get("token_id"),
            "recurring": "1",
            "description": f"Counterfact retry {req.action_name}",
            "notes": {"idempotency_key": req.idempotency_key, "event_id": req.event_id, "action": req.action_name},
        }
        body = {k: v for k, v in body.items() if v is not None}
        resp = self._wrap(self.client.payment.createRecurring, body)
        plan = plan_for(req.arm, req.delay_days)
        detail.update({
            "razorpay_payment_id": resp.get("razorpay_payment_id") or resp.get("id"),
            "schedule_days": list(plan.retry_days[: req.effective_retries]),
            "request": {k: v for k, v in body.items() if k not in ("email", "contact")},
        })
        return ExecutorResult(status="executed", attempts=0,
                              provider_ref=str(detail["razorpay_payment_id"]), detail=detail)


def verify_webhook(body: bytes, signature: str, secret: str) -> bool:
    """Razorpay webhook HMAC check via the SDK utility."""
    import razorpay  # type: ignore

    try:
        razorpay.Utility().verify_webhook_signature(body.decode("utf-8"), signature, secret)
        return True
    except Exception:  # noqa: BLE001 - SignatureVerificationError or malformed input
        return False


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)
