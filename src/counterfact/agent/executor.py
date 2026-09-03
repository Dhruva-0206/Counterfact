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

    def __init__(self, message: str, request: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.request = request or {}


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
    status: str  # executed | queued | failed | skipped | duplicate | deferred
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
                request = getattr(e, "request", {}) or {}
                result = ExecutorResult(status="failed", attempts=attempts, error=str(e),
                                        detail={"request": request, "mode": request.get("mode")})
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
    """Razorpay test-mode executor with two explicit modes (ADR-015):

    **Mode 1, tokenised recurring** (event carries ``customer_id`` + ``token_id`` from a card /
    UPI mandate registered through the Recurring Payments product): the agent executes the charge
    itself: ``order.create`` with ``receipt = idempotency_key`` and ``payment_capture = 1``, then
    ``payment.createRecurring`` on the token. Before charging, orders carrying the same receipt are
    checked for an existing payment, so a retry after a lost response never charges twice.
    Later attempts in the arm's schedule belong to the sequencer.

    **Mode 2, Razorpay Subscriptions** (event carries a ``subscription_id`` only): Razorpay owns
    the charge schedule and offers no merchant-initiated retry. The agent controls timing,
    outreach and escalation: retry arms record the outstanding invoice, its pay link and
    Razorpay's schedule (``invoice.all``) and are marked ``deferred``; reminders use
    ``invoice.notify_by`` or are ``deferred`` with Razorpay's exact refusal and the pay link;
    escalation snapshots the subscription (``subscription.fetch``). Nothing is reported as
    executed unless a provider call succeeded.

    Webhook HMAC via ``Utility.verify_webhook_signature``. ``live_injector`` makes the next
    ``createRecurring`` call fail at the transport layer so backoff and re-drive run against the
    live endpoint. Test mode cannot emit specific decline reasons (docs/ARCHITECTURE.md).
    """

    name = "razorpay"

    def __init__(
        self,
        ledger: AuditStore,
        client: Any,
        reminder_medium: str = "sms",
        live_injector: FailureInjector | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(ledger, **kw)
        self.client = client
        self.reminder_medium = reminder_medium
        self.live_injector = live_injector
        if live_injector is not None:
            self._install_transport_fault(live_injector)

    def _install_transport_fault(self, injector: FailureInjector) -> None:
        """Make the next armed ``POST /payments/create/recurring`` fail at the transport layer.

        The failure happens inside the real SDK call path (a ``requests`` ConnectionError raised
        before the request leaves the machine), so the executor's backoff, queueing and re-drive
        run against the live endpoint and nothing is charged by the failed attempt.
        """
        session = getattr(self.client, "session", None)
        if session is None:  # fake clients in tests
            return
        real_post = session.post

        def post(url, *args, **kwargs):
            if "/payments/create/recurring" in str(url) and injector.consume():
                import requests

                raise requests.exceptions.ConnectionError("injected transport fault: connection reset by peer")
            return real_post(url, *args, **kwargs)

        session.post = post

    @staticmethod
    def _wrap(fn, *args: Any, request: dict[str, Any] | None = None, **kwargs: Any) -> Any:
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
            raise PermanentError(f"{name}: {e}", request=request) from e

    def _charge_token(self, req: ExecutionRequest, notes: dict[str, Any], schedule: list[float]) -> ExecutorResult:
        """Mode 1, tokenised recurring: order (receipt = idempotency key) + ``payment.createRecurring``.

        Idempotency survives a lost response: before charging, any order already carrying this
        receipt is looked up and, if it has an authorized/captured payment, that payment is
        returned instead of a new charge.
        """
        p = req.payload
        key = req.idempotency_key
        existing = self._wrap(self.client.order.all, {"receipt": key, "count": 5},
                              request={"call": "order.all", "receipt": key})
        order = None
        for o in existing.get("items", []):
            pays = self._wrap(self.client.order.payments, o["id"], request={"call": "order.payments", "order_id": o["id"]})
            done = [pm for pm in pays.get("items", []) if pm.get("status") in ("captured", "authorized")]
            if done:
                return ExecutorResult(
                    status="executed", attempts=0, provider_ref=str(done[0]["id"]),
                    detail={"mode": "tokenized_recurring", "call": "order.payments", "order_id": o["id"],
                            "reused": True, "note": "a payment already existed for this idempotency key; no new charge",
                            "schedule_days": schedule},
                )
            order = o
            break
        if order is None:
            body = {"amount": int(round(req.amount * 100)), "currency": "INR", "receipt": key,
                    "payment_capture": 1, "notes": notes}
            order = self._wrap(self.client.order.create, body, request={"call": "order.create", **body})
        charge = {
            "email": p.get("email", "counterfact.test@example.com"),
            "contact": p.get("contact", "+919999999999"),
            "amount": int(round(req.amount * 100)),
            "currency": "INR",
            "order_id": order["id"],
            "customer_id": p["customer_id"],
            "token": p["token_id"],
            "recurring": "1",
            "description": f"Counterfact {req.action_name}",
            "notes": notes,
        }
        summary = {"mode": "tokenized_recurring", "call": "payment.createRecurring",
                   **{k: v for k, v in charge.items() if k not in ("email", "contact")}}
        resp = self._wrap(self.client.payment.createRecurring, charge, request=summary)
        pay_id = resp.get("razorpay_payment_id") or resp.get("id")
        return ExecutorResult(
            status="executed", attempts=0, provider_ref=str(pay_id),
            detail={"mode": "tokenized_recurring", "request": summary, "response": resp, "order_id": order["id"],
                    "schedule_days": schedule,
                    "note": "first attempt executed now; later attempts in schedule_days are the sequencer's"},
        )

    def _outstanding_invoice(self, subscription_id: str) -> dict[str, Any] | None:
        inv = self._wrap(self.client.invoice.all, {"subscription_id": subscription_id, "count": 10},
                         request={"call": "invoice.all", "subscription_id": subscription_id})
        items = [i for i in inv.get("items", []) if i.get("status") in ("issued", "partially_paid", "expired")]
        items.sort(key=lambda i: int(i.get("created_at") or 0), reverse=True)
        return items[0] if items else None

    def _call(self, req: ExecutionRequest) -> ExecutorResult:
        p = req.payload
        if req.arm == NO_ACTION:
            return ExecutorResult(status="skipped", attempts=0)
        if req.arm == ESCALATE_HUMAN:
            snap = self._wrap(self.client.subscription.fetch, req.subscription_id,
                              request={"call": "subscription.fetch", "subscription_id": req.subscription_id})
            return ExecutorResult(status="executed", attempts=0, provider_ref=req.subscription_id,
                                  detail={"call": "subscription.fetch", "queue": "merchant_ops",
                                          "subscription_status": snap.get("status")})
        plan = plan_for(req.arm, req.delay_days)
        schedule = list(plan.retry_days[: req.effective_retries])
        notes = {"idempotency_key": req.idempotency_key, "event_id": req.event_id, "action": req.action_name}

        if p.get("token_id") and p.get("customer_id"):
            return self._charge_token(req, notes, schedule)

        invoice = self._outstanding_invoice(req.subscription_id)
        if invoice is None:
            snap = self._wrap(self.client.subscription.fetch, req.subscription_id,
                              request={"call": "subscription.fetch", "subscription_id": req.subscription_id})
            return ExecutorResult(
                status="skipped", attempts=0, provider_ref=req.subscription_id,
                detail={"mode": "razorpay_subscriptions", "call": "invoice.all", "subscription_status": snap.get("status"),
                        "reason": "no outstanding invoice to re-present (subscription not authenticated or fully paid)",
                        "idempotency_key": req.idempotency_key, "schedule_days": schedule},
            )
        pay_link = invoice.get("short_url")
        base_detail = {
            "mode": "razorpay_subscriptions", "call": "invoice.all", "invoice_id": invoice["id"], "invoice_status": invoice.get("status"),
            "amount_due_paise": invoice.get("amount_due"), "pay_link": pay_link,
            "subscription_id": req.subscription_id, "schedule_days": schedule,
            "idempotency_key": req.idempotency_key,
            "note": ("Razorpay Subscriptions has no merchant-initiated retry: Razorpay retries a pending "
                     "subscription on its own schedule; a halted one is paid by the customer via the pay link"),
        }
        if req.arm != REMIND_AND_RETRY:
            # retry arms on the Subscriptions product: the retry itself is Razorpay's; we record the
            # outstanding invoice and its pay link so the schedule is auditable, and mark it deferred.
            return ExecutorResult(status="deferred", attempts=0, provider_ref=str(invoice["id"]), detail=base_detail)
        summary = {"call": "invoice.notify_by", "invoice_id": invoice["id"], "medium": self.reminder_medium,
                   "subscription_id": req.subscription_id, "notes": notes}
        try:
            resp = self._wrap(self.client.invoice.notify_by, invoice["id"], self.reminder_medium, request=summary)
        except PermanentError as e:
            # Razorpay refuses notifications on subscription-generated invoices without a customer
            # contact ("Operation not allowed for Invoice in issued status"). Record the exact error and
            # the pay link so the merchant sends the reminder through its own channel; never a silent no-op.
            return ExecutorResult(status="deferred", attempts=0, provider_ref=str(invoice["id"]),
                                  error=str(e), detail={**base_detail, "request": summary,
                                                        "message_send_at": req.message_send_at,
                                                        "reminder": "send pay_link via merchant channel"})
        return ExecutorResult(
            status="executed", attempts=0, provider_ref=str(invoice["id"]),
            detail={"request": summary, "response": resp, "schedule_days": schedule, "pay_link": pay_link,
                    "message_send_at": req.message_send_at},
        )


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


def build_executor(settings, ledger: AuditStore, injector: FailureInjector | None = None,
                   live_injector: FailureInjector | None = None, **kw: Any) -> BaseExecutor:
    """Executor selected by ``settings.executor``: ``mock`` (default) or ``razorpay`` (test keys only)."""
    if settings.executor == "razorpay":
        import razorpay  # type: ignore

        if not settings.razorpay_key_id or not settings.razorpay_key_secret:
            raise RuntimeError("COUNTERFACT_EXECUTOR=razorpay needs RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in .env")
        if not settings.razorpay_key_id.startswith("rzp_test_"):
            raise RuntimeError("refusing to build RazorpayExecutor: key is not a test key (rzp_test_ prefix)")
        client = razorpay.Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))
        return RazorpayExecutor(ledger, client, injector=injector, live_injector=live_injector, **kw)
    return MockExecutor(ledger, failure_rate=settings.executor_failure_rate, seed=settings.seed, injector=injector, **kw)
