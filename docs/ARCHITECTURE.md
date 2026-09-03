# Architecture

Filled progressively; component contracts are the docstrings of the named modules.

## Components
| module | contract |
|---|---|
| `sim/generator.py` | seeded population + 50k failures with decision-time features |
| `sim/outcome_model.py` | TRUE recovery process, three variants, common random numbers for coherent counterfactuals |
| `sim/logging_policy.py` | epsilon-uniform exploration policy; records `propensity` |
| `features/build.py` | decision-time feature matrix; leakage-checked |
| `models/uplift.py` | T-learner; `predict_uplift(X) -> (n, 5)` |
| `policy/ev.py` | net EV per arm; argmax with no-action tie-break and threshold |
| `policy/guardrails.py` | hard limits; machine-readable rejections |
| `policy/baselines.py` | `no_action`, `razorpay_default`, `heuristic` |
| `agent/loop.py` | event -> decide -> guard -> execute -> audit |
| `agent/executor.py` | `MockExecutor`, `RazorpayExecutor` (test mode); idempotency keys |
| `agent/audit.py` | append-only JSONL + SQLite view |
| `agent/explain.py` | LLM explanation, validated against action set |
| `eval/ab.py`, `eval/ope.py`, `eval/report.py` | randomized A/B with bootstrap CIs; IPS/SNIPS/DR; tables + figures |
| `eval/pipeline.py` | train (logged data only) and evaluate (counterfactual truth) one variant end to end; shared by `scripts/train.py`, `evaluate.py`, `sensitivity.py`, `z_dial.py` |
| `api/main.py` | FastAPI: `/webhook/payment_failed`, `/decisions`, `/metrics` |

## Agent runtime (Phase 4)
```
payment.failed ──► Agent.handle_batch ──► MLPolicy.decide (vectorised, gated z=2)
                        │                      │
                        │                      ▼
                        │              apply_guardrails (per row, reason codes)
                        │                      │
                        │                      ▼
                        │   idempotency_key = sha256(event_id | action | features_hash)
                        │                      │
                        │                      ▼
                        │   Executor.execute: ledger.reserve(key) ─► duplicate? return, no call
                        │                      │ provider call, 5xx -> backoff x3 -> queued
                        │                      ▼
                        └──────────► AuditStore.append_decision (JSONL + SQLite)
                                               │
              world / webhook ──► record_outcome ┘   redrive_queued() re-claims queued keys
```
* **Executors.** `MockExecutor` (default; counts charges per event so tests can assert zero
  duplicates; random or injected 5xx). `RazorpayExecutor` (test mode) has three execution paths
  (ADR-015, ADR-018), all idempotent on the same ledger:
  1. **Payment Link (live, executed).** `remind_and_retry` on a subscription event creates a
     Razorpay Payment Link for the outstanding amount with `reference_id = idempotency_key`,
     customer contact from the subscription's customer, SMS/email sent by Razorpay, 7-day expiry.
     Before creating, `payment_link.all(reference_id)` is checked so a re-drive after a lost
     response reuses the link. The ledger records `plink_id` and `short_url` as `executed`;
     `payment_link.paid` on the webhook writes the recovery outcome to the audit row. Verified
     live on 2026-09-03: links created on the real customer, one per key, including one whose
     first attempt hit an injected transport fault and one queued after three faults and
     re-driven; `scripts/verify_charges.py` confirms exactly one link per executed key.
  2. **Subscriptions timing and escalation (deferred with Razorpay's schedule).** Razorpay's
     Subscriptions product has no merchant-initiated retry (Razorpay retries `pending` itself;
     `halted` is paid by the customer). Retry arms record the outstanding invoice, its pay link
     and Razorpay's schedule (`invoice.all`) and are marked `deferred`; a paid-up subscription
     yields `skipped`; escalation snapshots the subscription (`subscription.fetch`). Never shown
     as executed.
  3. **Tokenised `createRecurring` (implemented, blocked on account enablement).**
     `order.create(receipt = idempotency_key)` then `payment.createRecurring` on a saved token,
     with reuse of an existing payment on the same receipt. On this test account
     `POST /v1/payments/create/recurring` (and the S2S `/payments/create/json`) return
     `BadRequestError: The requested URL was not found on the server` because Recurring Payments
     is not enabled for the account; orders, idempotency, backoff and re-drive were verified live
     up to that call, and `verify_charges.py` confirms zero payments on those keys.
  The transport-level fault injector covers the payment-link and recurring call paths, so
  backoff, queueing and re-drive are exercised on live endpoints. Webhook HMAC via
  `Utility.verify_webhook_signature`. Runs with `COUNTERFACT_EXECUTOR=razorpay` and test keys in
  `.env`; refuses non-test keys. Live tooling: `scripts/razorpay_setup.py`, `razorpay_token.py`,
  `verify_charges.py`, `webhook_selftest.py`.
* **Idempotency by construction.** The key is reserved in SQLite (`INSERT OR IGNORE`) before any
  provider call; a replayed webhook returns `duplicate` without a call; a queued key can only be
  re-driven through `claim_queued`, so an event can never be charged twice.
* **Graceful failure.** `scripts/inject_failure.py` (or `--inject-failure`) forces 5xx on the
  next provider calls: the executor backs off (0.05 s, 0.1 s, 0.2 s), parks the action as
  `queued`, the loop continues with the next event, and `redrive_queued` executes it later.
* **Audit row.** `event_id, idempotency_key, features_hash, uplift[5], net_ev[5], chosen_arm,
  proposed_arm, guardrail_checks[], rejection_codes, reason, executor_result, outcome,
  explanation`. JSONL is the append-only source of truth; SQLite is the queryable mirror.
* **Explanations.** `agent/explain.py`: Claude (`claude-haiku-4-5`) drafts two sentences from
  the audit row; a validator rejects text that names another arm or an out-of-set action
  (discounts, refunds, cancellations); rejected or failed calls fall back to a deterministic
  template; at most `MAX_EXPLANATIONS_PER_RUN = 50` calls per process; results are cached in the
  audit store and generated lazily (`--explain N`, `POST /decisions/{id}/explain`).
* **API.** `api/main.py`: `POST /webhook/payment_failed` (signature-verified when a Razorpay
  signature header is present), `GET /decisions`, `GET /decisions/{event_id}`,
  `POST /decisions/{event_id}/explain`, `GET /metrics`, `GET /health`.

## Known limitation: test mode cannot emit failure reasons
Razorpay test mode does not let us provoke specific decline codes (insufficient funds, expired card, risk decline). The failure taxonomy therefore comes from the simulator; retries, subscription reads and webhook verification run against real test-mode endpoints. We claim methodology and relative lift on calibrated synthetic data, not absolute rupees.
