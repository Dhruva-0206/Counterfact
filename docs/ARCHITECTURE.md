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
  duplicates; random or injected 5xx). `RazorpayExecutor` has two explicit modes (ADR-015):
  1. **Tokenised recurring** (event carries `customer_id` + `token_id` from a registered card or
     UPI mandate): the agent executes the charge itself: `order.create` with
     `receipt = idempotency_key` and `payment_capture = 1`, then `payment.createRecurring` on the
     token. Before charging, orders carrying the same receipt are checked for an existing payment,
     so a re-drive after a lost response never charges twice. Later attempts of the arm's schedule
     belong to the sequencer. Verified live on 2026-09-03 up to the charge call: orders are
     created and idempotency, backoff and re-drive run against the real endpoint, but
     `POST /v1/payments/create/recurring` (and the S2S `/payments/create/json`) return
     `BadRequestError: The requested URL was not found on the server` on this test account,
     which is Razorpay's response when Recurring Payments / server-to-server payments are not
     enabled for the account. Enablement is a dashboard/support step, not a code change;
     `scripts/verify_charges.py` confirms via `order.payments` and `payment.all` that no key was
     charged twice.
  2. **Razorpay Subscriptions** (event carries a `subscription_id` only): Razorpay owns the
     charge schedule and offers no merchant-initiated retry. The agent controls timing, outreach
     and escalation: retry arms record the outstanding invoice, its pay link and Razorpay's
     schedule (`invoice.all`) and are marked `deferred`; reminders use `invoice.notify_by` or
     are `deferred` with Razorpay's exact refusal and the pay link; escalation snapshots the
     subscription (`subscription.fetch`); no outstanding invoice -> `skipped` with the reason.
  Nothing is reported as `executed` unless a provider call succeeded. Webhook HMAC via
  `Utility.verify_webhook_signature`. Both executors share the ledger and backoff logic; the
  Razorpay one runs with `COUNTERFACT_EXECUTOR=razorpay` and test keys in `.env` and refuses
  non-test keys. `scripts/razorpay_setup.py`, `razorpay_token.py`, `verify_charges.py` and
  `webhook_selftest.py` are the live tooling.
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
