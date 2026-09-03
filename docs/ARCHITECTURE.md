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
  duplicates; random or injected 5xx). `RazorpayExecutor` (test mode, verified live on
  2026-09-03 against `sub_TXXDsVmg4d3fkR`, ADR-015): escalation -> `subscription.fetch` snapshot for
  the ops ticket; retry arms -> `payment.createRecurring` when the event carries a saved token,
  customer and order (Recurring Payments product), otherwise `invoice.all(subscription_id)` to
  find the outstanding invoice and record its pay link, marked `deferred` because Razorpay's
  Subscriptions product has no merchant-initiated retry (Razorpay retries `pending` subscriptions
  itself; `halted` ones are paid by the customer via the link); reminder -> `invoice.notify_by`,
  or `deferred` with the exact Razorpay error and the pay link when Razorpay refuses (it does on
  subscription-generated invoices without a customer contact). Webhook HMAC via
  `Utility.verify_webhook_signature`. Both executors share the ledger and backoff logic; the
  Razorpay one runs with `COUNTERFACT_EXECUTOR=razorpay` and test keys in `.env` and refuses
  non-test keys.
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
