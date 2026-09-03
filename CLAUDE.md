# CLAUDE.md — Counterfact project memory

## 1. Purpose
Counterfact recovers failed **subscription payments** for SaaS merchants on Razorpay. For every failed payment it predicts the *incremental* recovery probability of each intervention, prices the intervention against its costs (message cost, ops cost, contact fatigue), executes the best one under hard guardrails, and proves impact with randomized A/B and off-policy evaluation. Built for the Razorpay AI Buildathon 2026, AI Revenue Recovery track.

**Differentiator:** "Do nothing" is a first-class action with its own expected value. Every action is priced against the counterfactual, and we report the cost of intervening when we shouldn't have (wasted contacts).

Core loop: `failure context → predict uplift per action → choose max net EV (incl. no-action) → guardrails → execute → log → measure incremental ₹`

## 2. Judging bars (tick when demoable)
- [x] Measured money recovered **across a batch** (`make eval`: 20k-failure holdout, A/B + paired exact, per merchant)
- [x] Compliant escalation, **stopping rules**, **audit trail** (guardrails with reason codes, retry budget, contact caps, quiet hours; JSONL + SQLite audit)
- [x] **One failure handled gracefully** (`make demo`: injected 5xx -> backoff -> queued -> batch continues -> re-driven under the same key; duplicate charges = 0, tested)
- [x] Every money action **explainable, bounded, gated** (bounded plans, net-EV gate, guardrails, template/Claude explanations validated against the action set)
- [x] **Honest metrics incl. false-positive cost** (wasted contacts, abstention, null-uplift world, sensitivity table, conservatism dial)
- [x] Works on **Razorpay test-mode APIs** (live 2026-09-03: Payment Links executed one per key through an injected transport fault, zero duplicates confirmed Razorpay-side, `subscription.fetch`, `invoice.all`, `order.create`, HMAC-verified webhook over a public tunnel -> audit row; createRecurring blocked by account enablement. Row-by-row: `docs/LIVE_VERIFICATION.md`)

## 3. Architecture

```
                 ┌────────────────────────────── offline ───────────────────────────────┐
 sim/generator ──► data/<variant>/failures.parquet ──► features/build ──► models/uplift (T-learner)
       │                                                                        │
       └──► data/<variant>/counterfactuals.parquet   (NEVER read by training; test-enforced)
                                                                                │
                 ┌────────────────────────────── online ────────────────────────┼───────┐
 payment.failed ─► agent/loop ─► policy/ev (net EV per arm) ─► policy/guardrails ─► agent/executor
   (webhook /        │                 │                             │           (Mock | Razorpay test)
    batch script)    │                 └── no_action wins ties       └── machine-readable rejections
                     └──► agent/audit (JSONL + SQLite) ──► agent/explain (LLM, validated) ──► dashboard
                 ┌────────────────────────────── evaluation ──────────────────────────────┐
 eval/ab (randomized A/B, bootstrap CI)   eval/ope (IPS / SNIPS / DR vs A/B truth)   eval/report
```

Three simulator variants (`calibrated`, `misspecified`, `null_uplift`) share the same population and features; only the hidden outcome process differs.

## 4. Commands (Makefile; on Windows use `.\make.ps1 <target>`)
| target | what | expected runtime |
|---|---|---|
| `make setup` | `uv sync --all-groups` (Python 3.11, pinned) | 1–2 min clean |
| `make data` | 50k failures × 3 variants → `data/<variant>/` | ~30 s |
| `make train` | uplift models per variant → `data/<variant>/models/` | ~1–2 min |
| `make eval` | A/B + paired tables/figures → `reports/` | ~2 min |
| `make dial` | conservatism dial (z sweep) | ~2 min |
| `make sensitivity` | regenerate world per assumption, retrain, headline | ~15 min |
| `make demo` | 500-failure batch with one injected 5xx | ~5 s |
| `make dashboard` | Streamlit over audit + reports | interactive |
| `make test` | pytest | ~30 s |
| `make lint` | ruff | seconds |

Everything is seeded (`COUNTERFACT_SEED`, default 42) and regenerable from scratch with `make all`.

## 5. Key decisions (full log: `docs/DECISIONS.md`)
- Five-arm action space, fixed: `no_action, retry_now, retry_delayed(d∈{1,3,7}), remind_and_retry, escalate_human`.
- **No discount arms** (creates an incentive to let payments fail).
- **T-learner** (one LightGBM per arm) over S-learner: per-arm models cannot regularize the treatment effect away, and each arm has distinct support in the logs.
- **Razorpay default is the headline baseline**, not no-intervention. Headline column: Rs incremental vs `razorpay_default` per 1k; vs `no_action` second; abstention rate under all variants.
- **Equalized attempt budget (ADR-006):** every retry arm is a 3-attempt schedule (d, d+2, d+4) capped by the 3-retry guardrail; `razorpay_default` == `retry_delayed(1)` so the baseline lies inside the action space. Literal T+1/T+2/T+3 kept as `razorpay_t123` sensitivity action.
- Four simulator variants (`calibrated`, `misspecified`, `null_uplift`, `drifted`); `null_uplift` must produce ≥80% no-action; `drifted` is where ML beats the oracle-informed rule table (ADR-014).
- **LLM is explanation-only**; it never chooses an arm; output validated against the allowed action set.
- One decision per failed payment; each arm is a bounded plan executed over a 14-day window.
- Every simulator assumption that moves the headline is a sensitivity knob (`OutcomeModel(overrides=...)`) with a row in `docs/EVALUATION.md`.

## 6. Current status
**Done (Phases 0-5, drifted variant, live verification):** full pass/fail table with a regenerate
command per row in `docs/LIVE_VERIFICATION.md`. Summary:
- Explanations: live `claude-haiku-4-5`, 10/10 validated, 0 failing re-validation, cached, 0 API
  calls on re-run (ADR-016/017).
- Executor, three paths (ADR-015/018): Payment Links = live executed (4 links on customer
  `cust_SyobLyHEnrJxSn`, one per idempotency key, `evt_000007` executed on attempt 2 after an
  injected transport fault; `scripts/verify_charges.py` confirms one link per key Razorpay-side,
  zero duplicates). Subscriptions retry arms = 13 `deferred`/`skipped` with Razorpay's own refusal
  text, never reported executed. `escalate_human` = 7 executed via `subscription.fetch`. Tokenised
  `createRecurring` = implemented and tested, blocked by the account: `POST
  /v1/payments/create/recurring` returns `BadRequestError: The requested URL was not found on the
  server` (Recurring Payments not enabled).
- Webhook: `POST /webhook/payment_failed`, HMAC verified, tampered -> 401, signed -> 200 + audit
  row, proven locally and over the public tunnel. Razorpay payloads translated by
  `razorpay_event_to_context`; `payment_link.paid` -> outcome on the decision.
- Tunnel: **cloudflared, not ngrok** (ADR-019). `winget install Cloudflare.cloudflared`, then
  `cloudflared tunnel --url http://localhost:8000`. No account, no authtoken, passes Smart App
  Control. ngrok is dead on this machine: agent 3.3.1 < account minimum 3.20.0 (ERR_NGROK_121), and
  the updated 3.39.11 binary is blocked by Windows Smart App Control.
- 87 tests pass, ruff clean.
- Live test objects: plan `plan_TXXDsJxh3gYvKA`, subscription `sub_TXXDsVmg4d3fkR` (active, customer
  `cust_SyobLyHEnrJxSn`, token `token_TXXXCyI5Tx3WDS`), registration link
  `https://rzp.io/rzp/G4VFnmq2` (customer `cust_TXXUJrWXkXmEyw`, unused); `data/razorpay_test.json`.
- Docs: LIVE_VERIFICATION.md, WEBHOOK.md, JUDGE_QA.md, ADR-015..019, README/ARCHITECTURE/DEMO_SCRIPT.

**Left, both needing the Razorpay dashboard (rows 13-14 of the verification table):**
1. Start the server and tunnel, verify with `scripts/webhook_selftest.py --url https://<host>/webhook/payment_failed`,
   then create the Test Mode webhook: URL `https://<host>/webhook/payment_failed`, secret =
   `RAZORPAY_WEBHOOK_SECRET` from `.env`, events `payment.failed, payment.captured,
   subscription.pending, subscription.halted, subscription.charged, payment_link.paid`.
2. Trigger a failing payment (card 4100 2800 0008 0001) -> show the uvicorn log + audit row; pay one
   Payment Link with a test card -> show `payment_link.paid` -> outcome. Then flip rows 13-14 to
   pass and `git tag v1.0`.

**Known bugs:** none open. `--tokenized-events` rows show `failed` (404) by design until Recurring
Payments is enabled.
**Environment note:** the Bash tool truncates commands above roughly 8 KB; write large files with the
Write tool. `make`/`gh` absent, use `.\make.ps1`. cloudflared at
`C:\Program Files (x86)\cloudflared\cloudflared.exe`.

## 7. Conventions
- Python 3.11, type hints, `ruff` clean, docstrings on public functions.
- Small conventional commits after each working step.
- Tests required for: every guardrail rule, idempotency, no-leakage, OPE estimators on a toy case with a known answer.
- Never claim a number in docs that a script cannot regenerate.
- Ask before adding any dependency or arm not in the spec.
- Prefer Bash + heredocs for file edits in this environment; `make` is not installed on the dev machine, use `.\make.ps1`.

## 8. Do-not-do
- No discount arms.
- No counterfactual leakage: `data/*/counterfactuals.parquet` is never read by `features/` or `models/` (enforced by test).
- No LLM decisioning.
- No unreproducible numbers.
- No dependency additions without approval (pyarrow, ruff added at scaffold as parquet engine / linter — flagged at Checkpoint 1).
- No post-failure information in features.
