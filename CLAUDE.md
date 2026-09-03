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
- [x] Works on **Razorpay test-mode APIs** (live 2026-09-03: `subscription.fetch`, `invoice.all`, HMAC-verified webhook -> audit row; `notify_by`/`createRecurring` limits documented in ADR-015)

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
**Done:** Phase 0 scaffold; renamed to Counterfact, pushed to GitHub (origin = Dhruva-0206/Counterfact). Phase 1 simulator + baselines (Checkpoint 1 confirmed). ADR-006 equalized attempt budget. Phase 2: decision-time features with static + dynamic leak tests, IPS-weighted T-learner (10-member bootstrap ensemble), net-EV policy with confidence gate (gated z=2), guardrails with machine-readable reasons (every rule tested), two-arm A/B + paired-exact evaluator, per-merchant tables, conservatism dial (`make dial`), sensitivity harness (`make sensitivity`). ADR-013 escalation semantics + guardrailed baselines. Checkpoint 2 confirmed. Phase 3 OPE (IPS/SNIPS/DM/DR; DR within A/B CI in 9/9 cells; `scripts/ope.py`, toy-case tests). Abstention self-recovery table. Phase 4: agent loop, idempotent executors (Mock + Razorpay test mode), JSONL+SQLite audit with execution ledger, failure injection + re-drive, Claude/template explanations with validator and cache, FastAPI surface, `make demo` (Checkpoint 4). Drifted variant (ADR-014): ML beats the rule table by 32% under merchant drift. Phase 5: Streamlit dashboard (`make dashboard`, headless test), `docs/DEMO_SCRIPT.md`, README with pitch line and limitations. Live verification (2026-09-03): 10/10 Claude explanations validated + cached; Razorpay test-mode plan/subscription created (`scripts/razorpay_setup.py`), live `subscription.fetch`/`invoice.all`, deferred retry semantics (ADR-015); signed-webhook self-test passes (`scripts/webhook_selftest.py`, `docs/WEBHOOK.md`).
**In progress:** live verification: mode 1 (tokenised recurring) blocked by Razorpay account enablement of Recurring Payments (404 on /payments/create/recurring); step 3 webhook via ngrok waits on ngrok install.
**Next:** once Recurring Payments is enabled: `COUNTERFACT_EXECUTOR=razorpay python scripts/run_batch.py --n 20 --tokenized-events 3 --inject-failure --max-amount 299 --subscription-id sub_TXXDsVmg4d3fkR` then `python scripts/verify_charges.py --audit-dir <dir>`; once ngrok is authenticated, finish docs/WEBHOOK.md step 3 and the pass/fail table.
**Known bugs:** none. Known limitation: A/B CIs are wide (heavy-tailed rupees); paired exact is the ground truth.
**Environment note:** the Bash tool truncates commands above roughly 8 KB; write large files with the Write tool.

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
