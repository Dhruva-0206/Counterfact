# CLAUDE.md — RecoverAI project memory

## 1. Purpose
RecoverAI recovers failed **subscription payments** for SaaS merchants on Razorpay. For every failed payment it predicts the *incremental* recovery probability of each intervention, prices the intervention against its costs (message cost, ops cost, contact fatigue), executes the best one under hard guardrails, and proves impact with randomized A/B and off-policy evaluation. Built for the Razorpay AI Buildathon 2026, AI Revenue Recovery track.

**Differentiator:** "Do nothing" is a first-class action with its own expected value. Every action is priced against the counterfactual, and we report the cost of intervening when we shouldn't have (wasted contacts).

Core loop: `failure context → predict uplift per action → choose max net EV (incl. no-action) → guardrails → execute → log → measure incremental ₹`

## 2. Judging bars (tick when demoable)
- [ ] Measured money recovered **across a batch**
- [ ] Compliant escalation, **stopping rules**, **audit trail**
- [ ] **One failure handled gracefully** (API 5xx mid-retry → no double charge, backoff, logged, batch continues)
- [ ] Every money action **explainable, bounded, gated**
- [ ] **Honest metrics incl. false-positive cost** (wasted contacts reported voluntarily)
- [ ] Works on **Razorpay test-mode APIs**

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
| `make eval` | A/B + OPE tables/figures → `reports/` | ~2 min |
| `make demo` | 500-failure batch with one injected 5xx | ~20 s |
| `make test` | pytest | ~30 s |
| `make lint` | ruff | seconds |

Everything is seeded (`RECOVERAI_SEED`, default 42) and regenerable from scratch with `make all`.

## 5. Key decisions (full log: `docs/DECISIONS.md`)
- Five-arm action space, fixed: `no_action, retry_now, retry_delayed(d∈{1,3,7}), remind_and_retry, escalate_human`.
- **No discount arms** (creates an incentive to let payments fail).
- **T-learner** (one LightGBM per arm) over S-learner: per-arm models cannot regularize the treatment effect away, and each arm has distinct support in the logs.
- **Razorpay default (T+1/T+2/T+3 retries) is the headline baseline**, not no-intervention.
- Three simulator variants; `null_uplift` must produce ≥80% no-action — a headline result.
- **LLM is explanation-only**; it never chooses an arm; output validated against the allowed action set.
- One decision per failed payment; each arm is a bounded plan executed over a 14-day window.

## 6. Current status
**Done:** Phase 0 scaffold. Phase 1: seeded generator (5 merchants, 8k customers, 50k failures), true outcome process with three variants and common random numbers, calibrated so Razorpay default recovers ~60%, epsilon-uniform logging policy with propensities, baselines (`no_action`, `razorpay_default`, `heuristic`), counterfactual-leak test, Checkpoint 1 table (`scripts/checkpoint1.py`).
**In progress:** awaiting Checkpoint 1 confirmation.
**Next:** Phase 2 features + T-learner + net-EV policy + guardrails + A/B (Checkpoint 2).
**Known bugs:** none.
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
