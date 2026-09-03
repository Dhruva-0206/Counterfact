# Counterfact

Net-expected-value recovery of failed subscription payments for Razorpay merchants.
**"Do nothing" is a first-class action with its own expected value.**

Razorpay AI Buildathon 2026, AI Revenue Recovery track.

> Every number in this README is regenerated from seed 42 by `make data && make train && make eval`
> (plus `make dial`, `make sensitivity`, `make demo`). Dashboard: `make dashboard`. Demo flow:
> `docs/DEMO_SCRIPT.md`. Decisions: `docs/DECISIONS.md`. Methodology: `docs/EVALUATION.md`.

**Matches a rule table written with perfect knowledge of the failure taxonomy, using 63-83% fewer
reminders, abstains correctly when nothing works, and beats it when merchants diverge from the
taxonomy.** Every clause is a regenerable number: calibrated Rs 576k vs Rs 594k per 1,000
failures (-3%) at 127 vs 342 reminders; misspecified Rs 559k vs Rs 641k at 58 vs 342; null_uplift
80.9% abstention, 4 reminders and Rs -202 per 1k vs the rule table's 342 and Rs -10,399; drifted
Rs 818k vs Rs 617k (+32%, A/B CI excludes zero). Clause-by-clause table in `docs/EVALUATION.md`.

## Phase 2 results: ML policy vs Razorpay default (20k-failure holdout per variant)
The policy predicts the incremental recovery probability of each arm with a T-learner (one
LightGBM per action, 10-member bootstrap ensemble, trained on logged data with recorded
propensities), prices it against message cost, ops cost and contact fatigue, **acts only when the
ensemble lower confidence bound of some arm's net EV clears the merchant threshold** (gate z = 2),
and then runs the choice through hard guardrails. `razorpay_default` is Razorpay's automatic
retry schedule and, under the equalized attempt budget (ADR-006), is exactly our `retry_delayed(1)`
action, so no lift below comes from extra attempts.

| variant | policy | Rs incr vs razorpay_default /1k | Rs incr vs no_action /1k | recovery delta | raw recovery | abstention | contacts /1k | wasted contacts /1k | escalations /1k |
|---|---|---|---|---|---|---|---|---|---|
| calibrated | **ml_policy** | **576,174** | 3,006,553 | **+5.7%** | 65.2% | 13.0% | 127 | 28 | 234 |
| calibrated | razorpay_default | 0 | 2,430,380 | +0.0% | 59.5% | 0.0% | 0 | 0 | 0 |
| calibrated | heuristic (guardrailed rule table) | 594,178 | 3,024,558 | +8.1% | 67.6% | 6.6% | 342 | 106 | 207 |
| calibrated | oracle (true probabilities) | 896,241 | 3,326,621 | +10.9% | 70.5% | 0.0% | 174 | 61 | 279 |
| misspecified | **ml_policy** | **559,383** | 2,721,382 | **+6.1%** | 64.1% | 15.8% | 58 | 16 | 227 |
| misspecified | razorpay_default | 0 | 2,161,999 | +0.0% | 58.0% | 0.0% | 0 | 0 | 0 |
| misspecified | heuristic | 641,365 | 2,803,364 | +9.5% | 67.4% | 6.6% | 342 | 107 | 207 |
| misspecified | oracle | 977,449 | 3,139,448 | +12.8% | 70.8% | 0.0% | 213 | 77 | 290 |
| null_uplift | **ml_policy** | -202 | -202 | -0.0% | 23.4% | **80.9%** | 4 | 1 | 121 |
| null_uplift | razorpay_default | 0 | 0 | +0.0% | 23.5% | 0.0% | 0 | 0 | 0 |
| null_uplift | heuristic | -10,399 | -10,399 | -0.1% | 23.3% | 6.6% | 342 | 106 | 207 |
| drifted | **ml_policy** | **817,943** | 3,022,934 | **+6.4%** | 64.8% | 13.1% | 107 | 24 | 233 |
| drifted | razorpay_default | 0 | 2,204,991 | +0.0% | 58.5% | 0.0% | 0 | 0 | 0 |
| drifted | heuristic | 617,497 | 2,822,488 | +7.4% | 65.9% | 6.6% | 342 | 106 | 207 |
| drifted | oracle | 1,188,558 | 3,393,549 | +11.8% | 70.3% | 0.0% | 162 | 56 | 274 |

`drifted` = the calibrated world except that two merchants depart from the taxonomy (FitPulse:
bank errors behave like insufficient funds and reminders backfire; ScaleOps: insufficient funds
only recover on the day-7 invoice cycle and failed mandates need a human). Merchant id is a
feature, so the learner picks this up (FitPulse +17%, ScaleOps +57% vs the rule table); a category
rule table cannot (ADR-014).

Two-arm randomized A/B (`ml_policy` vs `razorpay_default`, ~10k units per arm, 95% bootstrap CI):
calibrated **+617,938** [192,676; 1,015,642]; misspecified **+537,563** [102,053; 914,654];
null_uplift -16,475 [-301,636; 288,010]; drifted **+847,510** [418,071; 1,233,256].

What the table says: the policy beats Razorpay's default by 5.7-6.1 recovery points (about Rs 5.8
lakh per 1,000 failures on this portfolio) with a significant A/B; it ties the rule table on
rupees while sending 63-83% fewer reminders and wasting 74-85% fewer; and in a world where no
intervention works it abstains on 81% of failures (the rest are guardrail-mandated escalations)
and loses nothing, where the rule table keeps sending 342 reminders per 1,000. The rule table is
an oracle-informed competitor: it was written with knowledge of the simulator's true per-category
probabilities, which no merchant has. The `drifted` variant (merchants whose failures depart from
the taxonomy) is where the two are expected to separate.

The confidence gate is a dial (`make dial`, `reports/figures/z_dial.png`): the plain point
estimate earns 13-26% more when uplift is real but intervenes on 77% of failures when it is not.
A sensitivity sweep over every simulator assumption that moves the headline (19 settings,
`make sensitivity`) keeps the ranking oracle >= ML > Razorpay default in all of them; details,
per-merchant tables and guardrail activity in `docs/EVALUATION.md`.

**Off-policy evaluation (Phase 3).** From logged data alone (recorded propensities), the
doubly-robust estimate of each policy's value lands within 0.6-3.2% of the paired-exact truth on
rupees for the ML policy, the rule table and Razorpay default under every variant, and its
difference vs Razorpay default sits inside the randomized A/B interval in 9 of 9 cells; plain
IPS drifts by up to 13% where the target policy rarely matches the logged action. Table:
`python scripts/ope.py --all-variants`, `docs/EVALUATION.md`.

![A/B, calibrated](reports/figures/ab_calibrated.png)
![conservatism dial](reports/figures/z_dial.png)

## Phase 4: the agent, end to end (`make demo`)
`python scripts/run_batch.py --n 500 --inject-failure` processes 500 held-out failures through
decide -> guardrails -> idempotent executor -> audit, with a provider 5xx injected mid-batch:

```
batch of 500 failures (calibrated) processed in 5.0s
  recovered: 307/500 = 61.4%   Rs recovered 1,824,718 of Rs 2,755,429 at risk
  contacts 61  escalations 129  abstentions 49  guardrail overrides 45
  executor ledger {'executed': 451, 'skipped': 49}  injected 5xx fired 3  queued mid-batch 1  re-driven 1
  duplicate charges: 0  (events charged more than once; must be 0)
```

Live (`COUNTERFACT_EXECUTOR=razorpay`): `python scripts/run_batch.py --n 20 --inject-failure --max-amount 299
--subscription-id sub_TXXDsVmg4d3fkR` creates real Payment Links on the test customer, one per
idempotency key, with the transport fault injected inside the live call path; then
`python scripts/verify_charges.py --audit-dir <dir>` confirms one link per key on Razorpay's side.

Every decision is one audit row (`event_id, idempotency_key, features_hash, uplift[5], net_ev[5],
chosen_arm, guardrail_checks[], reason, executor_result, outcome, explanation`) in an append-only
JSONL file with a SQLite mirror. The idempotency key is reserved in the ledger before any provider
call, so a replayed webhook or a re-driven queued action can never charge twice (tested). Claude
(`claude-haiku-4-5`) drafts two-sentence explanations lazily (`--explain N`, capped at 50 per run,
cached); a validator rejects any text naming an arm outside the action set and falls back to a
deterministic template. `RazorpayExecutor` maps the arms to test-mode endpoints
(`payment.createRecurring`, `invoice.notify_by`, `subscription.fetch`, webhook HMAC); see
`docs/ARCHITECTURE.md` for the limitation that test mode cannot emit the failure taxonomy.
What was actually proven against the live API, what is blocked by account enablement, and the
command that regenerates each row: `docs/LIVE_VERIFICATION.md`.

## Phase 1 results: baselines under three simulator variants
Per 1,000 failed payments, seed 42, 50,000 failures. `razorpay_default` is Razorpay's automatic
retry schedule after a failed subscription charge; under the equalized attempt budget (ADR-006)
it is exactly our `retry_delayed(1)` action (attempts at days 1, 3, 5), calibrated to the public
55-65% recovery band under `calibrated` and inside it under `misspecified` without re-tuning.
Under `null_uplift` no intervention has any causal effect, so the right answer is to abstain.

| variant | policy | raw recovery | Rs recovered /1k | Rs incr. vs no_action /1k | contacts /1k | wasted contacts /1k | escalations /1k |
|---|---|---|---|---|---|---|---|
| calibrated | no_action | 23.9% | 2,146,287 | 0 | 0 | 0 | 0 |
| calibrated | razorpay_default | 60.0% | 4,514,448 | 2,368,161 | 0 | 0 | 0 |
| calibrated | heuristic | 68.4% | 5,077,412 | 2,931,125 | 450 | 140 | 179 |
| misspecified | no_action | 23.1% | 2,324,649 | 0 | 0 | 0 | 0 |
| misspecified | razorpay_default | 58.0% | 4,481,552 | 2,156,902 | 0 | 0 | 0 |
| misspecified | heuristic | 67.9% | 5,081,791 | 2,757,142 | 450 | 142 | 179 |
| null_uplift | no_action | 23.9% | 2,146,287 | 0 | 0 | 0 | 0 |
| null_uplift | razorpay_default | 23.9% | 2,146,287 | 0 | 0 | 0 | 0 |
| null_uplift | heuristic | 23.7% | 2,127,730 | -18,556 | 450 | 140 | 179 |

Regenerate: `make data && python scripts/checkpoint1.py`. Methodology and calibration:
`docs/EVALUATION.md`.

## How it works
```
failure context -> predict uplift per arm (T-learner, 10-member ensemble, logged data + propensities)
              -> net EV = uplift x amount - cost - fatigue penalty, for all five arms incl. no_action
              -> act only if some arm's lower confidence bound clears the merchant minimum (z = 2)
              -> guardrails (kill switch, mandatory escalation, retry budget, expired card,
                 contact caps, quiet hours, minimum EV; machine-readable reason codes)
              -> idempotent executor (Mock | Razorpay test mode; ledger reserves the key first)
              -> append-only audit row (uplift[5], net_ev[5], checks, reason, result, outcome)
              -> measured: A/B, paired exact, OPE, wasted contacts, abstention, sensitivity
```
Five arms, fixed: `no_action`, `retry_now`, `retry_delayed(1|3|7)`, `remind_and_retry`,
`escalate_human`. No discount arm (ADR-002). Every retry arm is a three-attempt schedule so the
Razorpay default is one of our own actions (ADR-006). ML picks numbers; the LLM only writes the
explanation and is validated against the action set (ADR-009).

## Run
```bash
make setup         # uv sync (Python 3.11, pinned)            ~1-2 min
make data          # 50k failures x 4 simulator variants      ~5 s
make train         # 7 x 10 LightGBM models per variant       ~1 min
make eval          # A/B + paired + OPE -> reports/           ~5 min
make dial          # conservatism dial                        ~4 min
make sensitivity   # 19 worlds regenerated + retrained        ~15 min
make demo          # 500-failure batch with one injected 5xx  ~5 s
make dashboard     # Streamlit over audit + reports
make test          # 80+ tests: guardrails, idempotency, leakage, OPE, agent, API
```
Windows without GNU make: `.\make.ps1 <target>`. Copy `.env.example` to `.env` for Razorpay
test keys (`COUNTERFACT_EXECUTOR=razorpay`) and an Anthropic key (live explanations,
`claude-haiku-4-5`, capped at 50 calls per run).

## Limitations, stated plainly
- **Synthetic data.** Results are on a simulator calibrated to the public 55-65% recovery band
  with hidden structure the models cannot see (`docs/EVALUATION.md`). We claim methodology and
  relative lift, not absolute rupees for any merchant.
- **Razorpay test mode cannot emit failure reasons.** The taxonomy comes from the simulator.
  The live executor has three paths (ADR-015, ADR-018): *Payment Link* (live, executed: the
  reminder arm creates a Razorpay Payment Link with `reference_id = idempotency_key`, reused on
  re-drive, and `payment_link.paid` records the recovery), *Subscriptions timing and escalation*
  (Razorpay owns the charge schedule; retry arms are `deferred` with Razorpay's schedule and the
  invoice pay link, never shown as executed), and *tokenised `createRecurring`* (implemented and
  tested; on this test account `POST /v1/payments/create/recurring` returns 404 because Recurring
  Payments is not enabled for the account). Verified live: signed webhooks, payment links one per
  key through an injected transport fault, `subscription.fetch`, `invoice.all`, `order.create`,
  and zero duplicates via Razorpay's own records (`scripts/verify_charges.py`).
- **The rule table is oracle-informed.** It was written with the simulator's true per-category
  probabilities. The ML policy ties it under `calibrated` and beats it under `drifted`.
- **A/B intervals are wide** (heavy-tailed B2B invoices); the paired-exact numbers are the
  simulator's ground truth and are reported next to every A/B estimate.
- **Roadmap, not shipped:** per-merchant contextual bandit, sequential re-decision after each
  failed attempt, a scheduler for delayed attempts in the Razorpay executor.

## Layout
`src/counterfact/{sim,features,models,policy,agent,eval,api}`, `dashboard/app.py`, `scripts/`,
`tests/`, `docs/`. `CLAUDE.md` holds the architecture diagram, judging-bar checklist and status.
