# 5-minute demo script

Setup before the demo (once): `make setup && make data && make train && make eval && make dial`
(about 12 minutes), then keep two terminals open in the repo and the dashboard running
(`make dashboard`). Optional: `ANTHROPIC_API_KEY` in `.env` for live explanations.

| time | what you show | what you say |
|---|---|---|
| 0:00 | Slide / README top | Failed subscription payments: Razorpay's default retries at fixed days and every dunning tool treats intervention as free. Counterfact prices every action against doing nothing, including contact fatigue, and reports the cost of intervening when it should not have. "Do nothing" is a first-class action with its own expected value. |
| 0:30 | Terminal 1: `make demo` (= `python scripts/run_batch.py --n 500 --inject-failure`) | 500 held-out failures go through decide, guardrails, an idempotent executor and the audit trail in five seconds. Read the last lines aloud: recovered rupees of rupees at risk, contacts, escalations, abstentions, guardrail overrides. Then the failure line: a 5xx was injected mid-batch, the action backed off three times, was parked as queued, the batch continued, it was re-driven under the same key, duplicate charges = 0. |
| 1:30 | Dashboard, headline tiles (variant `calibrated`) | Per 1,000 failures: ML policy recovers Rs 5.8 lakh more than Razorpay's default, +5.7 recovery points, two-arm randomized A/B interval excludes zero. Contacts saved vs the rule table: 215 per 1,000 (127 vs 342), wasted-contact rate 22% vs 31%. Abstention 13%, and 23% of those self-recovered. |
| 2:15 | Sidebar: switch variant to `null_uplift` | Same policy in a world where nothing works: 81% abstention, 4 reminders per 1,000, loses nothing; the rule table keeps sending 342 reminders and loses Rs 10k per 1,000. This is the honest-metrics bar: we built the world in which the right answer is to do nothing and we report it. |
| 2:45 | Sidebar: switch to `drifted`; Per merchant tab | The rule table was written with perfect knowledge of the failure taxonomy. When two merchants drift from it (FitPulse's customers churn when reminded; ScaleOps pays on a day-7 AP cycle), the learner picks it up from merchant id: +17% and +57% on those merchants, +32% overall, at a third of the reminders. |
| 3:15 | Decision explorer: pick an escalated or overridden event | One audit row: uplift and net EV for all five arms, the merchant minimum as the dashed line, the guardrail checks with machine-readable codes (`MANDATORY_ESCALATION_AMOUNT`, `CARD_EXPIRED_NO_RETRY`, `CONTACT_CAP_24H`), the executor result with its idempotency key, the outcome, and a two-sentence explanation. The LLM only writes the sentence; it never picks the arm, and a validator rejects any text that names another action. |
| 4:00 | Terminal 2: `python scripts/inject_failure.py` then re-run one webhook (or point at the ledger in the explorer) | Guardrails and idempotency are code, not prompts: every rule has a test (`make test`, 80+ tests), and the ledger reserves the key before any provider call, so a replayed webhook returns `duplicate` without a charge. |
| 4:20 | OPE tab | Test a policy without exposing a customer: from logged data with recorded propensities, the doubly-robust estimate lands within 3% of the truth and inside the A/B interval for every policy and variant (12/12). |
| 4:40 | Sensitivity tab and dial | Nineteen simulator assumptions varied one at a time, world regenerated and models retrained each time: the ranking oracle >= ML > Razorpay default holds in all of them. The confidence gate is a dial the merchant owns: more lift when uplift is real, or more abstention when it is not. |
| 4:55 | Close | Synthetic data calibrated to public benchmarks: we claim methodology and relative lift, not absolute rupees. Razorpay test mode cannot emit the failure taxonomy, so retries and webhooks hit real test-mode endpoints while the reasons come from the simulator. Roadmap: per-merchant contextual bandit, sequential re-decision. |

Fallbacks: if the dashboard is slow, every table is a CSV under `reports/tables/` and every
figure a PNG under `reports/figures/`; the batch output alone covers bars 1, 2 and 3.
