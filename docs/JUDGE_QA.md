# The ten hardest questions, with evidence

Every answer points at a command or a file that regenerates the number.

**1. Your results are on synthetic data. Why should we believe any of it?**
We claim methodology and relative lift, not absolute rupees (`README.md`, Limitations). The
simulator is calibrated to the public 55-65% raw-recovery band for Razorpay's default retry
schedule (60.0% under `calibrated`, 58.0% under `misspecified` without re-tuning), it contains
hidden structure the models never see (latent liquidity and engagement, bank-outage clocks, a
hidden payday segment), and we test on three other worlds: `misspecified`, `null_uplift` (no
intervention works) and `drifted` (merchants depart from the taxonomy). A 19-setting sensitivity
sweep regenerates the world and retrains per row; the ranking oracle >= ML > Razorpay default
holds in all 19 (`make sensitivity`, `docs/EVALUATION.md`).

**2. Isn't the evaluation circular? The model is scored on the process that generated its features.**
No: training reads `failures.parquet` only; `counterfactuals.parquet` is opened by the evaluator
alone. Two tests enforce it: a static grep over `features/` and `models/` for any reference to the
counterfactual table or the outcome model, and a dynamic test that patches `pandas.read_parquet`
to raise during training (`tests/test_no_counterfactual_leak.py`, `tests/test_policy.py`). The
true process uses latents the feature matrix does not contain (`sim/outcome_model.py`).

**3. Why does the ML policy only tie the rule table under `calibrated`?**
Because the rule table is oracle-informed: it was written with the simulator's true per-category
probabilities (escalate expired cards, delay mandate retries, remind on insufficient funds),
knowledge no merchant has. The learner recovers that from logged data and ties it on rupees
(Rs 576k vs Rs 594k per 1,000 failures) at 63% fewer reminders and 74% fewer wasted ones. Where
merchants diverge from the taxonomy (`drifted`), the learner wins by 32% overall and 57% on the
drifted B2B merchant, because merchant id is a feature (`docs/EVALUATION.md`, ADR-014).

**4. How do you know the policy is not just intervening on everything?**
`null_uplift` is a world where no arm has any causal effect. The shipped policy abstains on 80.9%
of failures there (the rest are guardrail-mandated escalations: risk declines and very large
invoices), sends 4 reminders per 1,000 and loses Rs 202 per 1,000; the rule table sends 342 and
loses Rs 10,399. The abstention comes from a confidence gate on a 10-member bootstrap ensemble
(act only if some arm's lower bound clears the merchant minimum); the point estimate alone
abstained on 23%. The z dial trades calibrated lift for null abstention (`make dial`).

**5. The A/B confidence intervals are wide. Is the lift real?**
Rupees are heavy-tailed (B2B invoices up to Rs 243k), so a two-arm randomized A/B on 10,009
units per arm gives calibrated +617,938 [192,676; 1,015,642] and misspecified +537,563
[102,053; 914,654]: both exclude zero; null_uplift straddles zero as it should. The paired-exact
number (same units, every action's outcome from the counterfactual table) is reported next to
every A/B estimate and is the simulator's ground truth (`make eval`, ADR-013).

**6. Can you evaluate a new policy without exposing customers to it?**
Yes. From the logged holdout with recorded propensities, the doubly-robust estimate of each
policy's value lands within 0.3% to 3.2% of the truth on rupees for every policy and variant, and
its difference vs Razorpay default sits inside the A/B interval in 12 of 12 cells; plain IPS drifts
up to 13% where the target rarely matches the logged action. The direct method flatters the ML
policy (winner's curse) and DR corrects it (`python scripts/ope.py --all-variants`).

**7. Why is the live charge a 404?**
`POST /v1/payments/create/recurring` (and the S2S `/payments/create/json`) return
`BadRequestError: The requested URL was not found on the server` on our test account, while
`order.create`, `subscription.fetch`, `invoice.all`, `payment_link.create` and signed webhooks all
succeed with the same keys. That is Razorpay's response when Recurring Payments is not enabled for
the account; it is an account setting, not a code path. The tokenised path is implemented and
unit-tested (order with `receipt = idempotency_key`, reuse of an existing payment on that receipt,
then `createRecurring`), and it was exercised live up to the charge call: orders created,
transport fault injected inside the real call, backoff, queueing, re-drive, and
`scripts/verify_charges.py` confirming zero payments on those keys. The live executed path is
Payment Links, which every account has (ADR-015, ADR-018).

**8. How is double charging impossible rather than unlikely?**
The idempotency key is `sha256(event_id | action | features_hash)`, reserved in SQLite with
`INSERT OR IGNORE` before any provider call; a replayed webhook returns `duplicate` with no call;
a queued key can only be re-driven through `claim_queued`. On the provider side, Payment Links
carry `reference_id = idempotency_key` and are looked up before creation, and orders carry
`receipt = idempotency_key` and are checked for an existing payment. Evidence: `make demo`
(injected 5xx, duplicate charges = 0, tested in `tests/test_agent.py`) and the live run where
Razorpay's own records show exactly one link per key including the key whose first attempt hit an
injected transport fault (`scripts/verify_charges.py`).

**9. Where is the LLM, and can it move money?**
It cannot. The LLM (`claude-haiku-4-5`) writes the two-sentence explanation from the audit row
after the decision is made; a validator rejects any text that names an action outside the set,
recommends another arm, omits the no-action baseline as a probability, or uses certainty language
when that baseline exceeds 5%; rejected or failed calls fall back to a deterministic template; at
most 50 calls per run, cached in the audit store (ADR-009, ADR-016, ADR-017). Live: 10/10
explanations validated, second run 0 API calls.

**10. What did you deliberately not build?**
Discount arms (they teach customers that letting a payment fail is rewarded, ADR-002);
sequential re-decision after each failed attempt (roadmap, ADR-005); a per-merchant contextual
bandit (roadmap); a scheduler for the later attempts of a retry schedule in the live executor;
absolute-rupee claims for any real merchant. Every simulator assumption that moves the headline is
a sensitivity row, and escalation cost sits inside net EV (Rs 120-400 per merchant).
