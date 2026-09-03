# Decision log (ADR style)

Each entry: context, decision, consequences. Newest at the bottom.

## ADR-001: Five-arm action space, fixed
**Context.** The brief rewards bounded, explainable money actions. Every extra arm multiplies logged-data requirements and guardrail surface.
**Decision.** Exactly five arms: `no_action`, `retry_now`, `retry_delayed(delay_days in {1,3,7})`, `remind_and_retry`, `escalate_human`. `retry_delayed` is one arm with a parameter; the parameter is chosen by evaluating the arm's model at each delay.
**Consequences.** Simpler uplift models, cleaner OPE support, and a guardrail table that fits on one screen.

## ADR-002: No discount arms
**Context.** Dunning products sometimes offer a discount to recover a failed charge.
**Decision.** No discount or credit arm. A discount on a *failed* payment teaches customers that letting a payment fail is rewarded, and it is unbounded downside for the merchant.
**Consequences.** We may leave some recoveries on the table for price-sensitive customers; we prefer that to creating a moral-hazard loop.

## ADR-003: "Do nothing" is a first-class arm
**Context.** Existing recovery systems treat intervention as free and always act.
**Decision.** `no_action` is always eligible, has its own predicted outcome (customers self-serve), wins ties, and wins whenever the best net EV is below the merchant's `min_ev_threshold`.
**Consequences.** We can report *wasted contacts* (intervened where no-action would have recovered) and behave correctly on `null_uplift` data.

## ADR-004: Razorpay default retry schedule is the headline baseline
**Context.** Lift over "no intervention" flatters any system.
**Decision.** Headline lift is measured against `razorpay_default`: Razorpay's automatic retries after a failed subscription charge, then halt. `no_action` and a rule-table `heuristic` are reported alongside. The headline column is "Rs incremental vs razorpay_default per 1,000 failures"; "vs no_action" is reported second.
**Consequences.** Smaller but defensible numbers.

## ADR-005: One decision per failed payment; arms are bounded plans
**Context.** A fully sequential formulation (re-decide after every failed retry) needs trajectory-level off-policy evaluation and multi-step credit assignment, which is not defensible in 48h.
**Decision.** The agent decides once per `payment.failed` event. Each arm is a bounded plan over a 14-day window (retry schedule defined in ADR-006; `escalate_human` = flag to merchant ops; `no_action` = nothing). Events that already carry prior retry attempts (`attempt_number` > 1) flow through the same loop, and guardrails enforce the 3-retry cap.
**Consequences.** Per-arm T-learners and single-step OPE are exact for this formulation. Per-step re-decision (contextual bandit) is roadmap.

## ADR-006: Equalized attempt budget; Razorpay default lies inside the action space
**Context.** With one retry per ML arm and three retries for the Razorpay default, any comparison confounds decision quality with attempt budget, and it biases against the ML policy.
**Decision.** Every retry arm gets the same bounded schedule: `retry_delayed(d)` attempts at days d, d+2, d+4; `retry_now` at 0, 2, 4; `remind_and_retry` = one reminder plus the `retry_delayed(1)` schedule. All schedules are capped by the 3-retries-per-failure guardrail, so an event arriving with prior attempts gets only the remaining budget (also enforced inside the simulator). `razorpay_default` is therefore exactly `retry_delayed(1)` (days 1, 3, 5). Razorpay's literal T+1/T+2/T+3 spacing is kept as the `razorpay_t123` sensitivity action and reported next to the headline.
**Consequences.** The baseline is one of our own actions, so any lift comes from choosing the anchor day, abstaining, reminding or escalating better, never from more attempts. This makes the comparison conservative-to-fair. The global retry multiplier was recalibrated for the new schedule (see `docs/EVALUATION.md`).

## ADR-007: T-learner over S-learner
**Context.** We need `predict_uplift(X) -> (n, 5)`.
**Decision.** One LightGBM per arm on IPS-weighted logged data; uplift = mu_arm(x) - mu_no_action(x). An S-learner with the arm as a feature lets tree regularization shrink small treatment effects to zero, which is exactly the signal we care about.
**Consequences.** Slightly noisier per-arm estimates; mitigated by regularization. If abstention under `null_uplift` falls below 80% with point estimates, a lower-confidence-bound over an ensemble is the documented fallback (timeboxed to 90 minutes; otherwise ship the point estimate with the merchant minimum-EV threshold).

## ADR-008: Three simulator variants with hidden structure, plus sensitivity rows
**Context.** Evaluating a model on the process that generated its features is circular, and every simulator assumption that moves the headline must be visible.
**Decision.** The true outcome process uses latent customer state (liquidity, engagement, churn intent), hidden bank-outage durations and nonlinear interactions that never appear in the feature matrix. Variants: `calibrated` (Razorpay default lands in the public 55-65% band), `misspecified` (different hidden interactions, shifted-payday segment, heterogeneous message effects), `null_uplift` (no arm has any causal effect; costs remain). Every assumption that affects the headline (escalation resolve rate for retry-fixable categories, retry scale, self-serve level, reminder lift, churn hazard, attempt decay, payday boost) is exposed as a sensitivity knob and reported in a sensitivity table in `docs/EVALUATION.md`; the qualitative ranking of policies must hold across the range.
**Consequences.** Under `null_uplift` the correct behaviour is to abstain; we report that as a headline. Under `calibrated`, abstention above roughly 30% indicates a miscalibrated cost or threshold, not conservatism.

## ADR-009: ML for numbers, LLM for language
**Context.** LLM-chosen money actions are neither bounded nor reproducible.
**Decision.** The LLM only produces merchant-facing explanations and message drafts from the audit row. A validator rejects any output that names an arm outside the allowed set or contradicts the chosen arm.
**Consequences.** Decisions are reproducible from seed; explanations are optional and cached.

## ADR-010: Contact-fatigue penalty priced as churn risk on future cycles
**Context.** A message is not free even when it works.
**Decision.** `fatigue_penalty = fatigue_rate * contacts_last_7d * amount * ltv_cycles` with defaults `fatigue_rate = 0.01` (each recent contact adds one percentage point of churn hazard) and `ltv_cycles = 3`. Merchant-configurable.
**Consequences.** Messaging a customer already contacted twice this week on a Rs 999 plan costs about Rs 60 of expected LTV before the message fee; the policy must expect a larger uplift to justify it.

## ADR-011: Dependencies added beyond the brief
**Context.** The brief lists the allowed dependency set and asks for approval before additions.
**Decision.** `pyarrow` (pandas needs a parquet engine and the brief specifies parquet output) and `ruff` (dev-only; the brief's conventions require ruff-clean code). Approved at Checkpoint 1.
**Consequences.** None beyond two pinned packages.

## ADR-012: Data layout `data/<variant>/`
**Context.** Three simulator variants share one population and feature matrix.
**Decision.** Each variant writes `failures.parquet`, `counterfactuals.parquet`, `merchants.parquet` and `meta.json` under `data/<variant>/`. Approved at Checkpoint 1.
**Consequences.** The flat `data/failures.parquet` path from the brief is replaced by `Settings.variant_dir()`.
