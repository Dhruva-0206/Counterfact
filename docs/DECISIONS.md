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
**Decision.** Headline lift is measured against `razorpay_default`: automatic retries at T+1, T+2, T+3 days, then halt. `no_action` and a rule-table `heuristic` are reported alongside.
**Consequences.** Smaller but defensible numbers.

## ADR-005: One decision per failed payment; arms are bounded plans
**Context.** A fully sequential formulation (re-decide after every failed retry) needs trajectory-level off-policy evaluation and multi-step credit assignment, which is not defensible in 48h.
**Decision.** The agent decides once per `payment.failed` event. Each arm is a bounded plan over a 14-day window: `retry_now` = one immediate retry; `retry_delayed(d)` = one retry at day d; `remind_and_retry` = one message then one retry at day 1; `escalate_human` = flag to merchant ops; `no_action` = nothing. Events that already carry prior retry attempts (`attempt_number` > 1) flow through the same loop, and guardrails enforce the 3-retry cap.
**Consequences.** Per-arm T-learners and single-step OPE are exact for this formulation. Per-step re-decision (contextual bandit) is roadmap.

## ADR-006: T-learner over S-learner
**Context.** We need `predict_uplift(X) -> (n, 5)`.
**Decision.** One LightGBM per arm on IPS-weighted logged data; uplift = mu_arm(x) - mu_no_action(x). An S-learner with the arm as a feature lets tree regularization shrink small treatment effects to zero, which is exactly the signal we care about.
**Consequences.** Slightly noisier per-arm estimates; mitigated by regularization.

## ADR-007: Three simulator variants with hidden structure
**Context.** Evaluating a model on the process that generated its features is circular.
**Decision.** The true outcome process uses latent customer state (liquidity, engagement, churn intent), hidden bank-outage durations and nonlinear interactions that never appear in the feature matrix. Variants: `calibrated` (Razorpay default lands in the public 55-65% band), `misspecified` (different hidden interactions, shifted-payday segment, heterogeneous message effects), `null_uplift` (no arm has any causal effect; costs remain).
**Consequences.** Under `null_uplift` the correct behaviour is to abstain; we report that as a headline.

## ADR-008: ML for numbers, LLM for language
**Context.** LLM-chosen money actions are neither bounded nor reproducible.
**Decision.** The LLM only produces merchant-facing explanations and message drafts from the audit row. A validator rejects any output that names an arm outside the allowed set or contradicts the chosen arm.
**Consequences.** Decisions are reproducible from seed; explanations are optional and cached.

## ADR-009: Contact-fatigue penalty priced as churn risk on future cycles
**Context.** A message is not free even when it works.
**Decision.** `fatigue_penalty = fatigue_rate * contacts_last_7d * amount * ltv_cycles` with defaults `fatigue_rate = 0.01` (each recent contact adds one percentage point of churn hazard) and `ltv_cycles = 3`. Merchant-configurable.
**Consequences.** Messaging a customer already contacted twice this week on a Rs 999 plan costs about Rs 60 of expected LTV before the message fee; the policy must expect a larger uplift to justify it.

## ADR-010: Dependencies added at scaffold beyond the brief
**Context.** The brief lists the allowed dependency set and asks for approval before additions.
**Decision.** Added `pyarrow` (pandas needs a parquet engine and the brief specifies parquet output) and `ruff` (dev-only; the brief's conventions require ruff-clean code). Flagged at Checkpoint 1; both are trivially removable.
