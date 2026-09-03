"""Checkpoint 4: process a batch of failed payments end to end with the agent loop.

Usage::

    python scripts/run_batch.py --n 500 --inject-failure          # one forced 5xx mid-batch
    python scripts/run_batch.py --n 500 --failure-rate 0.02       # random transient 5xx
    python scripts/run_batch.py --n 200 --explain 20              # cache 20 LLM explanations

Events come from the holdout split of the configured variant; outcomes are resolved by the
simulator (the "world") after execution and written back to the audit store. Prints recovered
rupees, contacts, escalations, queued / re-driven executions, the duplicate-charge count (must be
zero) and the audit file path.
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime

import numpy as np

from counterfact.agent.audit import AuditStore
from counterfact.agent.executor import FailureInjector, MockExecutor, build_executor
from counterfact.agent.explain import (
    MAX_EXPLANATIONS_PER_RUN,
    ClaudeExplainer,
    TemplateExplainer,
    explain_pending,
)
from counterfact.agent.loop import Agent
from counterfact.config import ROOT, SIM_VARIANTS, get_settings
from counterfact.eval.pipeline import holdout_frame, load_merchants
from counterfact.eval.report import realised
from counterfact.models.uplift import TLearner

INJECT_FLAG = ROOT / "data" / "inject_5xx.flag"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--variant", default=None, choices=SIM_VARIANTS)
    ap.add_argument("--inject-failure", action="store_true", help="force a 5xx on one retry mid-batch")
    ap.add_argument("--inject-at", type=int, default=None, help="event index at which to inject (default n//2)")
    ap.add_argument("--failure-rate", type=float, default=None, help="random transient 5xx rate")
    ap.add_argument("--explain", type=int, default=0, help="generate up to N LLM explanations (cached)")
    ap.add_argument("--audit-dir", default=None)
    ap.add_argument("--subscription-id", default=None, help="live executor: route every event to this subscription")
    ap.add_argument("--customer-id", default=None)
    ap.add_argument("--token-id", default=None)
    ap.add_argument("--invoice-id", default=None)
    ap.add_argument("--max-amount", type=float, default=None, help="only events with amount <= this (live token caps)")
    ap.add_argument("--tokenized-events", type=int, default=0,
                    help="live executor: route the first N events through the tokenised recurring mode "
                         "(customer/token from data/razorpay_test.json['recurring'] unless given)")
    args = ap.parse_args()

    settings = get_settings()
    variant = args.variant or settings.sim_variant
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    audit_dir = ROOT / (args.audit_dir or f"reports/audit/batch_{variant}_{run_id}")
    store = AuditStore(audit_dir)
    injector = FailureInjector(flag_path=INJECT_FLAG)
    if settings.executor == "razorpay":
        # the fault is injected inside the live createRecurring call path, not before it
        executor = build_executor(settings, store, injector=None, live_injector=injector, backoff_base=0.5)
    else:
        executor = MockExecutor(
            store,
            failure_rate=args.failure_rate if args.failure_rate is not None else settings.executor_failure_rate,
            seed=settings.seed, injector=injector, backoff_base=0.02,
        )
    model = TLearner.load(settings.variant_dir(variant) / "models" / "uplift.pkl")
    merchants = load_merchants(settings, variant)
    agent = Agent(model, merchants, executor, store)

    df, cf = holdout_frame(settings, variant)
    if args.max_amount is not None:
        keep = (df["amount"] <= args.max_amount).to_numpy()
        df, cf = df[keep].reset_index(drop=True), cf[keep].reset_index(drop=True)
    df, cf = df.head(args.n).reset_index(drop=True), cf.head(args.n).reset_index(drop=True)
    if args.subscription_id:  # live executor: every event acts on the real test subscription
        df = df.copy()
        df["subscription_id"] = args.subscription_id
        for col, val in (("customer_id", args.customer_id), ("token_id", args.token_id), ("invoice_id", args.invoice_id)):
            if val:
                df[col] = val
    if args.tokenized_events > 0:  # mode 1 on the first N events, mode 2 on the rest
        import json as _json
        rec = {}
        state_path = ROOT / "data" / "razorpay_test.json"
        if state_path.exists():
            rec = _json.loads(state_path.read_text()).get("recurring", {})
        cust, tok = args.customer_id or rec.get("customer_id"), args.token_id or rec.get("token_id")
        if not (cust and tok):
            raise SystemExit("--tokenized-events needs a registered token: run scripts/razorpay_token.py first")
        df = df.copy()
        df["token_id"] = [tok if i < args.tokenized_events else "" for i in range(len(df))]
        df["customer_id"] = [cust if i < args.tokenized_events else c for i, c in enumerate(df["customer_id"])]
    inject_at = args.inject_at if args.inject_at is not None else args.n // 2

    t0 = time.time()
    handled = []
    # process in two halves so the injection lands mid-batch and the loop visibly continues
    if args.inject_failure and executor.name == "razorpay":
        # live: the next provider write (payment link / createRecurring) fails once at the transport
        # layer and succeeds on backoff; then a later one exhausts the budget -> queued -> re-driven
        injector.arm(1)
        handled += agent.handle_batch(df.iloc[:inject_at])
        injector.arm(executor.max_api_retries)
        handled += agent.handle_batch(df.iloc[inject_at:])
    elif args.inject_failure:
        handled += agent.handle_batch(df.iloc[:inject_at])
        injector.arm(executor.max_api_retries)  # enough 5xx to exhaust backoff -> queued
        handled += agent.handle_batch(df.iloc[inject_at:])
    else:
        handled += agent.handle_batch(df)
    elapsed = time.time() - t0

    # the world resolves outcomes for executed actions; queued actions have no outcome yet
    actions = np.array([h.row["action_name"] for h in handled], dtype=object)
    y = realised(cf, actions)
    amount = df["amount"].to_numpy(dtype=float)
    for i, h in enumerate(handled):
        if h.result.status in ("executed", "skipped", "duplicate"):
            agent.record_outcome(h.row["event_id"], bool(y[i]), float(amount[i]) if y[i] else 0.0)

    queued_before = list(store.queued())
    redriven = agent.redrive_queued()
    for eid, res in redriven:
        i = int(np.flatnonzero(df["event_id"].to_numpy() == eid)[0])
        if res.status in ("executed", "skipped"):
            agent.record_outcome(eid, bool(y[i]), float(amount[i]) if y[i] else 0.0, redriven=True)

    m = store.metrics()
    dup_charges = sum(1 for v in getattr(executor, "charges", {}).values() if v > 1)
    print(f"batch of {len(handled)} failures ({variant}, executor={executor.name}) processed in {elapsed:.1f}s -> {audit_dir}")
    print(f"  recovered: {m['recovered']}/{m['outcomes_known']} = {m['recovery_rate']:.1%}   "
          f"Rs recovered {m['rs_recovered']:,.0f} of Rs {m['rs_at_risk']:,.0f} at risk")
    print(f"  contacts {m['contacts']}  escalations {m['escalations']}  abstentions {m['abstentions']}  "
          f"guardrail overrides {m['overridden_by_guardrails']}")
    print(f"  executor ledger {m['executor']}  injected 5xx fired {injector.fired}  "
          f"queued mid-batch {len(queued_before)}  re-driven {len(redriven)}")
    print(f"  duplicate charges: {dup_charges}  (events charged more than once; must be 0)")
    if executor.name == "razorpay":
        modes = {}
        for h in handled:
            d = (h.row["executor_result"] or {}).get("detail") or {}
            k = (d.get("mode") or "-", h.result.status)
            modes[k] = modes.get(k, 0) + 1
        print("  live paths (mode, status):", dict(sorted(modes.items())))
        for h in handled:
            d = (h.row["executor_result"] or {}).get("detail") or {}
            if d.get("mode") == "payment_link" and h.result.status == "executed":
                print(f"  payment link {d.get('plink_id')} {d.get('short_url')} for {h.row['event_id']} (reference {d.get('reference_id')})")
        print(f"  verify: python scripts/verify_charges.py --audit-dir {audit_dir.relative_to(ROOT)}")
    transient = [e for e in executor.log if e['status'] == 'transient']
    if transient:
        print(f"  first transient error: {transient[0]}")
    print(f"  audit: {store.jsonl_path}  sqlite: {store.db_path}")

    if args.explain > 0:
        explainer = ClaudeExplainer(max_per_run=min(args.explain, MAX_EXPLANATIONS_PER_RUN)) if settings.anthropic_api_key else TemplateExplainer()
        n = explain_pending(store, explainer, limit=min(args.explain, MAX_EXPLANATIONS_PER_RUN))
        src = getattr(explainer, "calls", None)
        print(f"  explanations written: {n} (llm calls: {src if src is not None else 0}, "
              f"rejected: {getattr(explainer, 'rejected', 0)}, errors: {getattr(explainer, 'errors', 0)})")
        for row in store.recent(limit=2):
            print(f"    [{row['event_id']} {row['action_name']}] {row.get('explanation')}")


if __name__ == "__main__":
    main()
