"""Confirm zero duplicate charges for a live audit run via Razorpay's own records.

Usage::  python scripts/verify_charges.py --audit-dir reports/audit/<run>

For every tokenised decision in the audit store (mode ``tokenized_recurring``), lists the payments
Razorpay holds for its order (``order.payments``) and cross-checks ``payment.all`` on the token:
the number of captured/authorized payments per idempotency key must be exactly one for executed
rows and zero for queued/failed rows. Never prints key values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import razorpay  # type: ignore

from counterfact.agent.audit import AuditStore
from counterfact.config import ROOT, get_settings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-dir", required=True)
    args = ap.parse_args()
    s = get_settings()
    if not s.razorpay_key_id or not s.razorpay_key_id.startswith("rzp_test_"):
        sys.exit("refusing: RAZORPAY_KEY_ID is not a test key")
    c = razorpay.Client(auth=(s.razorpay_key_id, s.razorpay_key_secret))
    store = AuditStore(ROOT / args.audit_dir if not Path(args.audit_dir).is_absolute() else Path(args.audit_dir))

    decisions = store.all_decisions()
    links = [r for r in decisions if (r.get("executor_result") or {}).get("detail", {}).get("mode") == "payment_link"
             or ((r.get("executor_result") or {}).get("status") == "queued" and r["action_name"] == "remind_and_retry")]
    if links:
        print("payment links per reference_id (= idempotency key): exactly one expected per executed reminder")
        print(f"{'event_id':<14} {'status':<9} {'attempts':>8} {'links':>5} {'plink_id':<20} {'link_status':<10} ok")
        link_problems = 0
        for r in sorted(links, key=lambda r: r["event_id"]):
            res = r["executor_result"]
            resp = c.payment_link.all({"reference_id": r["idempotency_key"]})
            found = list(resp.get("payment_links") or resp.get("items") or [])
            expected = 1 if res["status"] == "executed" else 0
            ok = len(found) == expected
            link_problems += 0 if ok else 1
            first = found[0] if found else {}
            print(f"{r['event_id']:<14} {res['status']:<9} {res.get('attempts', 0):>8} {len(found):>5} {first.get('id', '-'):<20} {first.get('status', '-'):<10} {'yes' if ok else 'NO'}")
        if link_problems:
            sys.exit(f"DUPLICATE OR MISSING PAYMENT LINKS: {link_problems} mismatched rows")
        print("zero duplicate links: every executed reminder has exactly one payment link; queued/failed keys have none\n")
    rows = [r for r in decisions if (r.get("executor_result") or {}).get("detail", {}).get("mode") == "tokenized_recurring"
            or ((r.get("executor_result") or {}).get("status") == "queued" and r["action_name"] != "remind_and_retry")]
    if not rows:
        if links:
            return
        sys.exit("no tokenised or payment-link decisions in this audit run")
    keys = {r["idempotency_key"] for r in rows}
    token_ids = {t for t in ((r["executor_result"].get("detail") or {}).get("request", {}).get("token") for r in rows) if isinstance(t, str) and t}

    # Razorpay-side view: orders by receipt (= idempotency key) and their payments
    print(f"{'event_id':<14} {'status':<9} {'attempts':>8} {'order':<18} {'payments(captured/authorized)':<30} ok")
    problems = 0
    per_key_payments: dict[str, int] = {}
    for r in sorted(rows, key=lambda r: r["event_id"]):
        res = r["executor_result"]
        key = r["idempotency_key"]
        orders = c.order.all({"receipt": key, "count": 10}).get("items", [])
        n_pay = 0
        order_ids = []
        for o in orders:
            order_ids.append(o["id"])
            pays = c.order.payments(o["id"]).get("items", [])
            n_pay += sum(1 for pm in pays if pm.get("status") in ("captured", "authorized"))
        per_key_payments[key] = n_pay
        expected = 1 if res["status"] == "executed" else 0
        ok = n_pay == expected
        problems += 0 if ok else 1
        print(f"{r['event_id']:<14} {res['status']:<9} {res.get('attempts', 0):>8} {','.join(order_ids) or '-':<18} {n_pay:<30} {'yes' if ok else 'NO'}")

    # cross-check: every payment Razorpay holds on the token maps to exactly one of our keys
    all_pay = c.payment.all({"count": 100}).get("items", [])
    on_token = [pm for pm in all_pay if pm.get("token_id") in token_ids or (pm.get("notes") or {}).get("idempotency_key") in keys]
    by_key: dict[str, list[str]] = {}
    for pm in on_token:
        k = (pm.get("notes") or {}).get("idempotency_key", "?")
        by_key.setdefault(k, []).append(f"{pm['id']}:{pm['status']}")
    dup = {k: v for k, v in by_key.items() if len([x for x in v if x.endswith((":captured", ":authorized"))]) > 1}
    print(f"\npayments on token(s) {sorted(token_ids)} in the last 100: {len(on_token)}; keys with >1 successful payment: {len(dup)}")
    print(json.dumps(by_key, indent=1))
    if problems or dup:
        sys.exit(f"DUPLICATE OR MISSING CHARGES: {problems} mismatched rows, {len(dup)} duplicated keys")
    print("zero duplicate charges: every executed key has exactly one successful payment; queued/failed keys have none")


if __name__ == "__main__":
    main()
