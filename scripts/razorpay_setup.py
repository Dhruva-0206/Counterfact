"""Create (or reuse) a Razorpay TEST-MODE plan and subscription for the live demo.

Usage::  python scripts/razorpay_setup.py [--amount 299] [--force-new]

* Aborts unless the configured key id starts with ``rzp_test_``.
* Finds a plan named ``Counterfact test plan`` (or creates one: monthly, Rs 299) and creates a
  subscription on it (12 cycles, customer notifications on) unless ``data/razorpay_test.json``
  already records one that is still usable.
* Prints the ids and the hosted authentication URL (``short_url``). Never prints key values.

The customer authenticates the subscription on the hosted page with a Razorpay test card; a
failure-scenario card there produces a real ``payment.failed`` event in test mode.
"""

from __future__ import annotations

import argparse
import json
import sys

from counterfact.config import ROOT, get_settings

PLAN_NAME = "Counterfact test plan"
STATE = ROOT / "data" / "razorpay_test.json"


def client():
    import razorpay  # type: ignore

    s = get_settings()
    if not s.razorpay_key_id or not s.razorpay_key_secret:
        sys.exit("RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET missing in .env")
    if not s.razorpay_key_id.startswith("rzp_test_"):
        sys.exit("refusing to run: RAZORPAY_KEY_ID is not a test key (expected prefix rzp_test_)")
    return razorpay.Client(auth=(s.razorpay_key_id, s.razorpay_key_secret))


def find_or_create_plan(c, amount_rs: int) -> dict:
    for p in c.plan.all({"count": 100}).get("items", []):
        item = p.get("item", {})
        if item.get("name") == PLAN_NAME and int(item.get("amount", 0)) == amount_rs * 100:
            return p
    return c.plan.create({
        "period": "monthly",
        "interval": 1,
        "item": {"name": PLAN_NAME, "amount": amount_rs * 100, "currency": "INR",
                 "description": "Counterfact buildathon demo plan (test mode)"},
        "notes": {"project": "counterfact"},
    })


def create_subscription(c, plan_id: str) -> dict:
    return c.subscription.create({
        "plan_id": plan_id,
        "total_count": 12,
        "quantity": 1,
        "customer_notify": 1,
        "notes": {"project": "counterfact", "merchant_id": "m_fitpulse"},
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--amount", type=int, default=299)
    ap.add_argument("--force-new", action="store_true", help="create a new subscription even if one is recorded")
    args = ap.parse_args()
    c = client()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}

    plan = find_or_create_plan(c, args.amount)
    sub = None
    if state.get("subscription_id") and not args.force_new:
        sub = c.subscription.fetch(state["subscription_id"])
        if sub.get("status") in ("cancelled", "completed", "expired"):
            sub = None
    if sub is None:
        sub = create_subscription(c, plan["id"])

    state = {
        "plan_id": plan["id"],
        "plan_amount_rs": args.amount,
        "subscription_id": sub["id"],
        "subscription_status": sub.get("status"),
        "short_url": sub.get("short_url"),
        "customer_id": sub.get("customer_id"),
        "current_invoice_id": (sub.get("current_invoice_id") or None),
    }
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))
    print(f"\nrecorded in {STATE}")
    print("Authenticate the subscription on the short_url with a Razorpay test card; use a failure card "
          "(e.g. 4100 2800 0008 0001, 'insufficient account balance') to produce a real payment.failed.")


if __name__ == "__main__":
    main()
