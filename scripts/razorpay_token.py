"""Find the recurring token registered for the test customer (or the test subscription's customer).

Usage::

    python scripts/razorpay_token.py                       # standalone registration link path (mode 1)
    python scripts/razorpay_token.py --subscription-id sub_x   # customer + tokens from an authenticated subscription

Writes ``customer_id`` and ``token_id`` into ``data/razorpay_test.json['recurring']`` and prints
the token summary (never key values). Exits non-zero, with the reason, if no recurring token exists
yet (the customer has not authenticated the mandate).
"""

from __future__ import annotations

import argparse
import json
import sys

import razorpay  # type: ignore

from counterfact.config import ROOT, get_settings

STATE = ROOT / "data" / "razorpay_test.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subscription-id", default=None)
    args = ap.parse_args()
    s = get_settings()
    if not s.razorpay_key_id or not s.razorpay_key_id.startswith("rzp_test_"):
        sys.exit("refusing: RAZORPAY_KEY_ID is not a test key")
    c = razorpay.Client(auth=(s.razorpay_key_id, s.razorpay_key_secret))
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    rec = state.setdefault("recurring", {})

    if args.subscription_id:
        sub = c.subscription.fetch(args.subscription_id)
        print("subscription:", {k: sub.get(k) for k in ("id", "status", "customer_id", "payment_method", "auth_attempts", "paid_count")})
        customer_id = sub.get("customer_id")
        if not customer_id:
            sys.exit("subscription has no customer yet: it has not been authenticated")
        source = "subscription"
    else:
        customer_id = rec.get("registration_customer_id") or rec.get("customer_id")
        if not customer_id:
            sys.exit("no registration customer recorded; run the setup step first")
        if rec.get("registration_link_id"):
            link = c.invoice.fetch(rec["registration_link_id"])
            print("registration link:", {k: link.get(k) for k in ("id", "status", "payment_id", "customer_id")})
            customer_id = link.get("customer_id") or customer_id
        source = "registration_link"

    toks = c.token.all(customer_id).get("items", [])
    summary = [{k: t.get(k) for k in ("id", "method", "recurring", "recurring_status", "used_at", "expired_at")}
               | {"card": {k: (t.get("card") or {}).get(k) for k in ("last4", "network", "type")}} for t in toks]
    print("tokens for", customer_id, ":", json.dumps(summary, indent=1))
    usable = [t for t in toks if t.get("recurring") and str(t.get("recurring_status") or "confirmed") != "rejected"]
    if not usable:
        sys.exit("no recurring token yet: authenticate the mandate with a test card first")
    tok = usable[0]
    rec.update({"customer_id": customer_id, "token_id": tok["id"], "token_source": source})
    state["recurring"] = rec
    STATE.write_text(json.dumps(state, indent=2))
    print(f"recorded customer_id={customer_id} token_id={tok['id']} ({source}) in {STATE}")


if __name__ == "__main__":
    main()
