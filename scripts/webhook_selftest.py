"""Sign a Razorpay-shaped ``payment.failed`` body with the .env webhook secret and POST it locally.

Usage::  python scripts/webhook_selftest.py [--url http://127.0.0.1:8000/webhook/payment_failed]

Proves the webhook path without ngrok: a correctly signed request creates an audit row and a
decision; a tampered signature is rejected with 401. Never prints the secret.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
import time
import urllib.error
import urllib.request

from counterfact.config import ROOT, get_settings

STATE = ROOT / "data" / "razorpay_test.json"


def sample_event(subscription_id: str, invoice_id: str | None) -> dict:
    now = int(time.time())
    return {
        "entity": "event",
        "account_id": "acc_selftest",
        "event": "payment.failed",
        "contains": ["payment", "subscription"],
        "payload": {
            "payment": {"entity": {
                "id": f"pay_selftest{now}", "entity": "payment", "amount": 29900, "currency": "INR",
                "status": "failed", "method": "card", "invoice_id": invoice_id, "customer_id": None,
                "error_code": "BAD_REQUEST_ERROR", "error_description": "Your payment didn't go through due to insufficient balance.",
                "error_source": "customer", "error_step": "payment_authorization", "error_reason": "insufficient_funds",
                "created_at": now, "notes": {"merchant_id": "m_fitpulse"},
            }},
            "subscription": {"entity": {
                "id": subscription_id, "entity": "subscription", "status": "created", "paid_count": 0,
                "plan_amount": 29900, "current_invoice_id": invoice_id, "notes": {"merchant_id": "m_fitpulse"},
            }},
        },
        "created_at": now,
    }


def post(url: str, body: bytes, signature: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "X-Razorpay-Signature": signature})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8000/webhook/payment_failed")
    args = ap.parse_args()
    s = get_settings()
    if not s.razorpay_webhook_secret:
        sys.exit("RAZORPAY_WEBHOOK_SECRET missing in .env")
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    body = json.dumps(sample_event(state.get("subscription_id", "sub_selftest"), state.get("current_invoice_id"))).encode()
    good = hmac.new(s.razorpay_webhook_secret.encode(), body, hashlib.sha256).hexdigest()

    code, out = post(args.url, body, "0" * 64)
    print(f"tampered signature -> HTTP {code}: {out.get('detail')}")
    code, out = post(args.url, body, good)
    print(f"signed request     -> HTTP {code}")
    if code == 200:
        keys = ("event_id", "action_name", "chosen_arm", "rejection_codes", "reason", "idempotency_key", "webhook_event")
        print(json.dumps({k: out.get(k) for k in keys}, indent=1))
        print("executor:", json.dumps(out.get("executor_result"), indent=1)[:700])
        print("uplift:", out.get("uplift"), "\nnet_ev:", out.get("net_ev"))
    else:
        print(out)


if __name__ == "__main__":
    main()
