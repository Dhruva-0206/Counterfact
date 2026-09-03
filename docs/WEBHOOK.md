# Razorpay webhook, end to end (test mode)

Route: `POST /webhook/payment_failed` (`src/counterfact/api/main.py`). Secret variable:
`RAZORPAY_WEBHOOK_SECRET` in `.env` (`.env.example` lists it). Requests that carry an
`X-Razorpay-Signature` header are verified with `razorpay.Utility.verify_webhook_signature`
(HMAC-SHA256 over the raw body); a bad or missing secret returns 401. Requests without the header
(local demos, `scripts/`) are accepted as our own event schema.

Handled events: `payment.failed`, `subscription.pending`, `subscription.halted` (a failed
charge). `payment.captured` and `subscription.charged` are acknowledged and ignored (`{"ignored":
...}`) so that the dashboard subscription can tick them without creating decisions. The payload is
mapped to the agent's event schema by `razorpay_event_to_context`: ids, amount, method, timestamp
and `error_reason` come from the webhook; the behavioural history features default to a first
failure with no recent contacts because test mode does not carry them; the failure category maps
from `error_reason` (unknown reasons fall back to `bank_technical`).

## First-time setup
1. Test keys in `.env`: `RAZORPAY_KEY_ID=rzp_test_...`, `RAZORPAY_KEY_SECRET=...`. The executor
   refuses non-test keys.
2. Webhook secret: if `RAZORPAY_WEBHOOK_SECRET` is empty or a placeholder, generate one
   (32 random alphanumerics) and paste the same value into the Razorpay dashboard when creating the
   webhook. It is shown once at generation time and never logged.
3. Test objects: `python scripts/razorpay_setup.py` creates the plan and subscription and records
   their ids in `data/razorpay_test.json` and `docs/DEMO_SCRIPT.md`.
4. ngrok: install it (Windows: `winget install ngrok.ngrok`, or download from ngrok.com), then
   `ngrok config add-authtoken <token from dashboard.ngrok.com>`. Verify with `ngrok config check`.
5. Razorpay dashboard (Test Mode toggle on): Settings -> Webhooks -> Add New Webhook. URL =
   `https://<ngrok-subdomain>.ngrok-free.app/webhook/payment_failed`, secret = the value from
   step 2, active events: `payment.failed`, `payment.captured`, `subscription.pending`,
   `subscription.halted`, `subscription.charged`.

## Per session
1. Terminal 1: `COUNTERFACT_EXECUTOR=razorpay uv run uvicorn counterfact.api.main:app --port 8000`
   (omit the env var to use the mock executor; decisions and audit rows are identical, only the
   provider calls differ).
2. Terminal 2: `ngrok http 8000`. Copy the `https://...ngrok-free.app` forwarding URL.
3. Razorpay dashboard: edit the webhook URL to `<forwarding URL>/webhook/payment_failed` (the
   ngrok subdomain changes on every restart on the free plan).
4. Verify one event: open the subscription's authentication link (`data/razorpay_test.json`,
   `short_url`) and pay with the failure card `4100 2800 0008 0001` (any future expiry, any CVV).
   Razorpay emits `payment.failed`; the uvicorn log shows the signed request, the signature check,
   and the audit row (`GET /decisions?limit=1`).
5. Local self-check without ngrok: `python scripts/webhook_selftest.py` signs a Razorpay-shaped
   `payment.failed` body with the `.env` secret, POSTs it to `localhost:8000`, and shows the audit
   row; it also confirms that a tampered signature is rejected with 401.

## Before recording
- Restarting ngrok changes the public URL: update the webhook URL in the dashboard first, then
  send one test event and check `GET /decisions?limit=1` before you press record.
- Keep the Razorpay dashboard in Test Mode; a Live Mode webhook uses a different secret and hits
  live endpoints, which the executor refuses.
- If verification fails: 401 with "invalid webhook signature" means the secret differs between
  `.env` and the dashboard (or the dashboard webhook was created in Live Mode); a 404 means the URL
  path is wrong (must end in `/webhook/payment_failed`); a 422 means the body was not a Razorpay
  event and not our schema.
