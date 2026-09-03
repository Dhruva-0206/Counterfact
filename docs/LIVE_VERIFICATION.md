# Live verification, Razorpay test mode (2026-09-03)

Every row below is regenerable by the command in its last column. Nothing is marked pass on the
strength of a log line alone: the Razorpay-side checks read Razorpay's own records.

## Pass / fail

| # | What was verified | Result | Evidence | Regenerate with |
|---|---|---|---|---|
| 1 | Claude explanations on live decisions | **pass** | 10 of 10 validated, 0 failing re-validation, `explanation_source=claude` on all 10 | `revalidate(AuditStore("reports/audit/live_explain"))` |
| 2 | Explanations are cached, not re-billed | **pass** | second run makes 0 API calls | `python scripts/run_batch.py --explain 10 --audit-dir reports/audit/live_explain` |
| 3 | Payment Link executed on the live API | **pass** | 4 links on `cust_SyobLyHEnrJxSn`, one per idempotency key | `python scripts/verify_charges.py --audit-dir reports/audit/live_demo` |
| 4 | One failure handled gracefully | **pass** | `evt_000007` executed on attempt 2 after an injected transport fault, still exactly 1 link | same as row 3 |
| 5 | Zero duplicate charges, checked Razorpay-side | **pass** | 1 link per executed key, 0 per queued/failed key | same as row 3 |
| 6 | Retry arms on Subscriptions are never faked | **pass** | 13 rows `deferred`/`skipped` with Razorpay's own refusal text, 0 reported executed | `python scripts/run_batch.py --n 20 --audit-dir reports/audit/live_razorpay` |
| 7 | `escalate_human` hits the live API | **pass** | 7 rows executed via `subscription.fetch` | same as row 6 |
| 8 | Webhook rejects a tampered signature | **pass** | HTTP 401, `invalid webhook signature` | `python scripts/webhook_selftest.py` |
| 9 | Webhook accepts a signed body and writes an audit row | **pass** | HTTP 200 + decision + executor result | same as row 8 |
| 10 | The same over the public tunnel | **pass** | 401 then 200 through `*.trycloudflare.com`, `/health` returns `executor: razorpay` | `python scripts/webhook_selftest.py --url https://<host>/webhook/payment_failed` |
| 11 | Test suite and linter | **pass** | 87 tests pass, ruff clean | `uv run pytest` and `uv run ruff check .` |
| 12 | Tokenised `createRecurring` charge | **blocked, account** | `BadRequestError: The requested URL was not found on the server` on `POST /v1/payments/create/recurring`; Recurring Payments is not enabled on this test account (ADR-015) | `python scripts/run_batch.py --tokenized-events 3 --audit-dir reports/audit/live_tokenized_probe` |
| 13 | A webhook delivered by Razorpay itself | **pending, needs dashboard** | route, signature check and audit write already proven by rows 8-10; only Razorpay's delivery is untested | create the Test Mode webhook, then pay with card `4100 2800 0008 0001` |
| 14 | `payment_link.paid` recorded as a recovery outcome | **pending, needs a test payment** | implemented and unit-tested; `reference_id` maps back to the decision | pay one link from row 3 with a test card |

Rows 13 and 14 are the only two that need a human in the Razorpay dashboard. Rows 1-12 run
unattended.

## What "blocked" means in row 12
Razorpay's Recurring Payments product is not enabled on this test account, so the merchant-initiated
charge endpoint returns 404. The code path is complete and tested up to that call: order creation
with the idempotency key as the receipt succeeds, the pre-charge duplicate lookup runs, and the
fault injector exercises backoff and re-drive against the real endpoint. Nothing in the demo or the
docs presents those rows as executed; they are recorded as `failed` with Razorpay's message.
Payment Links (rows 3-5) are the live execution path precisely because they work on every account.

## Standing caveats
- A cloudflared quick tunnel changes hostname on every restart, so the dashboard webhook URL is
  re-pasted per session (ADR-019). Verify with row 10 before recording anything.
- Test mode only. The executor refuses a non-test key, so none of this can touch real money.
- Behavioural history features are defaulted on webhook-born events: test mode does not carry a
  customer's failure history. Documented in `docs/WEBHOOK.md`.
