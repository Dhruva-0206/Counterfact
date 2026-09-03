# Architecture

Filled progressively; component contracts are the docstrings of the named modules.

## Components
| module | contract |
|---|---|
| `sim/generator.py` | seeded population + 50k failures with decision-time features |
| `sim/outcome_model.py` | TRUE recovery process, three variants, common random numbers for coherent counterfactuals |
| `sim/logging_policy.py` | epsilon-uniform exploration policy; records `propensity` |
| `features/build.py` | decision-time feature matrix; leakage-checked |
| `models/uplift.py` | T-learner; `predict_uplift(X) -> (n, 5)` |
| `policy/ev.py` | net EV per arm; argmax with no-action tie-break and threshold |
| `policy/guardrails.py` | hard limits; machine-readable rejections |
| `policy/baselines.py` | `no_action`, `razorpay_default`, `heuristic` |
| `agent/loop.py` | event -> decide -> guard -> execute -> audit |
| `agent/executor.py` | `MockExecutor`, `RazorpayExecutor` (test mode); idempotency keys |
| `agent/audit.py` | append-only JSONL + SQLite view |
| `agent/explain.py` | LLM explanation, validated against action set |
| `eval/ab.py`, `eval/ope.py`, `eval/report.py` | randomized A/B with bootstrap CIs; IPS/SNIPS/DR; tables + figures |
| `api/main.py` | FastAPI: `/webhook/payment_failed`, `/decisions`, `/metrics` |

## Known limitation: test mode cannot emit failure reasons
Razorpay test mode does not let us provoke specific decline codes (insufficient funds, expired card, risk decline). The failure taxonomy therefore comes from the simulator; retries, subscription reads and webhook verification run against real test-mode endpoints. We claim methodology and relative lift on calibrated synthetic data, not absolute rupees.
