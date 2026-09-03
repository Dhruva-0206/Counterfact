"""Explanation validator, template fallback, budget cap and caching."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from counterfact.agent.audit import AuditStore
from counterfact.agent.explain import (
    MAX_EXPLANATIONS_PER_RUN,
    ClaudeExplainer,
    FakeExplainer,
    TemplateExplainer,
    explain_pending,
    validate,
)

ROW = {
    "event_id": "evt_000001", "idempotency_key": "k", "features_hash": "h", "merchant_id": "m_fitpulse",
    "amount": 999.0, "failure_category": "insufficient_funds", "attempt_number": 1,
    "chosen_arm": 2, "action_name": "retry_delayed_7", "delay_days": 7, "proposed_arm": 2, "overridden": False,
    "uplift": [0.0, 0.1, 0.3, 0.2, 0.1], "net_ev": [0.0, 98.0, 298.0, 150.0, -50.0],
    "guardrail_checks": [], "rejection_codes": "", "reason": "r", "p_no_action_hat": 0.29,
}


@pytest.mark.parametrize(
    "text, arm, ok",
    [
        ("We will retry after 7 days because funds are likely to arrive. Doing nothing recovers less.", "retry_delayed", True),
        ("We will retry after 7 days and also send a reminder tomorrow.", "retry_delayed", False),  # names another arm
        ("We will retry after 7 days and offer a 10% discount.", "retry_delayed", False),  # out-of-set action
        ("This payment will be escalated to a human reviewer. It exceeds the merchant threshold.", "escalate_human", True),
        ("We recommend cancelling the subscription.", "retry_delayed", False),
        ("", "retry_delayed", False),
        ("One. Two. Three. Four sentences is too many.", "retry_delayed", False),
        ("We take no action here because no intervention adds value. Self-serve recovery is likely.", "no_action", True),
        ("Retry immediately; this beats taking no action.", "retry_now", True),  # counterfactual mention allowed
    ],
)
def test_validator(text: str, arm: str, ok: bool) -> None:
    assert validate(text, arm)[0] is ok


@pytest.mark.parametrize(
    "text, baseline, ok",
    [
        ("We retry after 7 days. Doing nothing has about a 29% chance of recovering on its own.", 0.29, True),
        ("We retry after 7 days. Without action there is only a 29% chance of recovery, so the loss of Rs 999 is avoided.", 0.29, False),
        ("We retry after 7 days because it is necessary. On its own the payment recovers 29% of the time.", 0.29, False),
        ("We retry after 7 days. The customer will pay eventually.", 0.29, False),  # baseline not stated
        ("We retry after 7 days. Doing nothing recovers about 3% of the time, so the loss of Rs 999 is nearly certain.", 0.03, True),
    ],
)
def test_validator_baseline_and_certainty_rules(text: str, baseline: float, ok: bool) -> None:
    assert validate(text, "retry_delayed", baseline)[0] is ok


def test_template_is_deterministic_and_valid() -> None:
    t1, s1 = TemplateExplainer().explain(ROW)
    t2, _ = TemplateExplainer().explain(ROW)
    assert t1 == t2 and s1 == "template" and validate(t1, "retry_delayed", 0.29)[0]
    row = {**ROW, "chosen_arm": 4, "action_name": "escalate_human", "rejection_codes": "MANDATORY_ESCALATION_AMOUNT"}
    text, _ = TemplateExplainer().explain(row)
    assert "escalate" in text and "MANDATORY_ESCALATION_AMOUNT" in text


class _Resp:
    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.content = [SimpleNamespace(type="text", text=text)]
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _client(replies):
    return SimpleNamespace(messages=_FakeMessages(replies))


def test_claude_explainer_accepts_valid_and_falls_back_on_invalid() -> None:
    good = "We will retry after 7 days because funds usually arrive after payday. Doing nothing recovers about 29% of the time."
    bad = "We will retry after 7 days and offer a discount."
    ex = ClaudeExplainer(client=_client([_Resp(good), _Resp(bad), RuntimeError("network")]), max_per_run=10)
    assert ex.explain(ROW) == (good, "claude")
    text, source = ex.explain(ROW)
    assert source.startswith("template (llm output rejected") and validate(text, "retry_delayed")[0]
    text, source = ex.explain(ROW)
    assert source.startswith("template (llm error") and ex.errors == 1 and ex.rejected == 1


def test_claude_explainer_respects_budget_cap() -> None:
    good = "We will retry after 7 days. Doing nothing recovers about 29% of the time."
    ex = ClaudeExplainer(client=_client([_Resp(good)] * 5), max_per_run=2)
    sources = [ex.explain(ROW)[1] for _ in range(4)]
    assert sources == ["claude", "claude", "template", "template"] and ex.calls == 2
    assert MAX_EXPLANATIONS_PER_RUN == 50


def test_explain_pending_caches_in_store(tmp_path: Path) -> None:
    store = AuditStore(tmp_path)
    for i in range(3):
        store.append_decision({**ROW, "event_id": f"evt_{i}"})
    fake = FakeExplainer()
    assert explain_pending(store, fake, limit=2) == 2 and len(fake.calls) == 2
    assert explain_pending(store, fake, limit=10) == 1  # only the remaining unexplained row
    rows = store.all_decisions()
    assert all(r["explanation"] and r["explanation_source"] == "fake" for r in rows)
