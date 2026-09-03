"""LLM explanations of audit rows. Language only; the LLM never chooses an arm (ADR-009).

* ``TemplateExplainer``  deterministic two-sentence explanation from the audit row (always
  available, used as the fallback).
* ``ClaudeExplainer``    asks Claude (default ``claude-haiku-4-5``) for a two-sentence,
  merchant-facing explanation, capped at ``MAX_EXPLANATIONS_PER_RUN`` calls per process, cached in
  the audit store.
* ``FakeExplainer``      test double with the same interface (no network).

Every generated text goes through ``validate``: it must name the chosen action, must not
recommend another arm (contrastive mentions such as "rather than retrying immediately" are
allowed), and must not propose actions outside the action set (discounts, refunds, ...).
Invalid output is replaced by the template and the rejection is recorded in the audit store.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from counterfact.agent.audit import AuditStore
from counterfact.config import ARMS

MAX_EXPLANATIONS_PER_RUN = 50
DEFAULT_MODEL = "claude-haiku-4-5"

ARM_LABELS: dict[str, str] = {
    "no_action": "take no action",
    "retry_now": "retry immediately",
    "retry_delayed": "retry after a delay",
    "remind_and_retry": "send a reminder and retry",
    "escalate_human": "escalate to a human",
}
ARM_PATTERNS: dict[str, str] = {
    "no_action": r"\b(no action|do nothing|doing nothing|not intervene|not to intervene|no intervention|leave it)\b",
    "retry_now": (
        r"\b(retr(y|ies|ying|ied)|re-?attempt\w*)\W+(it\W+|the\W+\w+\W+)?(immediately|now|right away)\b"
        r"|\bimmediate(ly)? (retr\w+|re-?attempt\w*)"
    ),
    "retry_delayed": (
        r"\b(retr(y|ies|ying|ied)|re-?attempt\w*)\b[^.]{0,80}\b(after|in) \d+ days?\b"
        r"|\bdelayed (retry|re-?attempt)|\bwait(ing)? \d+ days?"
    ),
    "remind_and_retry": r"\bremind\w*\b",
    "escalate_human": r"\bescalat\w*\b|\bhuman\b|\bops team\b|\baccount manager\b|\bmanual review\b",
}
CONTRAST_MARKERS = (
    "instead of", "rather than", "than", "not ", "no ", "avoid", "without", "less effective",
    "would have", "would only", "unlikely", "lower", "rules out", "ruled out", "over ", "beats",
    "versus", " vs", "compared", "outperform",
)
FORBIDDEN_ACTIONS: tuple[str, ...] = (
    "discount", "coupon", "refund", "waive", "waiver", "credit note", "chargeback",
    "cancel the subscription", "cancel their subscription", "terminate", "block the customer",
    "blacklist", "legal action", "collections agency", "downgrade", "free month",
)


def validate(text: str, chosen_arm: str) -> tuple[bool, str]:
    """Accept only text that names the chosen action, recommends no other arm, and proposes no
    action outside the action set. Other arms may be mentioned contrastively ("rather than ...").
    """
    t = text.lower()
    if not t.strip():
        return False, "empty"
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]
    if len(sentences) > 3:
        return False, "too long (more than three sentences)"
    for phrase in FORBIDDEN_ACTIONS:
        if phrase in t:
            return False, f"names an action outside the action set: '{phrase}'"
    if not re.search(ARM_PATTERNS[chosen_arm], t):
        return False, f"does not name the chosen action ({chosen_arm})"
    for arm, pat in ARM_PATTERNS.items():
        if arm == chosen_arm or arm == "no_action":
            continue  # "doing nothing" is the counterfactual and may always be mentioned
        for sent in sentences:
            sl = sent.lower()
            if re.search(pat, sl) and not any(m in sl for m in CONTRAST_MARKERS):
                return False, f"recommends another arm: {arm}"
    return True, "ok"


class Explainer(Protocol):
    def explain(self, row: dict[str, Any]) -> tuple[str, str]:
        """Return (text, source) where source is 'claude', 'template' or 'fake'."""


def _arm_name(row: dict[str, Any]) -> str:
    return ARMS[int(row["chosen_arm"])]


def _fmt_rs(x: float) -> str:
    return f"Rs {x:,.0f}"


def _label(row: dict[str, Any]) -> str:
    arm = _arm_name(row)
    if arm == "retry_delayed":
        return f"retry after {int(row.get('delay_days', 0))} day(s)"
    return ARM_LABELS[arm]


class TemplateExplainer:
    """Deterministic explanation built only from the audit row."""

    def explain(self, row: dict[str, Any]) -> tuple[str, str]:
        arm = _arm_name(row)
        ev = row["net_ev"]
        amount = float(row.get("amount") or 0)
        cat = str(row.get("failure_category", "")).replace("_", " ")
        codes = [c for c in str(row.get("rejection_codes") or "").split("|") if c]
        best_ev = float(ev[int(row["chosen_arm"])])
        label = _label(row)
        if codes and any(c.startswith("MANDATORY_ESCALATION") for c in codes):
            why = "a hard rule requires human review for this payment"
        elif arm == "no_action":
            why = "no intervention is expected to add value net of its cost"
        else:
            why = f"it has the highest expected net value ({_fmt_rs(best_ev)} on a {_fmt_rs(amount)} payment)"
        s1 = f"For this {cat} failure we chose to {label} because {why}."
        if codes:
            s2 = f"Guardrails applied: {', '.join(codes)}."
        else:
            s2 = f"Doing nothing would have recovered an estimated {float(row.get('p_no_action_hat') or 0):.0%} of the time."
        return f"{s1} {s2}", "template"


@dataclass
class FakeExplainer:
    """Test double: deterministic text, records every call, can be told to misbehave."""

    calls: list[str] = field(default_factory=list)
    canned: str | None = None

    def explain(self, row: dict[str, Any]) -> tuple[str, str]:
        self.calls.append(row["event_id"])
        if self.canned is not None:
            return self.canned, "fake"
        text, _ = TemplateExplainer().explain(row)
        return text, "fake"


SYSTEM_PROMPT = (
    "You write two-sentence, plain-English explanations of an automated payment-recovery decision "
    "for a merchant's finance team. You are given the decision and the numbers behind it. "
    "Sentence one states the action taken, using the exact action phrase provided, and the main "
    "reason. Sentence two states what it costs or risks, or why doing nothing was worse. "
    "The numbers labelled 'incremental recovery probability' are lifts over doing nothing, not "
    "success rates; never call them success probabilities and never invent other numbers. "
    "Do not recommend a different action, and do not mention discounts, refunds, cancellations, "
    "blocking or legal steps. Use rupees as 'Rs'. Exactly two sentences, no bullet points, no preamble."
)


class ClaudeExplainer:
    """Claude-backed explainer with a per-run cap, validation, and template fallback."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_per_run: int = MAX_EXPLANATIONS_PER_RUN,
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.max_per_run = max_per_run
        self.calls = 0
        self.rejected = 0
        self.errors = 0
        self._client = client
        if api_key is None:
            from counterfact.config import get_settings  # .env is parsed here, never exported

            api_key = os.environ.get("ANTHROPIC_API_KEY") or get_settings().anthropic_api_key
        self._api_key = api_key
        self.template = TemplateExplainer()

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key) if self._api_key else anthropic.Anthropic()
        return self._client

    def _prompt(self, row: dict[str, Any]) -> str:
        label = _label(row)
        ev = {ARMS[i]: round(float(v)) for i, v in enumerate(row["net_ev"])}
        up = {ARMS[i]: round(float(v), 3) for i, v in enumerate(row["uplift"])}
        codes = row.get("rejection_codes") or "none"
        lines = [
            f"Action taken: '{label}' (use this exact phrase in sentence one).",
            f"Failure: {row.get('failure_category')} on a Rs {float(row.get('amount') or 0):,.0f} "
            f"subscription payment for merchant {row.get('merchant_id')}, attempt {row.get('attempt_number')}.",
            f"Incremental recovery probability of each action over doing nothing: {up}.",
            f"Net expected value of each action after its cost, in Rs: {ev}.",
            f"Estimated chance the payment recovers with no action at all: {float(row.get('p_no_action_hat') or 0):.0%}.",
            f"Guardrail codes applied: {codes} (if a MANDATORY_ code is present, say a rule required it).",
            "Do not recommend any other action.",
        ]
        return "\n".join(lines)

    def explain(self, row: dict[str, Any]) -> tuple[str, str]:
        if self.calls >= self.max_per_run:
            return self.template.explain(row)
        self.calls += 1
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": self._prompt(row)}],
            )
        except Exception as e:  # noqa: BLE001 - any SDK / network error -> template
            self.errors += 1
            text, _ = self.template.explain(row)
            return text, f"template (llm error: {type(e).__name__})"
        if getattr(resp, "stop_reason", None) == "refusal":
            self.rejected += 1
            text, _ = self.template.explain(row)
            return text, "template (llm refusal)"
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        ok, reason = validate(text, _arm_name(row))
        if not ok:
            self.rejected += 1
            fallback, _ = self.template.explain(row)
            return fallback, f"template (llm output rejected: {reason})"
        return text, "claude"


def explain_pending(store: AuditStore, explainer: Explainer, limit: int = MAX_EXPLANATIONS_PER_RUN) -> int:
    """Generate and cache explanations for up to ``limit`` decisions that lack one."""
    rows = store.recent(limit=limit, unexplained_only=True)
    n = 0
    for row in rows:
        row.setdefault("p_no_action_hat", 0.0)
        text, source = explainer.explain(row)
        store.set_explanation(row["event_id"], text, source)
        n += 1
    return n


def reexplain(store: AuditStore, explainer: Explainer, event_ids: list[str]) -> int:
    """Regenerate explanations for specific rows (used after a prompt or validator change)."""
    n = 0
    for eid in event_ids:
        row = store.get(eid)
        if row is None:
            continue
        text, source = explainer.explain(row)
        store.set_explanation(eid, text, source)
        n += 1
    return n
