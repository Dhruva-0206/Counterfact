"""Training code is forbidden from reading the counterfactual table or the true outcome model.

Two layers:
1. Static: no module under ``features/`` or ``models/`` mentions ``counterfactuals`` or imports
   ``counterfact.sim.outcome_model``.
2. Dynamic (added with Phase 2): ``build_features`` / ``train`` run with ``pandas.read_parquet``
   patched to raise on any path containing ``counterfactuals``.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "counterfact"
FORBIDDEN_PATTERNS = (
    re.compile(r"counterfactuals"),
    re.compile(r"counterfact\.sim\.outcome_model"),
    re.compile(r"from counterfact\.sim import .*outcome_model"),
    re.compile(r"\by_no_action\b|\bp_no_action\b|\bu_att\d\b|\bz_liquidity\b|\bz_engagement\b"),
)


def _training_modules() -> list[Path]:
    return sorted((SRC / "features").glob("*.py")) + sorted((SRC / "models").glob("*.py"))


def test_training_code_never_references_counterfactuals() -> None:
    offenders = []
    for path in _training_modules():
        text = path.read_text(encoding="utf-8")
        for pat in FORBIDDEN_PATTERNS:
            if pat.search(text):
                offenders.append((path.name, pat.pattern))
    assert not offenders, offenders


def test_outcome_model_not_imported_by_policy_or_agent() -> None:
    """The online path (policy, agent, api) must not depend on the true process either."""
    offenders = []
    for sub in ("policy", "agent", "api"):
        for path in (SRC / sub).glob("*.py"):
            if "outcome_model" in path.read_text(encoding="utf-8"):
                offenders.append(str(path))
    assert not offenders, offenders
