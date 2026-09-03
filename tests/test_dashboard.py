"""The dashboard renders headless without exceptions when report tables exist."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"


@pytest.mark.skipif(not (TABLES / "ab_paired.csv").exists(), reason="run `make eval` first")
def test_dashboard_renders_without_exceptions() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Counterfact"
    assert len(at.metric) >= 6
    # switching variant re-renders cleanly
    if len(at.sidebar.selectbox) and "null_uplift" in at.sidebar.selectbox[0].options:
        at.sidebar.selectbox[0].set_value("null_uplift").run()
        assert not at.exception, [e.value for e in at.exception]
