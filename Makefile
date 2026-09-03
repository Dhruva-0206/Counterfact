# Counterfact task runner. Windows without GNU make: use `.\make.ps1 <target>` (same targets).
UV ?= uv
PY := $(UV) run python

.PHONY: setup data train eval demo dashboard test lint clean all

setup:            ## create venv + install pinned deps (~1-2 min on a clean machine)
	$(UV) sync --all-groups
	@echo "OK: run 'make test' to verify"

data:             ## generate 50k failures for all three simulator variants (~30s)
	$(PY) scripts/make_data.py --all-variants

train:            ## fit uplift models per variant (~1-2 min)
	$(PY) scripts/train.py --all-variants

eval:             ## A/B + OPE, tables and figures into reports/ (~2 min)
	$(PY) scripts/evaluate.py --all-variants

demo:             ## end-to-end batch of 500 with one injected 5xx (~20s)
	$(PY) scripts/run_batch.py --n 500 --inject-failure

dashboard:        ## Streamlit dashboard over audit + reports
	$(UV) run streamlit run dashboard/app.py

test:             ## pytest (guardrails, idempotency, no-leakage, OPE toy cases)
	$(UV) run pytest

lint:
	$(UV) run ruff check src tests scripts

all: setup data train eval demo

clean:
	rm -rf data/* reports/* .pytest_cache .ruff_cache
