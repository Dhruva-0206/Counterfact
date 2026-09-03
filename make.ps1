# Windows mirror of the Makefile targets. Usage: .\make.ps1 <target>
param([Parameter(Position = 0)][string]$Target = "test")
$ErrorActionPreference = "Stop"
switch ($Target) {
    "setup"     { uv sync --all-groups }
    "data"      { uv run python scripts/make_data.py --all-variants }
    "train"     { uv run python scripts/train.py --all-variants }
    "eval"      { uv run python scripts/evaluate.py --all-variants }
    "sensitivity" { uv run python scripts/sensitivity.py }
    "dial"      { uv run python scripts/z_dial.py }
    "demo"      { uv run python scripts/run_batch.py --n 500 --inject-failure }
    "dashboard" { uv run streamlit run dashboard/app.py }
    "test"      { uv run pytest }
    "lint"      { uv run ruff check src tests scripts }
    "all"       { uv sync --all-groups; uv run python scripts/make_data.py --all-variants; uv run python scripts/train.py --all-variants; uv run python scripts/evaluate.py --all-variants; uv run python scripts/run_batch.py --n 500 --inject-failure }
    default     { Write-Error "Unknown target '$Target'. Targets: setup data train eval sensitivity dial demo dashboard test lint all" }
}
