"""Force a provider 5xx on the next retry executed by any running agent (API or batch).

Usage::  python scripts/inject_failure.py [--count 3]

Writes ``data/inject_5xx.flag`` containing the number of consecutive provider calls that must
fail. The executor's ``FailureInjector`` consumes the flag before its next call: with
``--count 3`` (the executor's retry budget) the action backs off three times, is parked as
``queued`` with no charge, the batch continues, and ``redrive`` executes it later under the same
idempotency key.
"""

from __future__ import annotations

import argparse

from counterfact.config import ROOT

FLAG = ROOT / "data" / "inject_5xx.flag"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=3)
    args = ap.parse_args()
    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text(str(args.count))
    print(f"armed: next {args.count} provider call(s) will fail with 502 -> {FLAG}")


if __name__ == "__main__":
    main()
