"""Append-only audit trail (JSONL) with a SQLite view, plus the execution ledger.

Every decision is written once as a JSON line (the source of truth) and mirrored into SQLite so
the dashboard and the API can query it. The same database holds the **execution ledger**: an
idempotency key is *reserved* before any executor call and *finished* after it, so a key can be
executed at most once no matter how many times an event is replayed. Duplicate execution is
impossible by construction, not by convention.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    event_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    features_hash TEXT NOT NULL,
    merchant_id TEXT, amount REAL, failure_category TEXT, attempt_number INTEGER,
    chosen_arm INTEGER NOT NULL, action_name TEXT NOT NULL, delay_days INTEGER NOT NULL,
    proposed_arm INTEGER, overridden INTEGER,
    uplift TEXT NOT NULL, net_ev TEXT NOT NULL,
    guardrail_checks TEXT NOT NULL, rejection_codes TEXT, reason TEXT,
    executor_status TEXT, executor_result TEXT,
    outcome TEXT, explanation TEXT, explanation_source TEXT,
    p_no_action_hat REAL, threshold REAL
);
CREATE TABLE IF NOT EXISTS executions (
    idempotency_key TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    action_name TEXT NOT NULL,
    status TEXT NOT NULL,          -- pending | executed | queued | failed | skipped
    attempts INTEGER NOT NULL DEFAULT 0,
    result TEXT,
    ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_exec_status ON executions(status);
"""

FINAL_STATUSES = ("executed", "failed", "skipped")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class AuditStore:
    """JSONL + SQLite audit store and execution ledger. Thread-safe within a process."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.dir / "audit.jsonl"
        self.db_path = self.dir / "audit.db"
        self._lock = threading.RLock()
        self._con = sqlite3.connect(self.db_path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        with self._con:
            self._con.executescript(SCHEMA)
            for col in ("p_no_action_hat REAL", "threshold REAL"):  # migrate older stores
                try:
                    self._con.execute(f"ALTER TABLE decisions ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass

    # ---- decisions -------------------------------------------------------------------------------
    def append_decision(self, row: dict[str, Any]) -> None:
        """Append one decision (JSON line) and upsert the SQLite mirror."""
        row = {"ts": _now(), **row}
        with self._lock:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
            with self._con:
                self._con.execute(
                    """INSERT INTO decisions
                       (event_id, ts, idempotency_key, features_hash, merchant_id, amount, failure_category,
                        attempt_number, chosen_arm, action_name, delay_days, proposed_arm, overridden,
                        uplift, net_ev, guardrail_checks, rejection_codes, reason, executor_status,
                        executor_result, outcome, explanation, explanation_source,
                        p_no_action_hat, threshold)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(event_id) DO UPDATE SET
                         ts = excluded.ts, idempotency_key = excluded.idempotency_key,
                         features_hash = excluded.features_hash, chosen_arm = excluded.chosen_arm,
                         action_name = excluded.action_name, delay_days = excluded.delay_days,
                         proposed_arm = excluded.proposed_arm, overridden = excluded.overridden,
                         uplift = excluded.uplift, net_ev = excluded.net_ev,
                         guardrail_checks = excluded.guardrail_checks,
                         rejection_codes = excluded.rejection_codes, reason = excluded.reason,
                         executor_status = excluded.executor_status,
                         executor_result = excluded.executor_result,
                         outcome = COALESCE(excluded.outcome, decisions.outcome),
                         explanation = COALESCE(excluded.explanation, decisions.explanation),
                         explanation_source = COALESCE(excluded.explanation_source, decisions.explanation_source),
                         p_no_action_hat = excluded.p_no_action_hat, threshold = excluded.threshold""",
                    (
                        row["event_id"], row["ts"], row["idempotency_key"], row["features_hash"],
                        row.get("merchant_id"), row.get("amount"), row.get("failure_category"),
                        row.get("attempt_number"), int(row["chosen_arm"]), row["action_name"],
                        int(row.get("delay_days", 0)), row.get("proposed_arm"), int(bool(row.get("overridden"))),
                        json.dumps(row["uplift"]), json.dumps(row["net_ev"]),
                        json.dumps(row["guardrail_checks"]), row.get("rejection_codes", ""), row.get("reason"),
                        (row.get("executor_result") or {}).get("status"),
                        json.dumps(row.get("executor_result"), default=str),
                        json.dumps(row.get("outcome"), default=str) if row.get("outcome") is not None else None,
                        row.get("explanation"), row.get("explanation_source"),
                        row.get("p_no_action_hat"), row.get("threshold"),
                    ),
                )

    def set_outcome(self, event_id: str, outcome: dict[str, Any]) -> None:
        with self._lock, open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _now(), "event_id": event_id, "outcome": outcome}, default=str) + "\n")
        with self._lock, self._con:
            self._con.execute("UPDATE decisions SET outcome = ? WHERE event_id = ?",
                              (json.dumps(outcome, default=str), event_id))

    def set_executor_result(self, event_id: str, result: dict[str, Any]) -> None:
        with self._lock, open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _now(), "event_id": event_id, "executor_result": result}, default=str) + "\n")
        with self._lock, self._con:
            self._con.execute("UPDATE decisions SET executor_status = ?, executor_result = ? WHERE event_id = ?",
                              (result.get("status"), json.dumps(result, default=str), event_id))

    def set_explanation(self, event_id: str, text: str, source: str) -> None:
        with self._lock, open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _now(), "event_id": event_id, "explanation": text, "source": source}) + "\n")
        with self._lock, self._con:
            self._con.execute("UPDATE decisions SET explanation = ?, explanation_source = ? WHERE event_id = ?",
                              (text, source, event_id))

    @staticmethod
    def _row_to_dict(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        for k in ("uplift", "net_ev", "guardrail_checks", "executor_result", "outcome"):
            if d.get(k):
                d[k] = json.loads(d[k])
        d["overridden"] = bool(d.get("overridden"))
        return d

    def get(self, event_id: str) -> dict[str, Any] | None:
        r = self._con.execute("SELECT * FROM decisions WHERE event_id = ?", (event_id,)).fetchone()
        return self._row_to_dict(r) if r else None

    def find_by_key(self, idempotency_key: str) -> dict[str, Any] | None:
        r = self._con.execute("SELECT * FROM decisions WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
        return self._row_to_dict(r) if r else None

    def recent(self, limit: int = 100, unexplained_only: bool = False) -> list[dict[str, Any]]:
        q = "SELECT * FROM decisions" + (" WHERE explanation IS NULL" if unexplained_only else "")
        q += " ORDER BY ts DESC LIMIT ?"
        return [self._row_to_dict(r) for r in self._con.execute(q, (limit,)).fetchall()]

    def all_decisions(self) -> list[dict[str, Any]]:
        return [self._row_to_dict(r) for r in self._con.execute("SELECT * FROM decisions ORDER BY ts").fetchall()]

    def metrics(self) -> dict[str, Any]:
        """Batch-level summary: counts, rupees recovered (where outcome known), contacts, queue."""
        rows = self.all_decisions()
        n = len(rows)
        known = [r for r in rows if r.get("outcome")]
        recovered = [r for r in known if r["outcome"].get("recovered")]
        return {
            "decisions": n,
            "outcomes_known": len(known),
            "recovered": len(recovered),
            "recovery_rate": (len(recovered) / len(known)) if known else None,
            "rs_recovered": float(sum(r["outcome"].get("recovered_amount", 0.0) for r in known)),
            "rs_at_risk": float(sum(r.get("amount") or 0.0 for r in rows)),
            "contacts": sum(r["action_name"] == "remind_and_retry" for r in rows),
            "escalations": sum(r["action_name"] == "escalate_human" for r in rows),
            "abstentions": sum(r["action_name"] == "no_action" for r in rows),
            "overridden_by_guardrails": sum(bool(r.get("overridden")) for r in rows),
            "executor": self.ledger_counts(),
        }

    # ---- execution ledger ------------------------------------------------------------------------
    def reserve(self, key: str, event_id: str, action_name: str) -> bool:
        """Atomically claim ``key``. True if this caller now owns it; False if it already existed."""
        with self._lock, self._con:
            cur = self._con.execute(
                "INSERT OR IGNORE INTO executions (idempotency_key, event_id, action_name, status, attempts, ts) "
                "VALUES (?, ?, ?, 'pending', 0, ?)",
                (key, event_id, action_name, _now()),
            )
            return cur.rowcount == 1

    def claim_queued(self, key: str) -> bool:
        """Move a queued key back to pending for a re-drive. True if this caller now owns it."""
        with self._lock, self._con:
            cur = self._con.execute(
                "UPDATE executions SET status = 'pending', ts = ? WHERE idempotency_key = ? AND status = 'queued'",
                (_now(), key),
            )
            return cur.rowcount == 1

    def finish(self, key: str, status: str, attempts: int, result: dict[str, Any] | None) -> None:
        with self._lock, self._con:
            self._con.execute(
                "UPDATE executions SET status = ?, attempts = ?, result = ?, ts = ? WHERE idempotency_key = ?",
                (status, attempts, json.dumps(result, default=str), _now(), key),
            )

    def lookup(self, key: str) -> dict[str, Any] | None:
        r = self._con.execute("SELECT * FROM executions WHERE idempotency_key = ?", (key,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["result"] = json.loads(d["result"]) if d.get("result") else None
        return d

    def queued(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._con.execute("SELECT * FROM executions WHERE status = 'queued'").fetchall()]

    def ledger_counts(self) -> dict[str, int]:
        rows = self._con.execute("SELECT status, COUNT(*) AS c FROM executions GROUP BY status").fetchall()
        return {r["status"]: int(r["c"]) for r in rows}

    def close(self) -> None:
        self._con.close()
