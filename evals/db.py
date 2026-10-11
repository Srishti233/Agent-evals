"""SQLite persistence for runs, per-question results and the improve-loop changelog."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_version INTEGER NOT NULL,
    started_at TEXT NOT NULL, finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',          -- running | complete | failed
    n_cases INTEGER, accuracy REAL, hallucination_rate REAL, quality REAL, composite REAL,
    n_errors INTEGER DEFAULT 0, eval_hash TEXT, model TEXT, judge_model TEXT,
    agent_config_json TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    case_id TEXT, category TEXT, question TEXT, gold_answer TEXT, answer TEXT,
    abstained INTEGER, contexts_json TEXT, tool_calls_json TEXT, latency_s REAL, steps INTEGER,
    fuzzy REAL, emb_sim REAL, retrieval_recall REAL,
    verdict TEXT, verdict_source TEXT, verdict_reason TEXT, correct INTEGER,
    n_claims INTEGER, n_unsupported INTEGER, unsupported_json TEXT, hallucinated INTEGER,
    relevance INTEGER, completeness INTEGER, clarity INTEGER, quality REAL,
    failed INTEGER, failure_reasons TEXT, error TEXT, agent_version INTEGER, created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
CREATE TABLE IF NOT EXISTS changelog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT, iteration INTEGER, base_version INTEGER, candidate_version INTEGER,
    accepted INTEGER, base_composite REAL, candidate_composite REAL, delta REAL,
    changes TEXT, reason TEXT
);
"""

JSON_COLS = ("contexts_json", "tool_calls_json", "unsupported_json")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")  # dashboard can read while an eval writes
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_run(path: Path, *, agent_version: int, eval_hash: str, model: str, judge_model: str,
               agent_config: dict[str, Any]) -> int:
    with connect(path) as c:
        cur = c.execute(
            "INSERT INTO runs(agent_version, started_at, status, eval_hash, model, judge_model, agent_config_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (agent_version, now(), "running", eval_hash, model, judge_model, json.dumps(agent_config)))
        return int(cur.lastrowid)


def save_result(path: Path, run_id: int, row: dict[str, Any]) -> None:
    data = {k: v for k, v in row.items() if k not in ("contexts", "tool_calls", "unsupported", "claims", "verdicts", "judge_failed")}
    data["contexts_json"] = json.dumps(row.get("contexts", []), ensure_ascii=False)
    data["tool_calls_json"] = json.dumps(row.get("tool_calls", []), ensure_ascii=False)
    data["unsupported_json"] = json.dumps(row.get("unsupported", []), ensure_ascii=False)
    data["run_id"], data["created_at"] = run_id, now()
    for k, v in list(data.items()):
        if isinstance(v, bool):
            data[k] = int(v)
    cols = ", ".join(data)
    marks = ", ".join("?" for _ in data)
    with connect(path) as c:
        c.execute(f"INSERT INTO results({cols}) VALUES ({marks})", list(data.values()))


def finish_run(path: Path, run_id: int, metrics: dict[str, Any], *, status: str = "complete", notes: str = "") -> None:
    with connect(path) as c:
        c.execute(
            "UPDATE runs SET finished_at=?, status=?, n_cases=?, accuracy=?, hallucination_rate=?, quality=?, "
            "composite=?, n_errors=?, notes=? WHERE run_id=?",
            (now(), status, metrics.get("n"), metrics.get("accuracy"), metrics.get("hallucination_rate"),
             metrics.get("quality"), metrics.get("composite"), metrics.get("n_errors", 0), notes, run_id))


def get_run(path: Path, run_id: int) -> dict[str, Any] | None:
    with connect(path) as c:
        row = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return dict(row) if row else None


def list_runs(path: Path, *, complete_only: bool = True) -> list[dict[str, Any]]:
    q = "SELECT * FROM runs " + ("WHERE status='complete' " if complete_only else "") + "ORDER BY run_id"
    with connect(path) as c:
        return [dict(r) for r in c.execute(q).fetchall()]


def latest_run(path: Path, *, version: int | None = None, eval_hash: str | None = None,
               full_only: bool = True) -> dict[str, Any] | None:
    q, args = "SELECT * FROM runs WHERE status='complete'", []
    if version is not None:
        q += " AND agent_version=?"
        args.append(version)
    if eval_hash is not None:
        q += " AND eval_hash=?"
        args.append(eval_hash)
    if full_only:
        q += " AND (notes IS NULL OR notes NOT LIKE '%partial%')"
    q += " ORDER BY run_id DESC LIMIT 1"
    with connect(path) as c:
        row = c.execute(q, args).fetchone()
    return dict(row) if row else None


def get_results(path: Path, run_id: int) -> list[dict[str, Any]]:
    with connect(path) as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM results WHERE run_id=? ORDER BY id", (run_id,))]
    for r in rows:
        r["contexts"] = json.loads(r.pop("contexts_json") or "[]")
        r["tool_calls"] = json.loads(r.pop("tool_calls_json") or "[]")
        r["unsupported"] = json.loads(r.pop("unsupported_json") or "[]")
    return rows


def add_changelog(path: Path, entry: dict[str, Any]) -> None:
    with connect(path) as c:
        c.execute(
            "INSERT INTO changelog(created_at, iteration, base_version, candidate_version, accepted, "
            "base_composite, candidate_composite, delta, changes, reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (now(), entry["iteration"], entry["base_version"], entry["candidate_version"], int(entry["accepted"]),
             entry["base_composite"], entry["candidate_composite"], entry["delta"],
             entry["changes"], entry["reason"]))


def list_changelog(path: Path) -> list[dict[str, Any]]:
    with connect(path) as c:
        return [dict(r) for r in c.execute("SELECT * FROM changelog ORDER BY id")]
