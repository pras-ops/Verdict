"""SQLite storage for registered models and the decision log."""
from __future__ import annotations

import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    name TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    config TEXT NOT NULL,
    temperature REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    model TEXT NOT NULL,
    kind TEXT,
    question TEXT,
    context TEXT,
    options TEXT,
    probs TEXT,
    choice TEXT,
    confidence REAL,
    coverage REAL,
    latency_ms REAL,
    method TEXT,
    error TEXT,
    label TEXT,
    source TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts);
CREATE INDEX IF NOT EXISTS idx_decisions_model_ts ON decisions(model, ts);
"""


class Store:
    def __init__(self, path: str = "verdict.db"):
        self.path = path
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def _exec(self, sql: str, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur

    def _all(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    # -- models -------------------------------------------------------------
    def upsert_model(self, name: str, backend: str, config: dict, temperature: float = 1.0) -> None:
        self._exec(
            "INSERT INTO models(name, backend, config, temperature, created_at) VALUES (?,?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET backend=excluded.backend, config=excluded.config, temperature=excluded.temperature",
            (name, backend, json.dumps(config), temperature, time.time()),
        )

    def set_temperature(self, name: str, temperature: float) -> None:
        self._exec("UPDATE models SET temperature=? WHERE name=?", (temperature, name))

    def delete_model(self, name: str) -> None:
        self._exec("DELETE FROM models WHERE name=?", (name,))

    def models(self) -> list[dict]:
        rows = self._all("SELECT * FROM models ORDER BY created_at")
        for r in rows:
            r["config"] = json.loads(r["config"])
        return rows

    def model(self, name: str) -> dict | None:
        rows = [m for m in self.models() if m["name"] == name]
        return rows[0] if rows else None

    # -- decisions ------------------------------------------------------------
    def log(self, *, model: str, decision: dict, result: dict | None, error: str | None, source: str) -> int:
        r = result or {}
        cur = self._exec(
            "INSERT INTO decisions(ts, model, kind, question, context, options, probs, choice, confidence, coverage, "
            "latency_ms, method, error, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                time.time(),
                model,
                decision.get("kind"),
                decision.get("question"),
                (decision.get("context") or "")[:4000],
                json.dumps(decision.get("options") or []),
                json.dumps(r.get("probs")) if r else None,
                r.get("choice"),
                r.get("confidence"),
                r.get("coverage"),
                r.get("latency_ms"),
                r.get("method"),
                error,
                source,
            ),
        )
        return cur.lastrowid

    def set_label(self, decision_id: int, label: str | None) -> bool:
        return self._exec("UPDATE decisions SET label=? WHERE id=?", (label, decision_id)).rowcount > 0

    def decisions(self, model: str | None = None, limit: int = 100, since: float = 0) -> list[dict]:
        sql, args = "SELECT * FROM decisions WHERE ts >= ?", [since]
        if model:
            sql += " AND model = ?"
            args.append(model)
        rows = self._all(sql + " ORDER BY id DESC LIMIT ?", (*args, limit))
        for r in rows:
            r["options"] = json.loads(r["options"] or "[]")
            r["probs"] = json.loads(r["probs"]) if r["probs"] else None
        return rows

    def stats(self, since: float, buckets: int = 60) -> dict:
        now = time.time()
        width = max((now - since) / buckets, 1.0)
        per_model = self._all(
            "SELECT model, COUNT(*) AS n, SUM(error IS NOT NULL) AS errors, AVG(confidence) AS avg_confidence, "
            "AVG(coverage) AS avg_coverage, SUM(label IS NOT NULL AND error IS NULL) AS labelled, "
            "SUM(label IS NOT NULL AND label = choice) AS correct, SUM(method = 'constrained') AS constrained "
            "FROM decisions WHERE ts >= ? GROUP BY model ORDER BY n DESC",
            (since,),
        )
        lat = self._all(
            "SELECT model, latency_ms FROM decisions WHERE ts >= ? AND error IS NULL AND latency_ms IS NOT NULL",
            (since,),
        )
        by_model: dict[str, list[float]] = {}
        for r in lat:
            by_model.setdefault(r["model"], []).append(r["latency_ms"])
        for m in per_model:
            xs = sorted(by_model.get(m["model"], []))
            m["p50_ms"] = _pct(xs, 0.5)
            m["p95_ms"] = _pct(xs, 0.95)
            m["accuracy"] = (m["correct"] / m["labelled"]) if m["labelled"] else None

        series_rows = self._all(
            "SELECT model, CAST((ts - ?) / ? AS INTEGER) AS b, COUNT(*) AS n, AVG(latency_ms) AS lat, "
            "AVG(confidence) AS conf, SUM(error IS NOT NULL) AS errors "
            "FROM decisions WHERE ts >= ? GROUP BY model, b ORDER BY b",
            (since, width, since),
        )
        series: dict[str, list[dict]] = {}
        for r in series_rows:
            series.setdefault(r["model"], []).append(
                {"t": since + (r["b"] + 0.5) * width, "count": r["n"], "latency_ms": r["lat"], "confidence": r["conf"], "errors": r["errors"]}
            )
        return {"since": since, "until": now, "bucket_seconds": width, "models": per_model, "series": series}

    def metric_totals(self, buckets: list[float]) -> list[dict]:
        """Cumulative per-model counters for the Prometheus endpoint."""
        cols = ", ".join(f"SUM(error IS NULL AND latency_ms <= {b}) AS le_{i}" for i, b in enumerate(buckets))
        return self._all(
            f"SELECT model, COUNT(*) AS n, SUM(error IS NOT NULL) AS errors, SUM(method = 'constrained') AS constrained, "
            f"SUM(CASE WHEN error IS NULL THEN latency_ms ELSE 0 END) AS latency_sum, SUM(error IS NULL) AS ok, "
            f"SUM(label IS NOT NULL AND error IS NULL) AS labelled, SUM(label IS NOT NULL AND label = choice) AS correct, {cols} "
            f"FROM decisions GROUP BY model"
        )


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2)
