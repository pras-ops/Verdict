"""REST API + dashboard. Run with `verdict serve`."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .backends import BACKENDS, make_backend
from .calibration import evaluate
from .engine import Verdict
from .store import Store
from .types import Decision

DASHBOARD = Path(__file__).parent / "dashboard"
LATENCY_BUCKETS = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]


class ModelIn(BaseModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[\w.:/@+-]+$")
    backend: str
    config: dict[str, Any] = Field(default_factory=dict)
    temperature: float = 1.0


class DecideIn(Decision):
    model: str
    log: bool = True


class CompareIn(Decision):
    models: list[str]
    log: bool = True


class EvaluateIn(BaseModel):
    models: list[str]
    examples: list[dict] = Field(min_length=1, max_length=5000)
    log: bool = False


class FeedbackIn(BaseModel):
    label: str | None


def create_app(db_path: str = "verdict.db") -> FastAPI:
    store = Store(db_path)
    engines: dict[str, Verdict] = {}
    engines_lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=8)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        for name in list(engines):
            drop_engine(name)
        pool.shutdown(wait=False)

    app = FastAPI(title="Verdict", version="0.1.0", lifespan=lifespan)
    app.state.store = store

    def engine(name: str) -> Verdict:
        with engines_lock:
            if name in engines:
                return engines[name]
            m = store.model(name)
            if not m:
                raise HTTPException(404, f"model '{name}' is not registered")
            v = Verdict(make_backend(m["backend"], **m["config"]), name=name, temperature=m["temperature"])
            engines[name] = v
            return v

    def drop_engine(name: str) -> None:
        with engines_lock:
            v = engines.pop(name, None)
        if v:
            v.close()

    def run(name: str, d: Decision, log: bool, source: str) -> dict:
        payload = d.model_dump(include={"kind", "question", "context", "options"})
        try:
            r = engine(name).decide(d)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            if log:
                store.log(model=name, decision=payload, result=None, error=err, source=source)
            return {"model": name, "error": err}
        out = r.model_dump()
        if log:
            out["id"] = store.log(model=name, decision=payload, result=out, error=None, source=source)
        return out

    # -- models ---------------------------------------------------------------
    @app.get("/api/backends")
    def backends():
        return {
            k: {n: {"type": t, "default": dflt, "help": h} for n, (t, dflt, h) in cls.config_fields.items()}
            for k, cls in BACKENDS.items()
        }

    @app.get("/api/models")
    def list_models():
        return store.models()

    @app.post("/api/models")
    def add_model(m: ModelIn):
        if m.backend not in BACKENDS:
            raise HTTPException(400, f"unknown backend '{m.backend}'")
        if "model" not in m.config:
            raise HTTPException(400, "config.model is required")
        drop_engine(m.name)
        store.upsert_model(m.name, m.backend, m.config, m.temperature)
        return store.model(m.name)

    @app.delete("/api/models/{name:path}")
    def delete_model(name: str):
        drop_engine(name)
        store.delete_model(name)
        return {"ok": True}

    @app.post("/api/models/{name:path}/test")
    def test_model(name: str):
        v = engine(name)
        health = v.backend.health()
        if not health.get("ok"):
            return health
        probe = run(name, Decision(kind="binary", question="Is the sky usually blue on a clear day?"), False, "test")
        return {**health, "probe": probe}

    @app.put("/api/models/{name:path}/temperature")
    def set_temperature(name: str, body: dict):
        t = float(body.get("temperature", 1.0))
        if not 0.05 <= t <= 50:
            raise HTTPException(400, "temperature must be between 0.05 and 50")
        engine(name).temperature = t
        store.set_temperature(name, t)
        return store.model(name)

    # -- decisions ------------------------------------------------------------
    @app.post("/api/decide")
    def decide(body: DecideIn):
        d = Decision(**body.model_dump(exclude={"model", "log"}))
        out = run(body.model, d, body.log, "api")
        if "error" in out:
            raise HTTPException(502, out["error"])
        return out

    @app.post("/api/compare")
    def compare(body: CompareIn):
        d = Decision(**body.model_dump(exclude={"models", "log"}))
        for n in body.models:
            engine(n)  # 404 early for unknown names
        futures = [pool.submit(run, n, d, body.log, "playground") for n in body.models]
        return [f.result() for f in futures]

    @app.get("/api/decisions")
    def decisions(model: str | None = None, limit: int = 100, since: float = 0):
        return store.decisions(model, min(limit, 1000), since)

    @app.post("/api/decisions/{decision_id}/feedback")
    def feedback(decision_id: int, body: FeedbackIn):
        if not store.set_label(decision_id, body.label):
            raise HTTPException(404, "no such decision")
        return {"ok": True}

    @app.get("/api/stats")
    def stats(range: int = 3600, buckets: int = 60):
        return store.stats(time.time() - max(60, range), max(5, min(buckets, 300)))

    @app.post("/api/evaluate")
    def run_eval(body: EvaluateIn):
        def one(name: str):
            v = engine(name)
            cb = None
            if body.log:
                def cb(d, r, answer):
                    i = store.log(model=name, decision=d.model_dump(), result=r.model_dump(), error=None, source="eval")
                    store.set_label(i, answer)
            return evaluate(v, body.examples, cb)

        futures = [pool.submit(one, n) for n in body.models]
        return [f.result() for f in futures]

    # -- Prometheus -------------------------------------------------------------
    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        rows = store.metric_totals(LATENCY_BUCKETS)
        lines = [
            "# HELP verdict_decisions_total Decisions made, by model.",
            "# TYPE verdict_decisions_total counter",
        ]
        for r in rows:
            lines.append(f'verdict_decisions_total{{model="{_esc(r["model"])}"}} {r["n"]}')
        for metric, col, help_ in [
            ("verdict_errors_total", "errors", "Failed decisions."),
            ("verdict_constrained_total", "constrained", "Decisions answered via constrained fallback (no probabilities)."),
            ("verdict_labelled_total", "labelled", "Decisions with ground-truth feedback."),
            ("verdict_correct_total", "correct", "Labelled decisions the model got right."),
        ]:
            lines += [f"# HELP {metric} {help_}", f"# TYPE {metric} counter"]
            lines += [f'{metric}{{model="{_esc(r["model"])}"}} {r[col] or 0}' for r in rows]
        lines += ["# HELP verdict_latency_ms Decision latency.", "# TYPE verdict_latency_ms histogram"]
        for r in rows:
            m = _esc(r["model"])
            for i, b in enumerate(LATENCY_BUCKETS):
                lines.append(f'verdict_latency_ms_bucket{{model="{m}",le="{b}"}} {r[f"le_{i}"] or 0}')
            lines.append(f'verdict_latency_ms_bucket{{model="{m}",le="+Inf"}} {r["ok"] or 0}')
            lines.append(f'verdict_latency_ms_sum{{model="{m}"}} {round(r["latency_sum"] or 0, 3)}')
            lines.append(f'verdict_latency_ms_count{{model="{m}"}} {r["ok"] or 0}')
        return "\n".join(lines) + "\n"

    # -- dashboard ----------------------------------------------------------------
    @app.get("/")
    def index():
        return FileResponse(DASHBOARD / "index.html")

    app.mount("/static", StaticFiles(directory=DASHBOARD), name="static")

    return app


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
