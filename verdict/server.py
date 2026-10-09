"""REST API + dashboard. Run with `verdict serve`."""
from __future__ import annotations

import hmac
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .backends import BACKENDS, BackendError, make_backend
from .calibration import evaluate
from .engine import Verdict
from .jev import SystemOneRequest, answer
from .store import Store
from .types import Decision, Result

DASHBOARD = Path(__file__).parent / "dashboard"
LATENCY_BUCKETS = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
LOOPBACK_HOSTS = ["localhost", "127.0.0.1", "::1"]
# Config keys whose values never leave the server.
SECRET_KEY = re.compile(r"(api_?key|token|secret|password)$", re.IGNORECASE)
# api_key_env may only name variables that are meant to hold an API key, so a request
# can't make the server send, say, AWS_SECRET_ACCESS_KEY to a URL of its choosing.
KEY_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*_API_KEY|VERDICT_[A-Z0-9_]+")
MASK = "***"


class ModelIn(BaseModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[\w.:/@+-]+$")
    backend: str
    config: dict[str, Any] = Field(default_factory=dict)
    temperature: float = Field(1.0, ge=0.05, le=50)


class TemperatureIn(BaseModel):
    temperature: float = Field(ge=0.05, le=50)


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


def _public(model: dict | None) -> dict | None:
    """A registered model as the API shows it: secrets masked."""
    if model is None:
        return None
    cfg = {k: (MASK if SECRET_KEY.search(k) and v else v) for k, v in model["config"].items()}
    return {**model, "config": cfg}


def create_app(
    db_path: str = "verdict.db",
    token: str | None = None,
    allowed_hosts: list[str] | None = None,
    default_model: str | None = None,
) -> FastAPI:
    """Build the API + dashboard app.

    token          when set, every /api/*, /v1/* and /metrics request must send
                   `Authorization: Bearer <token>` (or `X-Verdict-Token: <token>`)
    allowed_hosts  Host header values to accept; defaults to localhost only, which
                   blocks DNS-rebinding attacks from web pages. ["*"] accepts any.
    default_model  registered model that answers /v1/systemone requests whose `model`
                   isn't registered (e.g. "jev-latest"); defaults to the first one added
    """
    store = Store(db_path)
    engines: dict[str, Verdict] = {}
    engines_lock = threading.Lock()
    # Created on first use and dropped at shutdown, so the app survives being started again
    # (test clients, reloading servers) instead of failing with "cannot schedule new futures".
    pool_box: list[ThreadPoolExecutor | None] = [None]

    def pool() -> ThreadPoolExecutor:
        with engines_lock:
            if pool_box[0] is None:
                pool_box[0] = ThreadPoolExecutor(max_workers=8)
            return pool_box[0]

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        for name in list(engines):
            drop_engine(name)
        with engines_lock:
            if pool_box[0] is not None:
                pool_box[0].shutdown(wait=False)
                pool_box[0] = None

    app = FastAPI(title="Verdict", version=__version__, lifespan=lifespan)
    app.state.store = store
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or LOOPBACK_HOSTS)

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        path = request.url.path
        if token and (path.startswith(("/api/", "/v1/")) or path == "/metrics") and path != "/api/health":
            sent = request.headers.get("x-verdict-token") or ""
            auth = request.headers.get("authorization") or ""
            if auth.lower().startswith("bearer "):
                sent = auth[7:].strip()
            if not hmac.compare_digest(sent.encode(), token.encode()):
                return JSONResponse({"detail": "missing or wrong token"}, status_code=401)
        return await call_next(request)

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

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__}

    @app.get("/api/models")
    def list_models():
        return [_public(m) for m in store.models()]

    @app.post("/api/models")
    def add_model(m: ModelIn):
        if m.backend not in BACKENDS:
            raise HTTPException(400, f"unknown backend '{m.backend}'")
        if "model" not in m.config:
            raise HTTPException(400, "config.model is required")
        env = m.config.get("api_key_env")
        if env and not KEY_ENV_NAME.fullmatch(str(env)):
            raise HTTPException(400, "api_key_env must name a variable ending in _API_KEY (or starting with VERDICT_)")
        config = dict(m.config)
        old = store.model(m.name)
        for k, v in config.items():  # a masked value sent back unchanged keeps the stored secret
            if v == MASK and old and SECRET_KEY.search(k):
                config[k] = old["config"].get(k)
        drop_engine(m.name)
        store.upsert_model(m.name, m.backend, config, m.temperature)
        return _public(store.model(m.name))

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
    def set_temperature(name: str, body: TemperatureIn):
        engine(name).temperature = body.temperature
        store.set_temperature(name, body.temperature)
        return _public(store.model(name))

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
        futures = [pool().submit(run, n, d, body.log, "playground") for n in body.models]
        return [f.result() for f in futures]

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest):
        """Jev-compatible: one state, several named questions (see verdict/jev.py)."""
        name = req.model if store.model(req.model) else (
            default_model or os.environ.get("VERDICT_DEFAULT_MODEL") or next((m["name"] for m in store.models()), None)
        )
        if not name:
            raise HTTPException(404, "no model registered; add one first or start with --default-model")
        engine(name)  # 404 early for an unknown default

        def decide(d: Decision) -> Result:
            out = run(name, d, True, "systemone")
            if "error" in out:
                raise BackendError(out["error"])
            return Result.model_validate(out)

        try:
            return answer(req, decide, name, pool())
        except BackendError as e:
            raise HTTPException(502, str(e)) from None

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

        futures = [pool().submit(one, n) for n in body.models]
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
