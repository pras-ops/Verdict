import pytest
from fastapi.testclient import TestClient

from verdict.server import create_app


def client():
    c = TestClient(create_app(":memory:", allowed_hosts=["testserver"]))
    r = c.post("/api/models", json={"name": "m1", "backend": "mock", "config": {"model": "mock", "latency_ms": 0}})
    assert r.status_code == 200, r.text
    c.post("/api/models", json={"name": "m2", "backend": "mock", "config": {"model": "mock", "latency_ms": 0, "sharpness": 1}})
    return c


def test_decide_log_feedback_stats_metrics():
    c = client()
    r = c.post("/api/decide", json={"model": "m1", "question": "urgent invoice", "options": ["urgent invoice", "newsletter"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choice"] == "urgent invoice" and body["id"]

    assert c.post(f"/api/decisions/{body['id']}/feedback", json={"label": "urgent invoice"}).status_code == 200
    stats = c.get("/api/stats?range=600").json()
    m1 = next(m for m in stats["models"] if m["model"] == "m1")
    assert m1["n"] == 1 and m1["accuracy"] == 1.0 and m1["p50_ms"] is not None

    text = c.get("/metrics").text
    assert 'verdict_decisions_total{model="m1"} 1' in text
    assert 'verdict_correct_total{model="m1"} 1' in text
    assert 'verdict_latency_ms_bucket{model="m1",le="+Inf"} 1' in text


def test_compare_and_unknown_model():
    c = client()
    r = c.post("/api/compare", json={"models": ["m1", "m2"], "kind": "binary", "question": "ok?"})
    assert [x["model"] for x in r.json()] == ["m1", "m2"]
    assert c.post("/api/decide", json={"model": "nope", "question": "q", "kind": "binary"}).status_code == 404


def test_evaluate_and_set_temperature():
    c = client()
    ex = [{"question": "red apple", "options": ["red apple", "blue sky"], "answer": "red apple"}] * 3
    out = c.post("/api/evaluate", json={"models": ["m1"], "examples": ex}).json()[0]
    assert out["raw"]["accuracy"] == 1.0
    assert c.put("/api/models/m1/temperature", json={"temperature": 2.0}).json()["temperature"] == 2.0


def test_backend_error_is_logged():
    c = client()
    c.post("/api/models", json={"name": "dead", "backend": "ollama", "config": {"model": "x", "host": "http://127.0.0.1:9", "timeout": 1}})
    r = c.post("/api/decide", json={"model": "dead", "kind": "binary", "question": "q"})
    assert r.status_code == 502
    stats = c.get("/api/stats").json()
    assert next(m for m in stats["models"] if m["model"] == "dead")["errors"] == 1


def test_dashboard_served():
    c = client()
    assert c.get("/").status_code == 200
    assert set(c.get("/api/backends").json()) >= {"ollama", "openai", "huggingface", "mock"}


# -- security (0.2) --------------------------------------------------------------
def test_only_localhost_host_headers_by_default():
    c = TestClient(create_app(":memory:"), base_url="http://localhost")
    assert c.get("/api/models").status_code == 200
    # a page on another domain that DNS-rebinds to 127.0.0.1 still sends its own Host header
    assert c.get("/api/models", headers={"Host": "evil.example"}).status_code == 400


def test_token_is_required_when_set():
    c = TestClient(create_app(":memory:", token="s3cret", allowed_hosts=["testserver"]))
    assert c.get("/api/models").status_code == 401
    assert c.get("/metrics").status_code == 401
    assert c.post("/v1/systemone", json={}).status_code == 401
    assert c.get("/api/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/api/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/api/models", headers={"X-Verdict-Token": "s3cret"}).status_code == 200
    assert c.get("/api/health").status_code == 200  # for load-balancer health checks
    assert c.get("/").status_code == 200  # the page itself holds no data


def test_secrets_are_masked_and_kept():
    c = client()
    r = c.post("/api/models", json={"name": "o", "backend": "openai", "config": {"model": "gpt", "api_key": "sk-live-123"}})
    assert r.json()["config"]["api_key"] == "***"
    assert next(m for m in c.get("/api/models").json() if m["name"] == "o")["config"]["api_key"] == "***"
    # sending the mask back (e.g. re-saving the form) keeps the stored key
    c.post("/api/models", json={"name": "o", "backend": "openai", "config": {"model": "gpt2", "api_key": "***"}})
    assert c.app.state.store.model("o")["config"]["api_key"] == "sk-live-123"


def test_api_key_env_must_look_like_an_api_key_variable():
    c = client()
    bad = c.post("/api/models", json={"name": "x", "backend": "openai", "config": {"model": "m", "api_key_env": "AWS_SECRET_ACCESS_KEY"}})
    assert bad.status_code == 400
    ok = c.post("/api/models", json={"name": "x", "backend": "openai", "config": {"model": "m", "api_key_env": "GROQ_API_KEY"}})
    assert ok.status_code == 200


def test_temperature_is_validated():
    c = client()
    assert c.put("/api/models/m1/temperature", json={"temperature": 0}).status_code == 422
    assert c.put("/api/models/m1/temperature", json={"temperature": "hot"}).status_code == 422
    assert c.post("/api/models", json={"name": "t", "backend": "mock", "config": {"model": "m"}, "temperature": -1}).status_code == 422


# -- Jev-compatible endpoint -------------------------------------------------------
def test_systemone_answers_every_question_and_logs_them():
    c = client()
    body = {
        "model": "jev-latest",  # not registered: falls back to the first model
        "state": "urgent invoice from accounting",
        "questions": {
            "folder": {"type": "choice", "instructions": "Folder?", "criteria": {"invoices": "urgent invoice", "newsletters": "newsletter"}},
            "urgent": {"type": "noul", "instructions": "Urgent?"},
            "tone": {"type": "score", "instructions": "How formal?", "criteria": ["casual", "neutral", "formal"]},
        },
    }
    r = c.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["model"] == "m1" and set(out["answers"]) == {"folder", "urgent", "tone"}
    assert out["answers"]["folder"]["choice"] == "invoices"
    assert 0 <= out["answers"]["urgent"]["noul"] <= 1
    assert sum(out["answers"]["tone"]["probabilities"].values()) == pytest.approx(1, abs=1e-4)
    assert sum(1 for d in c.get("/api/decisions").json() if d["source"] == "systemone") == 3

    body["model"] = "m2"  # a registered name is used as-is
    assert c.post("/v1/systemone", json=body).json()["model"] == "m2"


def test_systemone_validation_and_backend_errors():
    c = client()
    assert c.post("/v1/systemone", json={"state": "x", "questions": {}}).status_code == 422
    c.post("/api/models", json={"name": "dead", "backend": "ollama", "config": {"model": "x", "host": "http://127.0.0.1:9", "timeout": 1}})
    r = c.post("/v1/systemone", json={"model": "dead", "state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}})
    assert r.status_code == 502
    empty = TestClient(create_app(":memory:", allowed_hosts=["testserver"]))
    assert empty.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}}).status_code == 404


def test_app_survives_a_second_startup():
    app = create_app(":memory:", allowed_hosts=["testserver"])
    with TestClient(app) as c:  # runs startup + shutdown
        c.post("/api/models", json={"name": "m", "backend": "mock", "config": {"model": "mock", "latency_ms": 0}})
    with TestClient(app) as c:
        r = c.post("/api/compare", json={"models": ["m"], "kind": "binary", "question": "ok?"})
        assert r.status_code == 200 and "error" not in r.json()[0]
