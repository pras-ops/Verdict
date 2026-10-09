from fastapi.testclient import TestClient

from verdict.server import create_app


def client():
    c = TestClient(create_app(":memory:"))
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
