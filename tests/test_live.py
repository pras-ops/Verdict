"""End-to-end checks against real models. Skipped unless you opt in:

    VERDICT_LIVE_MODEL=qwen2.5:7b pytest -m live          # needs `ollama serve` and the model pulled
    VERDICT_HF_MODEL=Qwen/Qwen2.5-0.5B-Instruct pytest -m hf   # needs pip install ".[hf]"
"""
import json
import os
import shutil
import sys

import anyio
import pytest
from fastapi.testclient import TestClient

from verdict import Verdict
from verdict.prompting import build_prompt
from verdict.types import Decision

LIVE_MODEL = os.environ.get("VERDICT_LIVE_MODEL")
OLLAMA = os.environ.get("VERDICT_OLLAMA_HOST", "http://127.0.0.1:11434")
HF_MODEL = os.environ.get("VERDICT_HF_MODEL")

live = pytest.mark.skipif(not LIVE_MODEL, reason="set VERDICT_LIVE_MODEL to run against a local Ollama model")

SPAM = "CONGRATULATIONS!!! You won a FREE iPhone. Click this link within 24 hours and enter your bank details."
OUTAGE = "The production database is down and no customer can complete a payment."
GLOWING = "Absolutely loved it. Best purchase I've made all year!"


@pytest.fixture(scope="module")
def ollama():
    v = Verdict.from_backend("ollama", model=LIVE_MODEL, host=OLLAMA)
    health = v.backend.health()
    if not health["ok"]:
        pytest.skip(f"Ollama not ready: {health.get('error')}")
    yield v
    v.close()


@pytest.fixture(scope="module")
def openai_compat():
    v = Verdict.from_backend("openai", model=LIVE_MODEL, base_url=f"{OLLAMA}/v1")
    yield v
    v.close()


@live
@pytest.mark.live
def test_ollama_three_kinds_with_full_coverage(ollama):
    r = ollama.choose("Which folder should this email go to?", ["Work", "Personal", "Spam"], context=SPAM)
    assert r.choice == "Spam" and r.method == "logprobs" and r.coverage > 0.9
    r = ollama.yes_no("Is this urgent?", context=OUTAGE)
    assert r.score > 0.9 and r.coverage > 0.9
    # the 0.1 bug: after "Answer:" the model writes " " first; coverage used to be ~0%
    r = ollama.score("How positive is this review?", context=GLOWING, scale=(1, 5))
    assert r.coverage > 0.9 and r.value >= 4.5


@live
@pytest.mark.live
def test_ollama_constrained_fallback_is_real(ollama):
    p = build_prompt(Decision(question="Which folder?", options=["Work", "Personal", "Spam"], context=SPAM))
    assert ollama.backend.constrained_choice(p) in p.labels


@live
@pytest.mark.live
def test_openai_compatible_endpoint(openai_compat):
    r = openai_compat.choose("Which folder should this email go to?", ["Work", "Personal", "Spam"], context=SPAM)
    assert r.choice == "Spam" and r.method == "logprobs" and r.coverage > 0.9
    r = openai_compat.score("How positive is this review?", context=GLOWING, scale=(1, 5))
    assert r.coverage > 0.9 and r.value >= 4.5
    p = build_prompt(Decision(kind="binary", question="Is this urgent?", context=OUTAGE))
    assert openai_compat.backend.constrained_choice(p) == "Yes"


@live
@pytest.mark.live
def test_rotate_debias_and_abstention(ollama):
    r = ollama.choose("Pick one at random.", ["Red", "Blue", "Green"], debias="rotate", min_confidence=0.9)
    assert sum(r.probs.values()) == pytest.approx(1.0)
    assert max(r.probs.values()) < 0.9 and r.abstained  # no right answer: position bias averaged out


@live
@pytest.mark.live
def test_jev_ask(ollama):
    res = ollama.ask(
        "I was charged twice for my subscription this month and nobody answers my emails. "
        "Refund the extra charge now. This is ridiculous.",
        {
            "department": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"billing": "Payments, refunds, invoices", "technical": "Bugs and outages",
                                        "sales": "Pricing and new accounts"}},
            "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Annoyed", "Very angry"]},
            "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
        },
    )
    a = res["answers"]
    assert a["department"]["choice"] == "billing"
    assert a["frustration"]["score"] >= 1.0
    assert a["refund"]["noul"] > 0.5


@live
@pytest.mark.live
def test_server_systemone_with_a_real_model(tmp_path):
    from verdict.server import create_app

    c = TestClient(create_app(str(tmp_path / "v.db"), allowed_hosts=["testserver"]))
    assert c.post("/api/models", json={"name": "local", "backend": "ollama", "config": {"model": LIVE_MODEL, "host": OLLAMA}}).status_code == 200
    r = c.post("/v1/systemone", json={"model": "jev-latest", "state": OUTAGE,
                                      "questions": {"urgent": {"type": "noul", "instructions": "Does this need an immediate response?"}}})
    assert r.status_code == 200, r.text
    assert r.json()["answers"]["urgent"]["noul"] > 0.9


@live
@pytest.mark.live
def test_mcp_server_over_stdio():
    mcp = pytest.importorskip("mcp")
    if not hasattr(mcp, "Client"):
        pytest.skip("needs mcp >= 2")
    from mcp.client.stdio import StdioServerParameters

    exe = shutil.which("verdict") or os.path.join(os.path.dirname(sys.executable), "verdict")
    params = StdioServerParameters(command=exe, args=["mcp", "-b", "ollama", "-m", LIVE_MODEL, "--base-url", OLLAMA])

    async def go():
        async with mcp.Client(params) as c:
            res = await c.call_tool("yes_no", {"question": "Is this urgent?", "context": OUTAGE})
            return json.loads(res.content[0].text)

    assert anyio.run(go)["p_yes"] > 0.9


@pytest.mark.hf
@pytest.mark.skipif(not HF_MODEL, reason="set VERDICT_HF_MODEL to run a Hugging Face model")
def test_huggingface_causal_model():
    pytest.importorskip("transformers")
    v = Verdict.from_backend("huggingface", model=HF_MODEL)
    assert v.backend.health()["ok"]
    r = v.choose("Which folder should this email go to?", ["Work", "Personal", "Spam"], context=SPAM)
    assert r.coverage > 0.9 and abs(sum(r.probs.values()) - 1) < 1e-6
    r = v.score("How positive is this review?", context=GLOWING, scale=(1, 5))
    assert r.coverage > 0.9 and r.value >= 4
    v.close()


@live
@pytest.mark.live
def test_systemone_backend_round_trip_through_a_real_verdict_server(tmp_path):
    """A Verdict server (backed by Ollama) used as a Jev-compatible model, over real HTTP."""
    import socket
    import threading
    import time

    import uvicorn

    from verdict.server import create_app

    app = create_app(str(tmp_path / "v.db"), token="t0k3n")
    with TestClient(app, base_url="http://localhost") as c:
        c.post("/api/models", headers={"Authorization": "Bearer t0k3n"},
               json={"name": "local", "backend": "ollama", "config": {"model": LIVE_MODEL, "host": OLLAMA}})
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    try:
        v = Verdict.from_backend("systemone", base_url=f"http://localhost:{port}/v1", api_key="t0k3n")
        r = v.choose("Which folder should this email go to?", ["Work", "Personal", "Spam"], context=SPAM)
        assert r.choice == "Spam" and r.coverage > 0.99
        assert v.yes_no("Is this urgent?", context=OUTAGE).score > 0.9
        assert v.score("How positive is this review?", context=GLOWING, scale=(1, 5)).value >= 4.5
    finally:
        server.should_exit = True
