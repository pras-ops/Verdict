"""Backend behaviour against recorded/simulated server responses (no network)."""
import json
import math

import httpx
import pytest

from verdict import Decision, Verdict
from verdict.backends.base import Backend, BackendError, LogprobsUnsupported
from verdict.backends.ollama import OllamaBackend
from verdict.backends.openai_compat import OpenAICompatBackend
from verdict.prompting import build_prompt


# -- whitespace before the label (Qwen / Llama 3 / GPT-4o tokenizers) ----------
class SpaceFirst(Backend):
    """Like Qwen after "Answer:": the next token is a bare space, the digit comes after it."""

    kind = "spacefirst"
    prefills = True

    def __init__(self, **config):
        super().__init__(**config)
        self.prefills_seen = []

    def top_logprobs(self, prompt, k=20):
        self.prefills_seen.append(prompt.prefill)
        if not prompt.prefill.endswith(" "):
            return [(" ", math.log(0.999)), ("5", math.log(0.0006)), ("4", math.log(0.0004))]
        return [("4", math.log(0.52)), ("5", math.log(0.48))]


def test_score_follows_a_leading_space_token():
    b = SpaceFirst()
    r = Verdict(b).score("How angry?", scale=(1, 5))
    assert b.prefills_seen == ["Answer:", "Answer: "]
    assert r.coverage == pytest.approx(1.0, abs=0.01)
    # the real distribution after the space, not the 60/40 noise in the tail
    assert r.probs["4"] == pytest.approx(0.52, abs=0.01)
    assert r.value == pytest.approx(4.48, abs=0.02)


def test_no_second_call_when_the_label_comes_first():
    class LabelFirst(SpaceFirst):
        def top_logprobs(self, prompt, k=20):
            self.prefills_seen.append(prompt.prefill)
            return [(" A", math.log(0.9)), (" ", math.log(0.05)), (" B", math.log(0.05))]

    b = LabelFirst()
    Verdict(b).choose("q", ["x", "y"])
    assert b.prefills_seen == ["Answer:"]


# -- OpenAI-compatible ---------------------------------------------------------
def _openai(handler, **config) -> OpenAICompatBackend:
    b = OpenAICompatBackend(model=config.pop("model", "m"), base_url="http://test/v1", **config)
    b.client = httpx.Client(transport=httpx.MockTransport(handler))
    return b


def _logprobs_reply(tokens):
    return {"choices": [{"logprobs": {"content": [{"token": tokens[0][0], "logprob": tokens[0][1],
                                                   "top_logprobs": [{"token": t, "logprob": lp} for t, lp in tokens]}]},
                         "message": {"content": tokens[0][0]}, "finish_reason": "length"}]}


def test_openai_reads_top_logprobs():
    sent = []

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json=_logprobs_reply([("B", math.log(0.7)), ("A", math.log(0.3))]))

    r = Verdict(_openai(handler)).choose("q", ["x", "y"])
    assert r.choice == "y" and r.probs["y"] == pytest.approx(0.7)
    assert sent[0]["logprobs"] is True and sent[0]["top_logprobs"] == 20 and sent[0]["max_tokens"] == 1
    assert sent[0]["temperature"] == 0


def test_openai_reasoning_model_drops_rejected_params_and_falls_back():
    """gpt-5 / o-series: temperature=0 and max_tokens are rejected, logprobs aren't offered."""
    sent = []

    def handler(req):
        body = json.loads(req.content)
        sent.append(body)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature' does not support 0 with this model. "
                                                       "Only the default (1) value is supported.", "param": "temperature"}})
        if "max_tokens" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'max_tokens' is not supported with this model. "
                                                       "Use 'max_completion_tokens' instead.", "param": "max_tokens"}})
        if body.get("logprobs"):
            return httpx.Response(400, json={"error": {"message": "logprobs are not supported with reasoning models", "param": "logprobs"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": "Yes"}'}, "finish_reason": "stop"}]})

    b = _openai(handler, model="gpt-5-mini", reasoning_effort="minimal")
    r = Verdict(b).yes_no("Is the sky blue?")
    assert r.choice == "Yes" and r.method == "constrained"
    final = sent[-1]
    assert "temperature" not in final and final["max_completion_tokens"] == 2048
    assert final["reasoning_effort"] == "minimal"
    # remembered: the next request goes out right first time
    n = len(sent)
    Verdict(b).yes_no("Again?")
    assert all("temperature" not in s and "max_tokens" not in s for s in sent[n:])


def test_openai_empty_answer_after_reasoning_explains_itself():
    def handler(req):
        body = json.loads(req.content)
        if body.get("logprobs"):
            return httpx.Response(400, json={"error": {"message": "logprobs not supported"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})

    with pytest.raises(BackendError, match="reasoning_effort"):
        Verdict(_openai(handler)).yes_no("q")


def test_openai_bare_label_when_json_schema_unsupported():
    def handler(req):
        body = json.loads(req.content)
        if body.get("logprobs"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "B"}}]})  # no logprobs field
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format is not supported"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "B."}, "finish_reason": "stop"}]})

    r = Verdict(_openai(handler)).choose("q", ["x", "y"])
    assert r.choice == "y" and r.method == "constrained"


def test_openai_unreachable_is_a_backend_error():
    def handler(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(BackendError, match="cannot reach"):
        _openai(handler).top_logprobs(build_prompt(Decision(kind="binary", question="q")))


# -- Ollama ----------------------------------------------------------------------
def _ollama(handler, **config) -> OllamaBackend:
    b = OllamaBackend(model="qwen2.5:7b", host="http://test", **config)
    b.client = httpx.Client(transport=httpx.MockTransport(handler))
    return b


def test_ollama_prefill_think_retry_and_logprobs():
    sent = []

    def handler(req):
        body = json.loads(req.content)
        sent.append(body)
        if body.get("think") is False:
            return httpx.Response(400, json={"error": '"qwen2.5:7b" does not support thinking'})
        return httpx.Response(200, json={"message": {"content": " Yes"},
                                         "logprobs": [{"token": " Yes", "logprob": -0.01,
                                                       "top_logprobs": [{"token": " Yes", "logprob": -0.01},
                                                                        {"token": " No", "logprob": -4.6}]}]})

    r = Verdict(_ollama(handler)).yes_no("Is it urgent?")
    assert r.choice == "Yes" and r.score > 0.98 and r.method == "logprobs"
    assert sent[0]["messages"][-1] == {"role": "assistant", "content": "Answer:"}
    assert sent[0]["think"] is False and "think" not in sent[1]  # retried without the flag
    assert sent[1]["top_logprobs"] == 20 and sent[1]["options"]["num_predict"] == 1


def test_ollama_old_version_without_logprobs_falls_back_to_constrained():
    def handler(req):
        body = json.loads(req.content)
        if "format" in body:
            return httpx.Response(200, json={"message": {"content": '{"answer": "C"}'}})
        return httpx.Response(200, json={"message": {"content": "C"}})  # no "logprobs" key

    r = Verdict(_ollama(handler)).choose("q", ["x", "y", "z"])
    assert r.choice == "z" and r.method == "constrained"


def test_ollama_logprobs_unsupported_without_fallback():
    def handler(req):
        return httpx.Response(200, json={"message": {"content": "C"}})

    with pytest.raises(LogprobsUnsupported):
        Verdict(_ollama(handler), fallback=False).choose("q", ["x", "y"])


def test_openai_newer_models_get_reasoning_effort_none_for_logprobs():
    """gpt-5.1+/gpt-6 return logprobs only when reasoning is off."""
    sent = []

    def handler(req):
        body = json.loads(req.content)
        sent.append(body)
        if body.get("logprobs") and body.get("reasoning_effort") != "none":
            return httpx.Response(400, json={"error": {"message": "logprobs are only supported when reasoning effort is none", "param": "logprobs"}})
        return httpx.Response(200, json=_logprobs_reply([("A", math.log(0.9)), ("B", math.log(0.1))]))

    b = _openai(handler, model="gpt-5.4-mini")
    r = Verdict(b).choose("q", ["x", "y"])
    assert r.method == "logprobs" and r.choice == "x"
    assert sent[-1]["reasoning_effort"] == "none" and len(sent) == 2
    Verdict(b).choose("q", ["x", "y"])
    assert len(sent) == 3  # remembered


def test_openai_model_that_always_reasons_falls_back_after_refusing_none():
    sent = []

    def handler(req):
        body = json.loads(req.content)
        sent.append(body)
        if body.get("reasoning_effort") == "none":
            return httpx.Response(400, json={"error": {"message": "Unsupported value: 'reasoning_effort' does not support 'none' with this model.",
                                                       "param": "reasoning_effort"}})
        if body.get("logprobs"):
            return httpx.Response(400, json={"error": {"message": "logprobs are not supported with reasoning models"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": "No"}'}, "finish_reason": "stop"}]})

    b = _openai(handler, model="gpt-5-mini")
    r = Verdict(b).yes_no("Is it raining?")
    assert r.choice == "No" and r.method == "constrained"
    assert "reasoning_effort" not in sent[-1]
    n = len(sent)
    Verdict(b).yes_no("Again?")
    assert len(sent) == n + 1 and not sent[-1].get("logprobs")  # goes straight to constrained output


# -- Jev / System One compatible APIs as a backend ---------------------------------
def test_systemone_backend_sends_typed_questions_and_reads_probabilities():
    from verdict.backends.systemone import SystemOneBackend

    sent = []

    def handler(req):
        body = json.loads(req.content)
        sent.append((req.headers.get("authorization"), body))
        q = body["questions"]["q"]
        if q["type"] == "noul":
            ans = {"type": "noul", "noul": 0.9}
        elif q["type"] == "choice":
            ans = {"type": "choice", "choice": "Spam", "probabilities": {"Work": 0.1, "Personal": 0.1, "Spam": 0.8}, "confidence": 0.5}
        else:
            ans = {"type": "score", "score": 1.9, "legend": {}, "probabilities": {"0": 0.1, "1": 0.0, "2": 0.9}, "confidence": 0.6}
        return httpx.Response(200, json={"model": "jev-1", "answers": {"q": ans}, "usage": {"input_tokens": 9, "output_tokens": 1}})

    b = SystemOneBackend(base_url="http://test/v1", api_key="k")
    b.client = httpx.Client(transport=httpx.MockTransport(handler), headers=b.client.headers)
    v = Verdict(b)
    r = v.choose("Which folder?", ["Work", "Personal", "Spam"], context="You won!")
    assert r.choice == "Spam" and r.probs["Spam"] == pytest.approx(0.8) and r.coverage == pytest.approx(1.0)
    auth, body = sent[-1]
    assert auth == "Bearer k" and body["state"] == "You won!" and body["questions"]["q"]["criteria"] == {o: o for o in ["Work", "Personal", "Spam"]}
    assert v.yes_no("Urgent?", context="x").score == pytest.approx(0.9)
    assert v.score("How?", context="x", scale=(1, 3)).value == pytest.approx(0.1 * 1 + 0.9 * 3)
    # calibration applies on top of the service's own probabilities
    assert Verdict(b, temperature=3).choose("Which folder?", ["Work", "Personal", "Spam"], context="You won!").confidence < 0.8


def test_systemone_backend_reports_a_bad_key():
    from verdict.backends.systemone import SystemOneBackend

    b = SystemOneBackend(base_url="http://test/v1")
    b.client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(401, json={"detail": "nope"})))
    with pytest.raises(BackendError, match="API key"):
        Verdict(b).yes_no("q")
