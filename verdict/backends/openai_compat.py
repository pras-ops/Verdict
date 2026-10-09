from __future__ import annotations

import json
import os

import httpx

from ..prompting import Prompt
from .base import Backend, BackendError, LogprobsUnsupported, answer_schema


class OpenAICompatBackend(Backend):
    """Anything that speaks the OpenAI chat-completions API: OpenAI, vLLM, LM Studio,
    OpenRouter, Groq, Together, llama.cpp server, Ollama's /v1 ..."""

    kind = "openai"
    config_fields = {
        "model": ("str", "gpt-4.1-mini", "Model name on that server"),
        "base_url": ("str", "https://api.openai.com/v1", "e.g. http://localhost:8000/v1 for vLLM, http://localhost:1234/v1 for LM Studio"),
        "api_key_env": ("str", "OPENAI_API_KEY", "Name of the environment variable holding the API key"),
        "prefill": ("bool", False, "vLLM only: continue an 'Answer:' assistant message"),
        "max_tokens_param": ("str", "max_tokens", "max_tokens or max_completion_tokens (switched automatically if the server asks)"),
        "reasoning_effort": ("str", "", "Reasoning models only, e.g. minimal, none or low; empty = server default"),
        "timeout": ("float", 120.0, "Request timeout in seconds"),
    }

    def __init__(self, **config):
        super().__init__(**config)
        self.base_url = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        key = config.get("api_key") or os.environ.get(config.get("api_key_env") or "OPENAI_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.client = httpx.Client(timeout=float(config.get("timeout", 120.0)), headers=headers)
        # Learned from the server's 400 errors, then reused for every later request.
        self._max_param = config.get("max_tokens_param") or "max_tokens"
        self._send_temperature = True
        self._effort: str | None = config.get("reasoning_effort") or None
        self._effort_auto = False  # True once we added reasoning_effort="none" ourselves
        self._no_logprobs = False  # learned: this model never returns logprobs

    @property
    def prefills(self) -> bool:
        return bool(self.config.get("prefill"))

    def _post(self, body: dict) -> dict:
        body = {"model": self.config["model"], **body}
        if self._send_temperature:
            body.setdefault("temperature", 0)
        if self._effort:
            body["reasoning_effort"] = self._effort
        body.update(self.config.get("extra_body") or {})
        for _ in range(5):
            try:
                r = self.client.post(f"{self.base_url}/chat/completions", json=body)
            except httpx.TimeoutException as e:
                raise BackendError(f"{self.base_url} timed out after {self.client.timeout.read}s") from e
            except httpx.HTTPError as e:
                raise BackendError(f"cannot reach {self.base_url}: {e}") from e
            if r.status_code == 200:
                return r.json()
            if r.status_code in (400, 422) and self._adapt(body, r):
                continue
            if "logprobs" in body and r.status_code in (400, 403, 422) and "logprob" in r.text.lower():
                raise LogprobsUnsupported(r.text[:300])
            raise BackendError(f"{self.base_url} error {r.status_code}: {r.text[:300]}")
        raise BackendError(f"{self.base_url} kept rejecting the request: {r.text[:300]}")

    def _adapt(self, body: dict, r: httpx.Response) -> bool:
        """Drop or rename a parameter the model rejected, remember it, and say whether to retry.

        Reasoning models (OpenAI gpt-5, o-series) reject temperature=0 and max_tokens;
        gpt-5.1 and later only return logprobs with reasoning_effort="none"; some servers
        cap top_logprobs below 20 or don't support JSON-schema output.
        """
        try:
            err = r.json().get("error") or {}
        except (ValueError, AttributeError):
            err = {}
        if isinstance(err, list):
            err = err[0] if err and isinstance(err[0], dict) else {}
        if not isinstance(err, dict):
            err = {"message": str(err)}
        param = str(err.get("param") or "")
        msg = str(err.get("message") or r.text).lower()

        def about(name: str) -> bool:
            return param == name or name in msg

        if self._effort_auto and "reasoning_effort" in body and about("reasoning_effort"):
            # our "none" was refused (gpt-5, o-series, gpt-6-astra): this model always reasons,
            # so it never returns logprobs; remember that and let the engine use constrained output
            body.pop("reasoning_effort")
            self._effort, self._effort_auto, self._no_logprobs = None, False, True
            if body.get("logprobs"):
                raise LogprobsUnsupported("this model always reasons before answering, so it returns no logprobs")
            return True
        if body.get("logprobs") and "logprob" in msg and not self._effort and not self._no_logprobs:
            # gpt-5.1+, gpt-6: logprobs are only available without reasoning
            body["reasoning_effort"] = self._effort = "none"
            self._effort_auto = True
            return True
        if "temperature" in body and about("temperature"):
            body.pop("temperature")
            self._send_temperature = False
            return True
        if "max_tokens" in body and (about("max_tokens") or "max_completion_tokens" in msg):
            body["max_completion_tokens"] = body.pop("max_tokens")
            self._max_param = "max_completion_tokens"
            return True
        if body.get("top_logprobs", 0) > 5 and about("top_logprobs"):
            body["top_logprobs"] = 5
            return True
        if "response_format" in body and about("response_format"):
            body.pop("response_format")
            return True
        return False

    def top_logprobs(self, prompt: Prompt, k: int = 20) -> list[tuple[str, float]]:
        if self._no_logprobs:
            raise LogprobsUnsupported("this model returns no logprobs (learned from an earlier request)")
        msgs = list(prompt.messages)
        body: dict = {self._max_param: 1, "logprobs": True, "top_logprobs": min(k, 20)}
        if self.prefills:
            msgs.append({"role": "assistant", "content": prompt.prefill})
            body.update(continue_final_message=True, add_generation_prompt=False)
        body["messages"] = msgs
        data = self._post(body)
        try:
            first = data["choices"][0]["logprobs"]["content"][0]
        except (KeyError, IndexError, TypeError) as e:
            raise LogprobsUnsupported("server returned no logprobs") from e
        return [(t["token"], float(t["logprob"])) for t in first.get("top_logprobs") or []]

    def constrained_choice(self, prompt: Prompt) -> str:
        # Reasoning models spend tokens thinking before they answer, so give them room.
        budget = int(self.config.get("max_answer_tokens") or (40 if self._max_param == "max_tokens" else 2048))
        data = self._post(
            {
                "messages": prompt.messages,
                self._max_param: budget,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "decision", "schema": answer_schema(prompt.labels), "strict": True},
                },
            }
        )
        choice = data["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
        if not text.strip() and choice.get("finish_reason") == "length":
            raise BackendError(
                "the model used its whole token budget before answering; "
                "set reasoning_effort (e.g. minimal) or raise max_answer_tokens"
            )
        try:
            return json.loads(text)["answer"]
        except (ValueError, KeyError, TypeError):
            # Servers without JSON-schema support: accept a bare label.
            stripped = text.strip().strip(".").strip().strip('"')
            if stripped in prompt.labels:
                return stripped
            raise BackendError(f"unparseable constrained output: {text[:200]}") from None

    def health(self) -> dict:
        try:
            r = self.client.get(f"{self.base_url}/models", timeout=10)
            if r.status_code != 200:
                return {"ok": False, "error": f"{r.status_code}: {r.text[:200]}"}
            names = [m.get("id") for m in r.json().get("data", [])]
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}
        model = self.config.get("model")
        listed = not names or model in names
        return {"ok": listed, "error": None if listed else f"'{model}' not listed", "available": names[:200]}

    def close(self) -> None:
        self.client.close()
