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
        "model": ("str", "gpt-4o-mini", "Model name on that server"),
        "base_url": ("str", "https://api.openai.com/v1", "e.g. http://localhost:8000/v1 for vLLM, http://localhost:1234/v1 for LM Studio"),
        "api_key_env": ("str", "OPENAI_API_KEY", "Name of the environment variable holding the API key"),
        "prefill": ("bool", False, "vLLM only: continue an 'Answer:' assistant message"),
        "max_tokens_param": ("str", "max_tokens", "Use max_completion_tokens for newer OpenAI models"),
        "timeout": ("float", 120.0, "Request timeout in seconds"),
    }

    def __init__(self, **config):
        super().__init__(**config)
        self.base_url = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        key = config.get("api_key") or os.environ.get(config.get("api_key_env") or "OPENAI_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.client = httpx.Client(timeout=float(config.get("timeout", 120.0)), headers=headers)

    def _post(self, body: dict) -> dict:
        body = {"model": self.config["model"], "temperature": 0, **body, **(self.config.get("extra_body") or {})}
        try:
            r = self.client.post(f"{self.base_url}/chat/completions", json=body)
        except httpx.HTTPError as e:
            raise BackendError(f"cannot reach {self.base_url}: {e}") from e
        if r.status_code != 200:
            if "logprobs" in body and r.status_code == 400 and "logprob" in r.text.lower():
                raise LogprobsUnsupported(r.text[:300])
            raise BackendError(f"{self.base_url} error {r.status_code}: {r.text[:300]}")
        return r.json()

    def top_logprobs(self, prompt: Prompt, k: int = 20) -> list[tuple[str, float]]:
        msgs = list(prompt.messages)
        body: dict = {
            self.config.get("max_tokens_param", "max_tokens"): 1,
            "logprobs": True,
            "top_logprobs": min(k, 20),
        }
        if self.config.get("prefill"):
            msgs.append({"role": "assistant", "content": prompt.prefill})
            body.update(continue_final_message=True, add_generation_prompt=False)
        body["messages"] = msgs
        data = self._post(body)
        try:
            content = data["choices"][0]["logprobs"]["content"]
            first = content[0]
        except (KeyError, IndexError, TypeError) as e:
            raise LogprobsUnsupported("server returned no logprobs") from e
        return [(t["token"], float(t["logprob"])) for t in first.get("top_logprobs") or []]

    def constrained_choice(self, prompt: Prompt) -> str:
        data = self._post(
            {
                "messages": prompt.messages,
                self.config.get("max_tokens_param", "max_tokens"): 40,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "decision", "schema": answer_schema(prompt.labels), "strict": True},
                },
            }
        )
        text = data["choices"][0]["message"].get("content") or ""
        try:
            return json.loads(text)["answer"]
        except (ValueError, KeyError):
            # Servers without JSON-schema support: accept a bare label.
            stripped = text.strip().strip(".").strip()
            if stripped in prompt.labels:
                return stripped
            raise BackendError(f"unparseable constrained output: {text[:200]}")

    def health(self) -> dict:
        try:
            r = self.client.get(f"{self.base_url}/models", timeout=10)
            if r.status_code != 200:
                return {"ok": False, "error": f"{r.status_code}: {r.text[:200]}"}
            names = [m.get("id") for m in r.json().get("data", [])]
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}
        model = self.config.get("model")
        return {"ok": not names or model in names, "error": None if (not names or model in names) else f"'{model}' not listed", "available": names[:200]}

    def close(self) -> None:
        self.client.close()
