from __future__ import annotations

import json

import httpx

from ..prompting import Prompt
from .base import Backend, BackendError, LogprobsUnsupported, answer_schema


class OllamaBackend(Backend):
    """Local models via Ollama (needs Ollama >= 0.12.11 for logprobs)."""

    kind = "ollama"
    config_fields = {
        "model": ("str", "qwen3:4b", "Model tag as shown by `ollama list`"),
        "host": ("str", "http://127.0.0.1:11434", "Ollama server URL"),
        "prefill": ("bool", True, "Start the reply with 'Answer:' so the first token is the label"),
        "keep_alive": ("str", "30m", "How long Ollama keeps the model loaded"),
        "timeout": ("float", 300.0, "Request timeout in seconds (first call loads the model)"),
    }

    def __init__(self, **config):
        super().__init__(**config)
        self.host = config.get("host", "http://127.0.0.1:11434").rstrip("/")
        self.client = httpx.Client(timeout=float(config.get("timeout", 300.0)))
        self._think_supported: bool | None = None

    @property
    def prefills(self) -> bool:
        return bool(self.config.get("prefill", True))

    def _messages(self, prompt: Prompt, prefill: bool) -> list[dict]:
        msgs = list(prompt.messages)
        if prefill:
            msgs.append({"role": "assistant", "content": prompt.prefill})
        return msgs

    def _chat(self, body: dict) -> dict:
        body = {
            "model": self.config["model"],
            "stream": False,
            "keep_alive": self.config.get("keep_alive", "30m"),
            **body,
        }
        # Turn off "thinking" for reasoning models; older/non-thinking models reject the flag.
        if self._think_supported is not False:
            body["think"] = False
        try:
            r = self.client.post(f"{self.host}/api/chat", json=body)
            if r.status_code == 400 and "think" in r.text and self._think_supported is None:
                self._think_supported = False
                body.pop("think", None)
                r = self.client.post(f"{self.host}/api/chat", json=body)
        except httpx.TimeoutException as e:
            raise BackendError(
                f"Ollama timed out after {self.client.timeout.read}s (model still loading? raise 'timeout')"
            ) from e
        except httpx.HTTPError as e:
            raise BackendError(f"cannot reach Ollama at {self.host}: {e}") from e
        if r.status_code != 200:
            raise BackendError(f"Ollama error {r.status_code}: {r.text[:300]}")
        return r.json()

    def top_logprobs(self, prompt: Prompt, k: int = 20) -> list[tuple[str, float]]:
        data = self._chat(
            {
                "messages": self._messages(prompt, self.prefills),
                "logprobs": True,
                "top_logprobs": min(k, 20),
                "options": {"num_predict": 1, "temperature": 0},
            }
        )
        lps = data.get("logprobs")
        if not lps:
            raise LogprobsUnsupported("this Ollama version did not return logprobs (upgrade to >= 0.12.11)")
        return [(t["token"], float(t["logprob"])) for t in lps[0].get("top_logprobs", [])]

    def constrained_choice(self, prompt: Prompt) -> str:
        data = self._chat(
            {
                "messages": self._messages(prompt, False),
                "format": answer_schema(prompt.labels),
                "options": {"num_predict": 40, "temperature": 0},
            }
        )
        try:
            return json.loads(data["message"]["content"])["answer"]
        except (KeyError, ValueError) as e:
            raise BackendError(f"unparseable constrained output: {data.get('message')}") from e

    def health(self) -> dict:
        try:
            r = self.client.get(f"{self.host}/api/tags", timeout=5)
            names = [m["name"] for m in r.json().get("models", [])]
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"cannot reach Ollama at {self.host}: {e}"}
        model = self.config.get("model", "")
        found = model in names or f"{model}:latest" in names
        return {"ok": found, "error": None if found else f"model '{model}' not pulled", "available": names}

    def close(self) -> None:
        self.client.close()
