from __future__ import annotations

import math
import os

import httpx

from ..prompting import Prompt
from .base import Backend, BackendError


class SystemOneBackend(Backend):
    """A Jev / System One compatible decision API used as a model: TypeSafe's hosted Jev,
    Laya, Ollama decision models (Ollama >= 0.35), or another Verdict server.

    These services already return a probability per answer, so Verdict sends one typed
    question per decision and reads the answer back. Calibration, debiasing, evaluation
    and the dashboard then work on top of them like on any other model.
    """

    kind = "systemone"
    config_fields = {
        "model": ("str", "jev-latest", "Model name on that server"),
        "base_url": ("str", "https://api.typesafe.ai/v1", "The server's /v1 URL, e.g. http://localhost:11434/v1 for Ollama decision models"),
        "api_key_env": ("str", "TYPESAFE_API_KEY", "Name of the environment variable holding the API key (empty for local servers)"),
        "timeout": ("float", 60.0, "Request timeout in seconds"),
    }

    def __init__(self, **config):
        super().__init__(**config)
        self.base_url = config.get("base_url", "https://api.typesafe.ai/v1").rstrip("/")
        key = config.get("api_key") or os.environ.get(config.get("api_key_env") or "TYPESAFE_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.client = httpx.Client(timeout=float(config.get("timeout", 60.0)), headers=headers)

    @staticmethod
    def _question(prompt: Prompt) -> dict:
        if prompt.kind == "binary":
            return {"type": "noul", "instructions": prompt.question}
        if prompt.kind == "score":
            return {"type": "score", "instructions": prompt.question, "criteria": list(prompt.options)}
        return {"type": "choice", "instructions": prompt.question, "criteria": {o: o for o in prompt.options}}

    def label_logprobs(self, prompt: Prompt):
        body = {
            "model": self.config.get("model", "jev-latest"),
            "state": prompt.input_text or prompt.question,
            "questions": {"q": self._question(prompt)},
        }
        try:
            r = self.client.post(f"{self.base_url}/systemone", json=body)
        except httpx.HTTPError as e:
            raise BackendError(f"cannot reach {self.base_url}: {e}") from e
        if r.status_code == 401:
            raise BackendError(f"{self.base_url} rejected the API key (set {self.config.get('api_key_env') or 'TYPESAFE_API_KEY'})")
        if r.status_code != 200:
            raise BackendError(f"{self.base_url} error {r.status_code}: {r.text[:300]}")
        try:
            a = r.json()["answers"]["q"]
            if prompt.kind == "binary":
                probs = [float(a["noul"]), 1.0 - float(a["noul"])]
            elif prompt.kind == "score":  # score probabilities are keyed by level index
                probs = [float(a["probabilities"].get(str(i), 0.0)) for i in range(len(prompt.options))]
            else:
                probs = [float(a["probabilities"].get(o, 0.0)) for o in prompt.options]
        except (KeyError, TypeError, ValueError) as e:
            raise BackendError(f"unexpected answer from {self.base_url}: {r.text[:300]}") from e
        label_lp = {label: math.log(max(p, 1e-12)) for label, p in zip(prompt.labels, probs)}
        return label_lp, sorted(label_lp.items(), key=lambda x: -x[1])

    def close(self) -> None:
        self.client.close()
