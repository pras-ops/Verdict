from __future__ import annotations

import threading
from typing import Any, ClassVar

from ..prompting import Prompt
from ..scoring import label_logprobs_from_top


class LogprobsUnsupported(Exception):
    """The backend/model cannot return token probabilities; callers may fall back to constrained output."""


class BackendError(Exception):
    pass


class Backend:
    """A model connector.

    Subclasses implement `top_logprobs` (preferred) and/or `constrained_choice` (fallback),
    or override `label_logprobs` entirely when they can compute exact label probabilities.
    """

    kind: ClassVar[str] = "base"
    # Shown in the dashboard's "add model" form: name -> (type, default, help)
    config_fields: ClassVar[dict[str, tuple[str, Any, str]]] = {}

    def __init__(self, **config: Any):
        self.config = config
        self.lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return str(self.config.get("model", self.kind))

    # -- probability path -------------------------------------------------
    def top_logprobs(self, prompt: Prompt, k: int = 20) -> list[tuple[str, float]]:
        raise LogprobsUnsupported(f"{self.kind} does not expose token log-probabilities")

    def label_logprobs(self, prompt: Prompt) -> tuple[dict[str, float], list[tuple[str, float]]]:
        """Return ({label: logprob}, raw top tokens)."""
        top = self.top_logprobs(prompt)
        return label_logprobs_from_top(top, prompt.labels), top

    # -- fallback path ----------------------------------------------------
    def constrained_choice(self, prompt: Prompt) -> str:
        raise BackendError(f"{self.kind} supports neither logprobs nor constrained output")

    # -- housekeeping -----------------------------------------------------
    def health(self) -> dict:
        return {"ok": True}

    def close(self) -> None:
        pass


def answer_schema(labels: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": labels}},
        "required": ["answer"],
        "additionalProperties": False,
    }
