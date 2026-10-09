from __future__ import annotations

import math
import threading
from dataclasses import replace
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

    @property
    def prefills(self) -> bool:
        """True when `top_logprobs` continues `prompt.prefill` as the start of the reply."""
        return False

    # -- probability path -------------------------------------------------
    def top_logprobs(self, prompt: Prompt, k: int = 20) -> list[tuple[str, float]]:
        raise LogprobsUnsupported(f"{self.kind} does not expose token log-probabilities")

    def label_logprobs(self, prompt: Prompt) -> tuple[dict[str, float], list[tuple[str, float]]]:
        """Return ({label: logprob}, raw top tokens)."""
        top = self.top_logprobs(prompt)
        if self.prefills:
            top = self._follow_whitespace(prompt, top)
        return label_logprobs_from_top(top, prompt.labels), top

    def _follow_whitespace(self, prompt: Prompt, top: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Look one token further when the model's next token is only whitespace.

        Most current tokenizers (Qwen, Llama 3, GPT-4o) keep the space before a digit as
        its own token, so after "Answer:" the model writes " " and the digit comes one
        step later. We extend the prefill with that whitespace, ask again, and merge both
        steps into one distribution over the answer: P(" " then "4") = P(" ") * P("4" | " ").
        """
        ws = max(((t, lp) for t, lp in top if t and not t.strip()), key=lambda x: x[1], default=None)
        if ws is None:
            return top
        label_mass = sum(math.exp(lp) for lp in label_logprobs_from_top(top, prompt.labels).values())
        if math.exp(ws[1]) <= label_mass:
            return top
        nxt = self.top_logprobs(replace(prompt, prefill=prompt.prefill + ws[0]))
        merged = [(t, lp) for t, lp in top if t != ws[0]] + [(ws[0] + t, ws[1] + lp) for t, lp in nxt]
        return sorted(merged, key=lambda x: -x[1])

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
