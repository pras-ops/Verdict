"""Public data types: what you ask (Decision) and what you get back (Result)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Kind = Literal["choice", "binary", "score"]
Method = Literal["logprobs", "constrained"]


class Decision(BaseModel):
    """A single question for the model.

    kind="choice"  -> pick one of `options`
    kind="binary"  -> yes/no; `options` is ignored, Result.score = P(yes)
    kind="score"   -> rate on an integer scale (single digits, e.g. 1-5); Result.value = expected rating
    """

    kind: Kind = "choice"
    question: str
    options: list[str] = Field(default_factory=list)
    context: str | None = None
    scale: tuple[int, int] = (1, 5)
    instructions: str | None = None
    debias: bool = False  # choice only: also ask with options reversed and average (2x cost, less position bias)

    @model_validator(mode="after")
    def _check(self) -> "Decision":
        if self.kind == "choice":
            if len(self.options) < 2:
                raise ValueError("choice decisions need at least 2 options")
            if len(self.options) > 26:
                raise ValueError("choice decisions support at most 26 options")
            if len(set(self.options)) != len(self.options):
                raise ValueError("options must be unique")
        if self.kind == "score":
            lo, hi = self.scale
            if not (0 <= lo < hi <= 9):
                raise ValueError("scale must be single digits with 0 <= low < high <= 9")
        return self


class Result(BaseModel):
    kind: Kind
    model: str
    choice: str
    probs: dict[str, float]
    confidence: float
    score: float | None = None  # binary: P(yes); score: rating normalised to 0-1
    value: float | None = None  # score: expected rating on the original scale
    coverage: float | None = None  # probability mass the model put on valid answers (low = model wanted to say something else)
    method: Method
    latency_ms: float
    temperature: float = 1.0
    top_tokens: list[tuple[str, float]] = Field(default_factory=list)
    id: int | None = None  # set when logged by the server
