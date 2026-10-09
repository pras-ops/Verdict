"""Jev / System One compatible requests: one state, several named, typed questions.

Same request and response shape as TypeSafe's `POST /v1/systemone`
(https://docs.typesafe.ai/api.md), so Jev clients can point at Verdict instead:

    {"model": "jev-latest", "state": "...",
     "questions": {"urgent":     {"type": "noul",   "instructions": "Needs a reply today?"},
                   "department": {"type": "choice", "instructions": "Which team?",
                                  "criteria": {"billing": "Payments", "technical": "Bugs"}},
                   "frustration":{"type": "score",  "instructions": "How frustrated?",
                                  "criteria": ["Calm", "Frustrated", "Very angry"]}}}

Verdict answers each question from token probabilities with whichever model you choose.
Differences from hosted Jev: at most 26 choice options, and `usage` is an estimate.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from concurrent.futures import Executor
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .prompting import build_prompt
from .types import Decision, Result

MAX_QUESTIONS = 64


class Question(BaseModel):
    type: Literal["noul", "choice", "score"]
    instructions: str | dict | list
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def _check(self) -> Question:
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or len(self.criteria) < 2:
                raise ValueError("choice questions need a criteria object with at least 2 options")
            if len(self.criteria) > 26:
                raise ValueError("Verdict supports at most 26 choice options (hosted Jev allows 255)")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
                raise ValueError("score questions need a criteria array of 2-10 ordered levels")
        elif self.criteria is not None and not isinstance(self.criteria, dict):
            raise ValueError("noul criteria must be an object like {\"true\": \"...\", \"false\": \"...\"}")
        return self


class SystemOneRequest(BaseModel):
    model: str = "jev-latest"
    state: str | dict | list
    questions: dict[str, Question] = Field(min_length=1, max_length=MAX_QUESTIONS)


def _text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, indent=2, ensure_ascii=False)


def entropy_confidence(probs: list[float]) -> float:
    """1 - normalised entropy: 1.0 when all mass is on one answer, 0.0 when spread evenly."""
    if len(probs) < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return round(max(0.0, 1.0 - h / math.log(len(probs))), 4)


def to_decision(q: Question, state: Any) -> tuple[Decision, Callable[[Result], dict]]:
    """A Jev question as a Verdict Decision, plus a function that turns the Result into a Jev answer."""
    ctx, instr = _text(state), _text(q.instructions).strip()

    if q.type == "noul":
        crit = q.criteria or {}
        hints = [f"{word} means: {crit[key]}" for key, word in (("true", "Yes"), ("false", "No")) if crit.get(key)]
        d = Decision(kind="binary", question=instr, context=ctx, instructions="\n".join(hints) or None)
        return d, lambda r: {"type": "noul", "noul": round(r.score, 6)}

    if q.type == "choice":
        keys = list(q.criteria)
        options = [f"{k}: {v}" if v else k for k, v in q.criteria.items()]
        d = Decision(kind="choice", question=instr, options=options, context=ctx)

        def choice_answer(r: Result) -> dict:
            probs = {k: round(r.probs[o], 6) for k, o in zip(keys, options)}
            return {
                "type": "choice",
                "choice": max(probs, key=probs.get),
                "probabilities": probs,
                "confidence": entropy_confidence(list(probs.values())),
            }

        return d, choice_answer

    levels = list(q.criteria)
    legend = {str(i): level for i, level in enumerate(levels)}
    scale_text = "Scale:\n" + "\n".join(f"{i} = {level}" for i, level in legend.items())
    d = Decision(kind="score", question=instr, context=ctx, scale=(0, len(levels) - 1), instructions=scale_text)

    def score_answer(r: Result) -> dict:
        probs = {i: round(r.probs[i], 6) for i in legend}
        return {
            "type": "score",
            "score": round(r.value, 4),
            "legend": legend,
            "probabilities": probs,
            "confidence": entropy_confidence(list(probs.values())),
        }

    return d, score_answer


def estimate_tokens(d: Decision) -> int:
    p = build_prompt(d)
    return sum(len(m["content"]) for m in p.messages) // 4


def answer(
    req: SystemOneRequest,
    decide: Callable[[Decision], Result],
    model_name: str,
    pool: Executor | None = None,
) -> dict:
    """Run every question (in parallel when a pool is given) and build the Jev response."""
    plan = {name: to_decision(q, req.state) for name, q in req.questions.items()}
    if pool:
        futures = {name: pool.submit(decide, d) for name, (d, _) in plan.items()}
        results = {name: f.result() for name, f in futures.items()}
    else:
        results = {name: decide(d) for name, (d, _) in plan.items()}
    return {
        "model": model_name,
        "answers": {name: to_answer(results[name]) for name, (_, to_answer) in plan.items()},
        "usage": {  # estimated: Verdict reads one token per question and doesn't count input tokens
            "input_tokens": sum(estimate_tokens(d) for d, _ in plan.values()),
            "output_tokens": len(plan),
        },
    }
