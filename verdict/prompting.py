"""Turn a Decision into chat messages plus the short answer labels we read probabilities for."""
from __future__ import annotations

import string
from dataclasses import dataclass

from .types import Decision

SYSTEM_PROMPT = (
    "You are a decision engine. Read the context and the question, then answer with only "
    "the label of the single best answer. Do not explain."
)
PREFILL = "Answer:"


@dataclass
class Prompt:
    messages: list[dict]  # system + user; the assistant prefill is added by backends that support it
    labels: list[str]  # what the model should emit, e.g. ["A", "B"] or ["Yes", "No"] or ["1".."5"]
    options: list[str]  # human-readable option for each label, same order
    prefill: str = PREFILL
    input_text: str = ""  # the raw text being judged (context, else question), for classifier heads
    kind: str = "choice"  # the Decision's kind, for backends that take typed questions (systemone)
    question: str = ""


def build_prompt(d: Decision, order: list[int] | None = None) -> Prompt:
    """`order` permutes choice options (used for position-bias correction)."""
    parts = []
    if d.context:
        parts.append(f"Context:\n{d.context.strip()}")
    if d.instructions:
        parts.append(d.instructions.strip())

    if d.kind == "choice":
        opts = [d.options[i] for i in (order or range(len(d.options)))]
        labels = list(string.ascii_uppercase[: len(opts)])
        listing = "\n".join(f"{l}) {o}" for l, o in zip(labels, opts))
        parts.append(f"Question: {d.question.strip()}\n\nOptions:\n{listing}")
        parts.append(f"Reply with a single letter ({labels[0]}-{labels[-1]}).")
    elif d.kind == "binary":
        opts = ["Yes", "No"]
        labels = ["Yes", "No"]
        parts.append(f"Question: {d.question.strip()}")
        parts.append("Reply with Yes or No.")
    else:
        lo, hi = d.scale
        labels = [str(i) for i in range(lo, hi + 1)]
        opts = labels
        parts.append(f"Question: {d.question.strip()}")
        parts.append(f"Reply with a single whole number from {lo} (lowest) to {hi} (highest).")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    return Prompt(
        messages=messages,
        labels=labels,
        options=opts,
        input_text=(d.context or d.question).strip(),
        kind=d.kind,
        question="\n\n".join(p for p in (d.instructions, d.question) if p).strip(),
    )
