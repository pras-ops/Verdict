from __future__ import annotations

import hashlib
import math
import time

from ..prompting import Prompt
from .base import Backend


class MockBackend(Backend):
    """Deterministic fake model for tests and for trying the dashboard without a real model.

    Scores each label by keyword overlap between the option text and the prompt, plus a
    stable hash-based jitter.
    """

    kind = "mock"
    config_fields = {
        "model": ("str", "mock", "Any name"),
        "latency_ms": ("float", 5.0, "Simulated latency"),
        "sharpness": ("float", 3.0, "Higher = more confident"),
    }

    def top_logprobs(self, prompt: Prompt, k: int = 20):
        time.sleep(float(self.config.get("latency_ms", 5.0)) / 1000)
        text = prompt.messages[-1]["content"].lower()
        sharp = float(self.config.get("sharpness", 3.0))
        scores = []
        for _label, option in zip(prompt.labels, prompt.options):
            words = [w for w in option.lower().split() if len(w) > 2]
            overlap = sum(max(0, text.count(w) - 1) for w in words)  # minus the option listing itself
            jitter = int(hashlib.md5(f"{text}|{option}".encode()).hexdigest()[:6], 16) / 0xFFFFFF
            scores.append(sharp * overlap + jitter)
        z = math.log(sum(math.exp(s) for s in scores) + 0.05)
        return [(f" {l}", s - z) for l, s in zip(prompt.labels, scores)][:k]

    def constrained_choice(self, prompt: Prompt) -> str:
        top = self.top_logprobs(prompt)
        return max(top, key=lambda t: t[1])[0].strip()
