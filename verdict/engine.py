"""The decision engine: Decision in, calibrated Result out, for any backend."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .backends import Backend, BackendError, LogprobsUnsupported, make_backend
from .jev import SystemOneRequest, answer
from .prompting import build_prompt
from .scoring import finalize
from .types import Decision, Result


class Verdict:
    """Wrap any backend and ask it structured questions.

    >>> v = Verdict.from_backend("ollama", model="qwen3:4b")
    >>> v.choose("Which folder?", ["Work", "Personal", "Spam"], context=email).choice
    """

    def __init__(self, backend: Backend, name: str | None = None, temperature: float = 1.0, fallback: bool = True):
        self.backend = backend
        self.name = name or backend.model_id
        self.temperature = temperature  # calibration: >1 softens overconfident models, <1 sharpens
        self.fallback = fallback

    @classmethod
    def from_backend(cls, kind: str, name: str | None = None, temperature: float = 1.0, **config) -> Verdict:
        return cls(make_backend(kind, **config), name=name, temperature=temperature)

    # -- convenience ------------------------------------------------------
    def choose(self, question: str, options: list[str], context: str | None = None, **kw) -> Result:
        return self.decide(Decision(kind="choice", question=question, options=options, context=context, **kw))

    def yes_no(self, question: str, context: str | None = None, **kw) -> Result:
        return self.decide(Decision(kind="binary", question=question, context=context, **kw))

    def score(self, question: str, context: str | None = None, scale: tuple[int, int] = (1, 5), **kw) -> Result:
        return self.decide(Decision(kind="score", question=question, context=context, scale=scale, **kw))

    def ask(self, state: Any, questions: dict) -> dict:
        """Several named questions about one state, in the Jev / System One format.

        >>> v.ask(ticket, {"urgent": {"type": "noul", "instructions": "Needs a reply today?"}})
        {"model": ..., "answers": {"urgent": {"type": "noul", "noul": 0.97}}, "usage": {...}}
        """
        return answer(SystemOneRequest(state=state, questions=questions), self.decide, self.name)

    def decide_many(
        self, decisions: list[Decision | dict], workers: int = 4, return_exceptions: bool = False
    ) -> list[Result | Exception]:
        """Run many decisions in parallel threads, keeping input order.

        With return_exceptions=True a failed item gives its exception instead of stopping the batch.
        """

        def one(d):
            try:
                return self.decide(d)
            except Exception as e:  # noqa: BLE001
                if not return_exceptions:
                    raise
                return e

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(one, decisions))

    # -- core ---------------------------------------------------------------
    def decide(self, d: Decision | dict) -> Result:
        if isinstance(d, dict):
            d = Decision(**d)
        t0 = time.perf_counter()
        orders: list[list[int] | None] = [None]
        n = len(d.options)
        if d.kind == "choice" and d.debias == "rotate":
            orders += [[(i + k) % n for i in range(n)] for k in range(1, n)]
        elif d.kind == "choice" and d.debias:
            orders.append(list(reversed(range(n))))

        runs = [self._run_once(d, order) for order in orders]
        options = runs[0][0]
        method = "constrained" if any(r[2] == "constrained" for r in runs) else "logprobs"
        probs = {o: sum(r[1][o] for r in runs) / len(runs) for o in options}
        coverages = [r[3] for r in runs if r[3] is not None]
        latency = (time.perf_counter() - t0) * 1000

        choice = max(probs, key=probs.get)
        res = Result(
            kind=d.kind,
            model=self.name,
            choice=choice,
            probs=probs,
            confidence=probs[choice],
            coverage=sum(coverages) / len(coverages) if coverages else None,
            method=method,
            latency_ms=round(latency, 2),
            temperature=self.temperature,
            top_tokens=[(t, round(lp, 4)) for t, lp in runs[0][4]][:10],
            abstained=d.min_confidence is not None and probs[choice] < d.min_confidence,
        )
        if d.kind == "binary":
            res.score = probs["Yes"]
        elif d.kind == "score":
            lo, hi = d.scale
            res.value = sum(int(o) * p for o, p in probs.items())
            res.score = (res.value - lo) / (hi - lo)
        return res

    def _run_once(self, d: Decision, order: list[int] | None):
        """Returns (options in original order, {option: p}, method, coverage, top_tokens)."""
        p = build_prompt(d, order)
        try:
            label_lp, top = self.backend.label_logprobs(p)
            probs, coverage = finalize(label_lp, p.labels, top, self.temperature)
            method = "logprobs"
        except (LogprobsUnsupported, ValueError):
            if not self.fallback:
                raise
            answer = self.backend.constrained_choice(p)
            if answer not in p.labels:
                raise BackendError(f"model answered '{answer}', expected one of {p.labels}") from None
            probs = [1.0 if l == answer else 0.0 for l in p.labels]
            coverage, top, method = None, [], "constrained"
        by_option = dict(zip(p.options, probs))
        original = d.options if d.kind == "choice" else p.options
        return original, {o: by_option[o] for o in original}, method, coverage, top

    def close(self) -> None:
        self.backend.close()
