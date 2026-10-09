"""Evaluate a model on labelled examples and fit a calibration temperature."""
from __future__ import annotations

import math

from .engine import Verdict
from .scoring import logsumexp
from .types import Decision


def evaluate(v: Verdict, examples: list[dict], on_result=None) -> dict:
    """examples: [{"question", "options"?, "context"?, "kind"?, "scale"?, "answer"}].

    Runs at temperature 1, then fits the temperature that minimises log-loss and reports
    metrics for both.
    """
    raw = Verdict(v.backend, name=v.name, temperature=1.0, fallback=v.fallback)  # don't disturb live traffic
    rows = []  # (log-probs per option, index of correct option, latency, error)
    for ex in examples:
        ex = dict(ex)
        answer = str(ex.pop("answer", ""))
        try:
            d = Decision(**ex)
            r = raw.decide(d)
            opts = list(r.probs)
            if answer not in opts:
                raise ValueError(f"answer '{answer}' is not one of {opts}")
            lps = [math.log(max(r.probs[o], 1e-12)) for o in opts]
            rows.append((lps, opts.index(answer), r.latency_ms, None))
            if on_result:
                on_result(d, r, answer)
        except Exception as e:  # noqa: BLE001
            rows.append((None, None, None, str(e)))

    ok = [r for r in rows if r[0] is not None]
    best_t = fit_temperature([(r[0], r[1]) for r in ok]) if ok else 1.0
    return {
        "model": v.name,
        "n": len(rows),
        "errors": [r[3] for r in rows if r[3]][:20],
        "error_count": sum(1 for r in rows if r[3]),
        "raw": _metrics(ok, 1.0),
        "calibrated": _metrics(ok, best_t),
        "fitted_temperature": best_t,
        "avg_latency_ms": round(sum(r[2] for r in ok) / len(ok), 2) if ok else None,
    }


def _probs(lps: list[float], t: float) -> list[float]:
    s = [x / t for x in lps]
    z = logsumexp(s)
    return [math.exp(x - z) for x in s]


def _metrics(rows, t: float) -> dict:
    if not rows:
        return {"accuracy": None, "log_loss": None, "ece": None}
    n = len(rows)
    correct, nll, bins = 0, 0.0, [[0, 0.0, 0] for _ in range(10)]  # count, conf sum, correct
    for lps, gold, *_ in rows:
        p = _probs(lps, t)
        pred = max(range(len(p)), key=p.__getitem__)
        hit = int(pred == gold)
        correct += hit
        nll -= math.log(max(p[gold], 1e-12))
        b = bins[min(int(p[pred] * 10), 9)]
        b[0] += 1
        b[1] += p[pred]
        b[2] += hit
    ece = sum(abs(b[1] / b[0] - b[2] / b[0]) * b[0] / n for b in bins if b[0])
    return {"accuracy": round(correct / n, 4), "log_loss": round(nll / n, 4), "ece": round(ece, 4)}


def fit_temperature(rows: list[tuple[list[float], int]]) -> float:
    """Grid + golden-section search for the temperature minimising negative log-likelihood."""

    def nll(t: float) -> float:
        return sum(-math.log(max(_probs(lps, t)[g], 1e-12)) for lps, g in rows)

    grid = [0.25 * 1.25**i for i in range(20)]  # 0.25 .. ~21
    best = min(grid, key=nll)
    lo, hi = best / 1.25, best * 1.25
    g = (math.sqrt(5) - 1) / 2
    for _ in range(30):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return round((lo + hi) / 2, 3)
