"""Evaluate a model on labelled examples and fit a calibration temperature."""
from __future__ import annotations

import math

from .engine import Verdict
from .scoring import logsumexp
from .types import Decision

# Fewer examples than this (or a perfect score) can't tell how over- or under-confident a
# model is: with only right answers the best fit is always "be even more confident".
MIN_FIT_EXAMPLES = 30
T_MIN, T_MAX = 0.25, 20.0


def evaluate(v: Verdict, examples: list[dict], on_result=None) -> dict:
    """examples: [{"question", "options"?, "context"?, "kind"?, "scale"?, "answer"}].

    Runs at temperature 1, then fits the temperature that minimises log-loss and reports
    metrics for both. `reliable` says whether the fitted temperature is safe to apply.
    """
    raw = Verdict(v.backend, name=v.name, temperature=1.0, fallback=v.fallback)  # don't disturb live traffic
    rows = []  # (log-probs per option, index of correct option, latency, error, method)
    for ex in examples:
        ex = dict(ex)
        answer = str(ex.pop("answer", ""))
        try:
            d = Decision(**ex)
            r = raw.decide(d)
            opts = list(r.probs)
            if answer not in opts:
                # be forgiving about case and stray spaces ("yes" for "Yes")
                match = [o for o in opts if o.strip().lower() == answer.strip().lower()]
                if not match:
                    raise ValueError(f"answer '{answer}' is not one of {opts}")
                answer = match[0]
            lps = [math.log(max(r.probs[o], 1e-12)) for o in opts]
            rows.append((lps, opts.index(answer), r.latency_ms, None, r.method))
            if on_result:
                on_result(d, r, answer)
        except Exception as e:  # noqa: BLE001
            rows.append((None, None, None, str(e), None))

    ok = [r for r in rows if r[0] is not None]
    # Constrained answers are 0/1 and ignore temperature, so they would only distort the fit.
    fit = [(r[0], r[1]) for r in ok if r[4] == "logprobs"]
    best_t = fit_temperature(fit) if fit else 1.0
    raw_metrics = _metrics(ok, 1.0)
    wrong = sum(1 for lps, gold in fit if max(range(len(lps)), key=lps.__getitem__) != gold)
    reliable = len(fit) >= MIN_FIT_EXAMPLES and wrong > 0
    warning = None
    if not reliable:
        if len(fit) < MIN_FIT_EXAMPLES:
            warning = f"only {len(fit)} examples with probabilities; use at least {MIN_FIT_EXAMPLES} before applying a temperature"
        else:
            warning = "the model got every example right, so there is nothing to calibrate against; add harder examples"
    return {
        "model": v.name,
        "n": len(rows),
        "errors": [r[3] for r in rows if r[3]][:20],
        "error_count": sum(1 for r in rows if r[3]),
        "raw": raw_metrics,
        "calibrated": _metrics(ok, best_t),
        "fitted_temperature": best_t,
        "fit_examples": len(fit),
        "reliable": reliable,
        "warning": warning,
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
    """Grid + golden-section search for the temperature minimising negative log-likelihood,
    kept within [T_MIN, T_MAX]."""

    def nll(t: float) -> float:
        return sum(-math.log(max(_probs(lps, t)[g], 1e-12)) for lps, g in rows)

    grid = [T_MIN * 1.25**i for i in range(20)]  # 0.25 .. ~21
    grid = [t for t in grid if t <= T_MAX]
    best = min(grid, key=nll)
    lo, hi = max(T_MIN, best / 1.25), min(T_MAX, best * 1.25)
    g = (math.sqrt(5) - 1) / 2
    for _ in range(30):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return round((lo + hi) / 2, 3)
