"""Map raw next-token log-probabilities onto answer labels and normalise them."""
from __future__ import annotations

import math

_STRIP = " \t\n\r*_\"'`([{"
_TRAIL = " \t\n\r*_\"'`)]}.:,;"


def normalize_token(tok: str) -> str:
    return tok.lstrip(_STRIP).rstrip(_TRAIL)


def _matches(norm: str, label: str) -> bool:
    if not norm:
        return False
    # Single letters are case-sensitive so the article "a" is not read as option A.
    if len(label) == 1 and label.isalpha():
        return norm == label
    return norm.lower() == label.lower()


def label_logprobs_from_top(top: list[tuple[str, float]], labels: list[str]) -> dict[str, float]:
    """Sum the probability of every token variant of a label (" A", "A", "A)" ...)."""
    mass: dict[str, float] = {}
    for tok, lp in top:
        norm = normalize_token(tok)
        for label in labels:
            if _matches(norm, label):
                mass[label] = mass.get(label, 0.0) + math.exp(lp)
                break
    return {l: math.log(m) for l, m in mass.items() if m > 0}


def logsumexp(xs: list[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


def finalize(
    label_lp: dict[str, float],
    labels: list[str],
    top: list[tuple[str, float]] | None = None,
    temperature: float = 1.0,
) -> tuple[list[float], float]:
    """Return (probabilities per label, coverage).

    Labels missing from a top-k list get an upper-bound estimate: at most the smallest
    probability we saw, and together no more than the unseen mass.
    """
    if not label_lp:
        raise ValueError("no answer label found in model output")
    coverage = sum(math.exp(v) for v in label_lp.values())
    missing = [l for l in labels if l not in label_lp]
    lps = dict(label_lp)
    if missing:
        if top:
            floor = min(lp for _, lp in top)
            unseen = max(1e-12, 1.0 - sum(math.exp(lp) for _, lp in top))
            fill = min(floor, math.log(unseen / len(missing)))
        else:
            fill = min(label_lp.values()) - 10.0
        for l in missing:
            lps[l] = fill
    t = max(temperature, 1e-3)
    scaled = [lps[l] / t for l in labels]
    z = logsumexp(scaled)
    return [math.exp(s - z) for s in scaled], min(coverage, 1.0)
