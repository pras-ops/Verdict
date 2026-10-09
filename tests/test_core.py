import math

import pytest

from verdict import Decision, Verdict, fit_temperature
from verdict.backends.base import Backend, LogprobsUnsupported
from verdict.calibration import evaluate
from verdict.prompting import build_prompt
from verdict.scoring import finalize, label_logprobs_from_top


def test_label_matching_merges_variants_and_ignores_lowercase_article():
    top = [(" A", math.log(0.5)), ("A)", math.log(0.1)), (" a", math.log(0.2)), (" B", math.log(0.15))]
    lp = label_logprobs_from_top(top, ["A", "B"])
    assert math.exp(lp["A"]) == pytest.approx(0.6)
    assert math.exp(lp["B"]) == pytest.approx(0.15)


def test_yes_no_is_case_insensitive():
    lp = label_logprobs_from_top([(" yes", -0.1), ("NO", -2.5)], ["Yes", "No"])
    assert set(lp) == {"Yes", "No"}


def test_finalize_normalises_and_reports_coverage():
    probs, cov = finalize({"A": math.log(0.6), "B": math.log(0.2)}, ["A", "B"])
    assert probs == pytest.approx([0.75, 0.25])
    assert cov == pytest.approx(0.8)


def test_finalize_fills_missing_labels_below_smallest_seen():
    top = [(" A", math.log(0.7)), (" B", math.log(0.29))]
    probs, _ = finalize(label_logprobs_from_top(top, ["A", "B", "C"]), ["A", "B", "C"], top)
    assert probs[2] < probs[1] < probs[0]
    assert sum(probs) == pytest.approx(1.0)


def test_temperature_softens():
    sharp, _ = finalize({"A": math.log(0.9), "B": math.log(0.1)}, ["A", "B"], temperature=1.0)
    soft, _ = finalize({"A": math.log(0.9), "B": math.log(0.1)}, ["A", "B"], temperature=3.0)
    assert soft[0] < sharp[0]


def test_prompt_labels_per_kind():
    assert build_prompt(Decision(question="q", options=["x", "y", "z"])).labels == ["A", "B", "C"]
    assert build_prompt(Decision(kind="binary", question="q")).labels == ["Yes", "No"]
    assert build_prompt(Decision(kind="score", question="q", scale=(0, 3))).labels == ["0", "1", "2", "3"]


def test_decision_validation():
    with pytest.raises(ValueError):
        Decision(question="q", options=["only one"])
    with pytest.raises(ValueError):
        Decision(kind="score", question="q", scale=(1, 10))


class Fixed(Backend):
    """Always puts `p` on the first label (in presented order)."""

    kind = "fixed"

    def top_logprobs(self, prompt, k=20):
        p = self.config.get("p", 0.8)
        rest = (1 - p) / (len(prompt.labels) - 1)
        return [(" " + prompt.labels[0], math.log(p))] + [(" " + l, math.log(rest)) for l in prompt.labels[1:]]


def test_engine_choice_binary_score():
    v = Verdict(Fixed(p=0.8))
    r = v.choose("q", ["x", "y"])
    assert r.choice == "x" and r.probs["x"] == pytest.approx(0.8) and r.method == "logprobs"
    assert v.yes_no("q").score == pytest.approx(0.8)
    s = v.score("q", scale=(1, 3))
    assert s.value == pytest.approx(0.8 * 1 + 0.1 * 2 + 0.1 * 3)


def test_debias_cancels_pure_position_bias():
    r = Verdict(Fixed(p=0.8)).choose("q", ["x", "y"], debias=True)
    assert r.probs["x"] == pytest.approx(0.5)


class NoLogprobs(Backend):
    kind = "nolp"

    def constrained_choice(self, prompt):
        return prompt.labels[-1]


def test_constrained_fallback():
    r = Verdict(NoLogprobs()).choose("q", ["x", "y", "z"])
    assert r.choice == "z" and r.method == "constrained" and r.coverage is None
    with pytest.raises(LogprobsUnsupported):
        Verdict(NoLogprobs(), fallback=False).choose("q", ["x", "y"])


def test_fit_temperature_recovers_overconfidence():
    # Model says 99% but is right 70% of the time -> temperature should be > 1.
    rows = [([math.log(0.99), math.log(0.01)], 0)] * 70 + [([math.log(0.99), math.log(0.01)], 1)] * 30
    t = fit_temperature(rows)
    assert t > 2
    p = 1 / (1 + math.exp(-(math.log(0.99) - math.log(0.01)) / t))
    assert p == pytest.approx(0.7, abs=0.02)


def test_evaluate_reports_metrics():
    v = Verdict.from_backend("mock", model="mock", latency_ms=0)
    ex = [
        {"question": "invoice from accounting team", "options": ["accounting invoice", "holiday photos"], "answer": "accounting invoice"},
        {"question": "holiday photos from mum", "options": ["accounting invoice", "holiday photos"], "answer": "holiday photos"},
        {"question": "bad row", "options": ["a", "b"], "answer": "not an option"},
    ]
    out = evaluate(v, ex)
    assert out["n"] == 3 and out["error_count"] == 1
    assert out["raw"]["accuracy"] == 1.0
