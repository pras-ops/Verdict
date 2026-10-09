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


# -- added in 0.2 ------------------------------------------------------------------
def test_calibration_refuses_to_recommend_from_too_few_or_perfect_examples():
    v = Verdict.from_backend("mock", model="mock", latency_ms=0)
    ex = [{"question": "red apple", "options": ["red apple", "blue sky"], "answer": "red apple"}] * 5
    out = evaluate(v, ex)
    assert out["reliable"] is False and "at least 30" in out["warning"]
    assert 0.25 <= out["fitted_temperature"] <= 20  # never below the search range
    out = evaluate(v, ex * 8)  # 40 examples, all right: still nothing to calibrate against
    assert out["reliable"] is False and "every example right" in out["warning"]


def test_calibration_recommends_with_enough_mixed_examples():
    v = Verdict(Fixed(p=0.95))  # always 95% on the first option
    ex = [{"question": "q", "options": ["x", "y"], "answer": "x"}] * 21 + [{"question": "q", "options": ["x", "y"], "answer": "y"}] * 9
    out = evaluate(v, ex)
    assert out["reliable"] is True and out["warning"] is None and out["fitted_temperature"] > 1


def test_calibration_ignores_constrained_answers_and_accepts_answer_case():
    out = evaluate(Verdict(NoLogprobs()), [{"question": "q", "kind": "binary", "answer": "no"}])
    assert out["error_count"] == 0 and out["fit_examples"] == 0 and out["fitted_temperature"] == 1.0


def test_rotate_debias_cancels_position_bias_for_many_options():
    r = Verdict(Fixed(p=0.7)).choose("q", ["a", "b", "c", "d"], debias="rotate")
    assert all(p == pytest.approx(0.25) for p in r.probs.values())


def test_min_confidence_flags_abstention():
    v = Verdict(Fixed(p=0.6))
    assert v.choose("q", ["x", "y"], min_confidence=0.8).abstained is True
    assert v.choose("q", ["x", "y"], min_confidence=0.5).abstained is False


def test_decide_many_keeps_order_and_can_return_errors():
    v = Verdict(Fixed(p=0.8))
    out = v.decide_many([{"question": "q", "options": ["x", "y"]}, {"question": "q", "options": ["solo"]}], return_exceptions=True)
    assert out[0].choice == "x" and isinstance(out[1], Exception)
    with pytest.raises(ValueError):
        v.decide_many([{"question": "q", "options": ["solo"]}])


def test_ask_returns_the_jev_response_shape():
    res = Verdict(Fixed(p=0.8)).ask(
        {"subject": "Stripe 403", "body": "keeps failing"},  # structured state is allowed
        {
            "urgent": {"type": "noul", "instructions": "Needs a reply today?", "criteria": {"true": "Blocking", "false": "Can wait"}},
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "technical": "Bugs"}},
            "mood": {"type": "score", "instructions": "How frustrated?", "criteria": ["Calm", "Annoyed", "Furious"]},
        },
    )
    a = res["answers"]
    assert a["urgent"] == {"type": "noul", "noul": pytest.approx(0.8)}
    assert a["team"]["choice"] == "billing" and set(a["team"]["probabilities"]) == {"billing", "technical"}
    assert 0 <= a["team"]["confidence"] <= 1
    assert a["mood"]["legend"] == {"0": "Calm", "1": "Annoyed", "2": "Furious"}
    assert a["mood"]["score"] == pytest.approx(0.8 * 0 + 0.1 * 1 + 0.1 * 2)
    assert res["usage"]["output_tokens"] == 3


def test_jev_questions_are_validated():
    from verdict.jev import Question

    with pytest.raises(ValueError):
        Question(type="score", instructions="?", criteria=["only one level"])
    with pytest.raises(ValueError):
        Question(type="choice", instructions="?", criteria={f"o{i}": "" for i in range(27)})


def test_prompt_carries_raw_input_text_for_classifier_heads():
    assert build_prompt(Decision(question="q", options=["x", "y"], context=" the email ")).input_text == "the email"
    assert build_prompt(Decision(kind="binary", question="Is it spam?")).input_text == "Is it spam?"
