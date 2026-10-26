"""Unit tests for the metric functions.

These are the tests that matter most in this repo: a bug in a metric does not
crash anything, it just produces a number you then defend in a review. Each
test below pins a property the metric must have, not merely a value it happened
to return.
"""
from __future__ import annotations

import pytest

from ragevals.evaluation.metrics import (
    answer_correctness_lexical,
    citation_metrics,
    context_precision,
    context_recall,
    extract_claims,
    faithfulness,
    key_fact_recall,
    refusal_metrics,
    token_f1,
    token_recall,
)

CONTEXT = [
    "The controller must notify the supervisory authority within 72 hours of becoming "
    "aware of a personal data breach.",
    "A data protection impact assessment is required where processing is likely to "
    "result in a high risk to the rights and freedoms of natural persons.",
]


# --------------------------------------------------------------------------- #
# token_f1
# --------------------------------------------------------------------------- #
def test_token_f1_identical_is_one():
    assert token_f1("the breach must be notified", "the breach must be notified") == 1.0


def test_token_f1_disjoint_is_zero():
    assert token_f1("completely unrelated wording", "supervisory authority notification") == 0.0


def test_token_f1_punishes_padding():
    reference = "notify the supervisory authority within 72 hours"
    tight = token_f1("notify the supervisory authority within 72 hours", reference)
    padded = token_f1("notify the supervisory authority within 72 hours " + "filler " * 40,
                      reference)
    assert tight > padded


def test_token_f1_punishes_omission():
    reference = "notify the supervisory authority within 72 hours of awareness"
    assert token_f1("notify someone", reference) < 0.5


def test_token_f1_is_symmetric_in_value():
    a, b = "alpha beta gamma", "beta gamma delta"
    assert token_f1(a, b) == pytest.approx(token_f1(b, a))


# --------------------------------------------------------------------------- #
# faithfulness
# --------------------------------------------------------------------------- #
def test_faithfulness_grounded_answer_scores_one():
    answer = ("The controller must notify the supervisory authority within 72 hours "
              "of becoming aware of a personal data breach. [gdpr.pdf, p. 3]")
    assert faithfulness(answer, CONTEXT).value == 1.0


def test_faithfulness_detects_an_invented_claim():
    answer = ("The controller must notify the authority within 72 hours. "
              "Offenders are permanently banned from trading in the single market.")
    result = faithfulness(answer, CONTEXT)
    assert result.value < 1.0
    assert result.detail["claims"] == 2


def test_faithfulness_of_a_refusal_is_one_by_convention():
    # A refusal makes no claims, so there is nothing to be unfaithful about.
    assert faithfulness("INSUFFICIENT_CONTEXT", CONTEXT).value == 1.0
    assert faithfulness("", CONTEXT).value == 1.0


def test_faithfulness_ignores_correctness():
    # Grounded but wrong-for-the-question content is still faithful: that is the
    # definition, and the reason faithfulness alone is not enough.
    answer = "A data protection impact assessment is required for high risk processing."
    assert faithfulness(answer, CONTEXT).value == 1.0


def test_faithfulness_with_empty_context_is_zero_for_a_real_answer():
    assert faithfulness("The deadline is 72 hours and the fine is 4 per cent.", []).value == 0.0


# --------------------------------------------------------------------------- #
# context precision / recall
# --------------------------------------------------------------------------- #
def test_context_precision_rewards_putting_relevant_chunks_first():
    reference = "Notify the supervisory authority within 72 hours."
    good = context_precision([CONTEXT[0], "irrelevant filler text about parking"], reference)
    bad = context_precision(["irrelevant filler text about parking", CONTEXT[0]], reference)
    assert good.value > bad.value


def test_context_precision_is_one_when_every_chunk_is_relevant():
    reference = "Notify the supervisory authority within 72 hours."
    assert context_precision([CONTEXT[0]], reference).value == 1.0


def test_context_precision_uses_annotations_when_available():
    gold = {("gdpr.pdf", 3)}
    hits = [("gdpr.pdf", 3), ("gdpr.pdf", 9)]
    result = context_precision(["a", "b"], "", gold_keys=gold, hit_keys=hits)
    assert result.value == 1.0
    assert result.detail["flags"] == [True, False]


def test_context_precision_empty_retrieval_is_zero():
    assert context_precision([], "anything").value == 0.0


def test_context_recall_annotated_counts_found_gold_chunks():
    gold = {("gdpr.pdf", 3), ("gdpr.pdf", 4)}
    hits = [("gdpr.pdf", 3)]
    assert context_recall("", [], gold_keys=gold, hit_keys=hits).value == 0.5


def test_context_recall_lexical_detects_missing_evidence():
    reference = ("The controller must notify the supervisory authority within 72 hours "
                 "of becoming aware of a personal data breach.")
    assert context_recall(reference, [CONTEXT[0]]).value == 1.0
    assert context_recall(reference, ["an unrelated passage about parking"]).value == 0.0


# --------------------------------------------------------------------------- #
# answer correctness
# --------------------------------------------------------------------------- #
def test_correctness_rewards_exact_numbers():
    reference = "The deadline is 72 hours."
    right = answer_correctness_lexical("The deadline is 72 hours.", reference).value
    wrong = answer_correctness_lexical("The deadline is 27 hours.", reference).value
    assert right > wrong, "key-fact recall must separate 72 from 27"


def test_key_fact_recall_catches_article_references():
    assert key_fact_recall("See Article 35 of the Regulation.", "Article 35 applies.") == 1.0
    assert key_fact_recall("See Article 22.", "Article 35 applies.") == 0.0


def test_correctness_of_an_empty_answer_is_low():
    assert answer_correctness_lexical("", "The deadline is 72 hours.").value < 0.3


# --------------------------------------------------------------------------- #
# refusal
# --------------------------------------------------------------------------- #
def test_refusal_metrics_perfect_behaviour():
    out = refusal_metrics([True, False, True, False], [True, False, True, False])
    assert out["refusal_accuracy"] == 1.0
    assert out["over_refusal_rate"] == 0.0
    assert out["hallucination_rate_on_unanswerable"] == 0.0


def test_refusal_metrics_never_refusing():
    out = refusal_metrics([True, True, False, False], [False, False, False, False])
    assert out["refusal_accuracy"] == 0.5
    assert out["refusal_recall"] == 0.0
    assert out["hallucination_rate_on_unanswerable"] == 1.0
    assert out["over_refusal_rate"] == 0.0


def test_refusal_metrics_always_refusing():
    out = refusal_metrics([True, True, False, False], [True, True, True, True])
    assert out["refusal_recall"] == 1.0
    assert out["over_refusal_rate"] == 1.0
    assert out["refusal_precision"] == 0.5


def test_refusal_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        refusal_metrics([True], [True, False])


def test_refusal_metrics_empty_input():
    assert refusal_metrics([], [])["n"] == 0


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def test_extract_claims_drops_fragments():
    claims = extract_claims("Yes. The controller must notify the authority within 72 hours.")
    assert len(claims) == 1


def test_token_recall_bounds():
    assert token_recall("supervisory authority", CONTEXT[0]) == 1.0
    assert token_recall("parking permits", CONTEXT[0]) == 0.0


def test_citation_metrics_over_records():
    records = [
        {"refused": False, "valid_citations": 2, "invalid_citations": 0},
        {"refused": False, "valid_citations": 1, "invalid_citations": 1},
        {"refused": True, "valid_citations": 0, "invalid_citations": 0},
    ]
    out = citation_metrics(records)
    assert out["citation_coverage"] == 1.0
    assert out["citation_validity"] == pytest.approx(3 / 4)
