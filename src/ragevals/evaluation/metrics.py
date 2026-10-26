r"""Evaluation metrics -- the part of this repo that is actually the product.

Read this file before defending the methodology. Every metric below answers a
*different* question, and the reason there are six of them is that a single
"accuracy" number cannot tell you which component to fix.

    retrieval broken?      -> context_recall, context_precision
    generator inventing?   -> faithfulness
    generator unhelpful?   -> answer_correctness
    system over/under      -> refusal_accuracy (and over_refusal_rate)
      -confident?
    system shippable?      -> p95 latency, cost per query

Definitions
-----------
**faithfulness** = (claims in the answer that the retrieved context supports) /
(claims in the answer). This is *groundedness*, and it is deliberately blind to
whether the answer is correct: an answer can be perfectly faithful to a
retrieved passage that does not answer the question. Faithfulness is the metric
that catches hallucination, and the only one you can compute without a gold
answer -- so it is the one you can also run in production on live traffic.

**context_precision@k** = rank-weighted share of retrieved chunks that are
relevant: mean over relevant positions of precision@i. Rank weighting is the
point. Two configurations can retrieve the same 2 relevant chunks out of 5, but
the one that puts them at ranks 1-2 feeds a much better prompt than the one
that puts them at ranks 4-5, because the generator attends unevenly and the
irrelevant text dilutes the window. Plain precision cannot see that difference.

**context_recall** = share of the ground-truth answer's claims that are
attributable to the retrieved context. This is the ceiling on everything
downstream: a claim that was never retrieved cannot be generated faithfully.
When correctness is low, context_recall tells you whether to fix the retriever
or the generator -- high recall + low correctness means the generator is
wasting good context.

**answer_correctness** = agreement with a human-written reference, scored 0-4
by an LLM judge against a fixed rubric at temperature 0, normalised to [0, 1].
With no judge available it falls back to a lexical F1 blended with key-fact
recall (numbers, article references, quoted phrases) -- weaker on paraphrase,
but deterministic, free, and good enough to rank configurations.

**refusal_accuracy** = share of items where the refusal decision was right,
over the whole gold set (answerable *and* unanswerable). Reported alongside
``over_refusal_rate`` (refusing an answerable question) and ``hallucination
_rate_on_unanswerable`` (answering an unanswerable one) because the two errors
have wildly different costs and a single accuracy number hides the trade-off.
This is why a quarter of the gold set is unanswerable by construction: without
negatives, a system that never refuses looks perfect.

Implementation stance
---------------------
Every function here is pure, deterministic, dependency-light and unit-tested.
The LLM judge is an *optional upgrade* injected as a callable, never a hard
dependency -- so the metric suite itself is never the flaky part of CI.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..citations import split_sentences, strip_citations
from ..embeddings import tokenize

STOPWORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "do", "does", "for", "from", "had", "has", "have", "how", "i",
    "if", "in", "into", "is", "it", "its", "may", "might", "must", "no", "not",
    "of", "on", "only", "or", "other", "our", "shall", "should", "so", "some",
    "such", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "to", "under", "up", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "why", "will", "with", "would",
    "you", "your", "about", "all", "also", "any", "each", "more", "most",
}

_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_REFERENCE_RE = re.compile(r"\b(?:article|art\.|section|annex|recital|chapter)\s*"
                           r"\d+[a-z]?(?:\(\d+\))?", re.IGNORECASE)
_QUOTED_RE = re.compile(r"[\"\u201c]([^\"\u201d]{3,80})[\"\u201d]")


@dataclass
class MetricResult:
    """A score plus the evidence behind it -- never just a float."""
    value: float
    detail: dict[str, object] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.value)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(strip_citations(text)) if t not in STOPWORDS and len(t) > 1]


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-content-words F1 -- the SQuAD-style primitive.

    Precision punishes padding, recall punishes omission; F1 is the harmonic
    mean, so an answer cannot win by being long *or* by being terse.
    """
    pred = content_tokens(prediction)
    ref = content_tokens(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    pred_counts: dict[str, int] = {}
    for t in pred:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    overlap = 0
    for t in ref:
        if pred_counts.get(t, 0) > 0:
            pred_counts[t] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def token_recall(text: str, container: str) -> float:
    """Share of ``text``'s content words that occur in ``container``."""
    toks = content_tokens(text)
    if not toks:
        return 1.0
    pool = set(content_tokens(container))
    return sum(1 for t in toks if t in pool) / len(toks)


def key_facts(text: str) -> set[str]:
    """Numbers, legal references and quoted phrases: the parts you cannot paraphrase."""
    facts = {m.group(0).lower() for m in _NUMBER_RE.finditer(text or "")}
    facts |= {re.sub(r"\s+", " ", m.group(0).lower())
              for m in _REFERENCE_RE.finditer(text or "")}
    facts |= {m.group(1).strip().lower() for m in _QUOTED_RE.finditer(text or "")}
    return facts


def key_fact_recall(prediction: str, reference: str) -> float:
    ref_facts = key_facts(reference)
    if not ref_facts:
        return 1.0
    pred_blob = re.sub(r"\s+", " ", (prediction or "").lower())
    return sum(1 for f in ref_facts if f in pred_blob) / len(ref_facts)


REFUSAL_MARKER = "INSUFFICIENT_CONTEXT"


def extract_claims(answer: str) -> list[str]:
    """Split an answer into checkable claims (one sentence == one claim).

    Sentence-level claims are coarser than the proposition-level decomposition
    RAGAS performs with an LLM, and that is a deliberate trade: it is free,
    deterministic, and errs towards *harsher* scores on compound sentences,
    because a sentence counts as supported only if all of it is.
    """
    if (answer or "").strip().startswith(REFUSAL_MARKER):
        return []            # a refusal asserts nothing
    claims: list[str] = []
    for sentence in split_sentences(strip_citations(answer)):
        sentence = sentence.strip()
        if len(sentence) < 12:
            continue
        claims.append(sentence)
    return claims


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def faithfulness(answer: str, contexts: Sequence[str], *, threshold: float = 0.55,
                 judge: Callable[[str, str], float] | None = None) -> MetricResult:
    """Share of answer claims supported by the retrieved context."""
    claims = extract_claims(answer)
    if not claims:
        return MetricResult(1.0, {"claims": 0, "note": "no claims (refusal or empty)"})
    blob = "\n".join(contexts)
    supported: list[tuple[str, float]] = []
    for claim in claims:
        if judge is not None:
            score = float(judge(claim, blob))
        else:
            score = max((token_recall(claim, ctx) for ctx in contexts), default=0.0)
            score = max(score, token_recall(claim, blob))
        supported.append((claim, score))
    hits = sum(1 for _, s in supported if s >= threshold)
    return MetricResult(hits / len(claims), {
        "claims": len(claims), "supported": hits, "threshold": threshold,
        "per_claim": [{"claim": c[:160], "support": round(s, 3)} for c, s in supported],
    })


def _relevance_flags(hit_texts: Sequence[str], reference: str,
                     gold_keys: set[tuple[str, int]] | None = None,
                     hit_keys: Sequence[tuple[str, int]] | None = None,
                     threshold: float = 0.30) -> list[bool]:
    """Relevance per retrieved chunk.

    Preferred: exact match against human-annotated gold (source, page) pairs.
    Fallback: the chunk covers at least ``threshold`` of the reference answer's
    content words. The fallback is noisier, which is exactly why the gold-set
    schema has a ``gold_sources`` field and the curation script pre-fills it.
    """
    if gold_keys and hit_keys:
        return [key in gold_keys for key in hit_keys]
    return [token_recall(reference, text) >= threshold for text in hit_texts]


def context_precision(hit_texts: Sequence[str], reference: str, *,
                      gold_keys: set[tuple[str, int]] | None = None,
                      hit_keys: Sequence[tuple[str, int]] | None = None,
                      threshold: float = 0.30) -> MetricResult:
    """Rank-weighted precision@k (average precision over the retrieved list)."""
    if not hit_texts:
        return MetricResult(0.0, {"k": 0, "relevant": 0})
    flags = _relevance_flags(hit_texts, reference, gold_keys, hit_keys, threshold)
    relevant = sum(flags)
    if relevant == 0:
        return MetricResult(0.0, {"k": len(flags), "relevant": 0, "flags": flags})
    running = 0
    total = 0.0
    for i, flag in enumerate(flags, start=1):
        if flag:
            running += 1
            total += running / i          # precision@i at each relevant position
    return MetricResult(total / relevant,
                        {"k": len(flags), "relevant": relevant, "flags": flags})


def context_recall(reference: str, contexts: Sequence[str], *,
                   threshold: float = 0.55,
                   gold_keys: set[tuple[str, int]] | None = None,
                   hit_keys: Sequence[tuple[str, int]] | None = None) -> MetricResult:
    """Share of the reference answer that the retrieved context can support."""
    if gold_keys:
        if not gold_keys:
            return MetricResult(1.0, {"note": "no gold keys"})
        found = len(gold_keys & set(hit_keys or []))
        return MetricResult(found / len(gold_keys),
                            {"gold_chunks": len(gold_keys), "found": found,
                             "mode": "annotated"})
    claims = extract_claims(reference)
    if not claims:
        return MetricResult(1.0, {"claims": 0, "mode": "lexical"})
    blob = "\n".join(contexts)
    scores = [max(max((token_recall(c, ctx) for ctx in contexts), default=0.0),
                  token_recall(c, blob)) for c in claims]
    hits = sum(1 for s in scores if s >= threshold)
    return MetricResult(hits / len(claims),
                        {"claims": len(claims), "attributable": hits,
                         "threshold": threshold, "mode": "lexical"})


def answer_correctness_lexical(answer: str, reference: str) -> MetricResult:
    """Deterministic stand-in for the LLM judge.

    0.6 * token F1 + 0.4 * key-fact recall. The second term exists because in
    this domain the facts that matter are numbers and article references, and a
    bag-of-words F1 scores "72 hours" and "27 hours" almost identically.
    """
    f1 = token_f1(answer, reference)
    facts = key_fact_recall(answer, reference)
    return MetricResult(0.6 * f1 + 0.4 * facts,
                        {"token_f1": round(f1, 4), "key_fact_recall": round(facts, 4),
                         "mode": "lexical"})


def refusal_metrics(expected: Sequence[bool], observed: Sequence[bool]) -> dict[str, float]:
    """Confusion matrix over the refusal decision.

    expected[i] is True when the gold set marks the item unanswerable.
    """
    if len(expected) != len(observed):
        raise ValueError("expected and observed must have the same length")
    n = len(expected)
    if n == 0:
        return {"refusal_accuracy": 0.0, "refusal_precision": 0.0, "refusal_recall": 0.0,
                "over_refusal_rate": 0.0, "hallucination_rate_on_unanswerable": 0.0,
                "n": 0}
    tp = sum(1 for e, o in zip(expected, observed, strict=False) if e and o)
    fp = sum(1 for e, o in zip(expected, observed, strict=False) if not e and o)
    fn = sum(1 for e, o in zip(expected, observed, strict=False) if e and not o)
    tn = sum(1 for e, o in zip(expected, observed, strict=False) if not e and not o)
    answerable = tn + fp
    unanswerable = tp + fn
    return {
        "refusal_accuracy": (tp + tn) / n,
        "refusal_precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "refusal_recall": tp / unanswerable if unanswerable else 0.0,
        "over_refusal_rate": fp / answerable if answerable else 0.0,
        "hallucination_rate_on_unanswerable": fn / unanswerable if unanswerable else 0.0,
        "n": float(n), "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
    }


def citation_metrics(records: Sequence[dict[str, object]]) -> dict[str, float]:
    """How well the citation contract held across a run."""
    answered = [r for r in records if not r.get("refused")]
    if not answered:
        return {"citation_coverage": 0.0, "citation_validity": 0.0, "mean_citations": 0.0}
    with_cites = sum(1 for r in answered if int(r.get("valid_citations", 0))
                     + int(r.get("invalid_citations", 0)) > 0)
    valid = sum(int(r.get("valid_citations", 0)) for r in answered)
    total = valid + sum(int(r.get("invalid_citations", 0)) for r in answered)
    return {
        "citation_coverage": with_cites / len(answered),
        "citation_validity": (valid / total) if total else 0.0,
        "mean_citations": total / len(answered),
    }


def aggregate(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0
