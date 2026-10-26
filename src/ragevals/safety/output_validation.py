"""Output-side mitigation: refuse to ship an answer we cannot back.

Checks, each of which maps to a real failure mode
-------------------------------------------------
``missing_citation``   a factual sentence with no ``[source, p. N]``. An
                       injected payload is usually emitted bare, so this is the
                       single most effective output-side filter -- and it is
                       also what keeps honest answers auditable.
``invalid_citation``   a citation pointing at a (source, page) that was never
                       retrieved: a fabricated reference.
``unsupported``        a sentence whose content words do not appear in any
                       retrieved chunk. Lexical, therefore conservative: it
                       catches invention, not paraphrase.
``prompt_leak``        an 8-gram shared with the system prompt.
``instruction_echo``   the answer itself reads as an imperative addressed to a
                       model, i.e. the attacker's text came back out.

Deliberate omission: this module does **not** look for the injection suite's
canary tokens. A defence that greps for the exact string the evaluation uses to
score success would inflate the mitigation result and generalise to nothing.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..citations import CITATION_RE, split_sentences, validate_citations
from ..embeddings import tokenize
from ..generation.prompts import REFUSAL_MARKER
from ..ingest.chunking import Chunk
from .sanitize import INJECTION_PATTERNS, detection_view

_STOP: set[str] = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "that",
    "this", "is", "are", "be", "as", "by", "it", "its", "at", "from", "not",
    "must", "shall", "may", "can", "which", "who", "any", "all", "if", "then",
}


@dataclass
class ValidationResult:
    ok: bool
    answer: str
    reasons: list[str] = field(default_factory=list)
    unsupported_sentences: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    support_scores: list[float] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.answer.strip().startswith(REFUSAL_MARKER)


def _content(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOP and len(t) > 2}


def support_score(sentence: str, context_tokens: set[str]) -> float:
    """Fraction of a sentence's content words present in the retrieved context."""
    tokens = _content(CITATION_RE.sub(" ", sentence))
    if not tokens:
        return 1.0
    return len(tokens & context_tokens) / len(tokens)


def validate_answer(answer: str, chunks: Sequence[Chunk], *,
                    system_prompt: str = "", support_threshold: float = 0.25,
                    require_citations: bool = True) -> ValidationResult:
    answer = (answer or "").strip()
    if answer.startswith(REFUSAL_MARKER):
        return ValidationResult(ok=True, answer=REFUSAL_MARKER)
    if not answer:
        return ValidationResult(ok=False, answer=REFUSAL_MARKER, reasons=["empty_answer"])

    reasons: list[str] = []
    context_tokens: set[str] = set()
    for chunk in chunks:
        context_tokens |= _content(chunk.text)

    report = validate_citations(answer, chunks, require_per_sentence=require_citations)
    if require_citations and not report.has_citations:
        reasons.append("missing_citation")
    if report.invalid:
        reasons.append("invalid_citation")
    if require_citations and report.uncited_sentences:
        reasons.append("uncited_sentence")

    unsupported: list[str] = []
    scores: list[float] = []
    for sentence in split_sentences(answer):
        score = support_score(sentence, context_tokens)
        scores.append(score)
        if score < support_threshold:
            unsupported.append(sentence)
    if unsupported and len(unsupported) >= max(1, len(scores) // 2):
        reasons.append("unsupported")

    normalised = detection_view(answer)
    if any(pattern.search(normalised) for name, pattern in INJECTION_PATTERNS
           if name in {"override", "new_directive", "role_play", "leak_prompt"}):
        reasons.append("instruction_echo")

    if system_prompt:
        sys_tokens = tokenize(system_prompt)
        grams = {" ".join(sys_tokens[i:i + 8]) for i in range(max(0, len(sys_tokens) - 7))}
        ans_tokens = tokenize(answer)
        ans_grams = {" ".join(ans_tokens[i:i + 8]) for i in range(max(0, len(ans_tokens) - 7))}
        if grams & ans_grams:
            reasons.append("prompt_leak")

    ok = not reasons
    return ValidationResult(
        ok=ok,
        answer=answer if ok else REFUSAL_MARKER,
        reasons=reasons,
        unsupported_sentences=unsupported,
        invalid_citations=[str(c) for c in report.invalid],
        support_scores=scores,
    )
