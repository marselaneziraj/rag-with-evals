"""Citation parsing and validation.

The contract the generator must honour is: every factual sentence ends with one
or more citations of the form ``[source.pdf, p. 12]``. That shape is chosen
because it is (a) unambiguous to parse with a regex, (b) checkable against
retrieval metadata, and (c) readable to a human reviewer.

"Hallucinated citation" is a distinct failure from "hallucinated content": a
model can invent a page number while stating a true fact, or cite correctly
while mis-stating the fact. ``validate_citations`` catches the first;
``faithfulness`` in the metrics module catches the second.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .ingest.chunking import Chunk

# [name.pdf, p. 3]  /  [name.pdf, p. 3-4]  /  [name.pdf, page 3]  /  [name.pdf p.3]
CITATION_RE = re.compile(
    r"\[\s*(?P<source>[^\[\],]+?)\s*[,;]?\s*(?:p\.?|pp\.?|page[s]?)\s*"
    r"(?P<page>\d+)\s*(?:[-\u2013]\s*(?P<page_end>\d+))?\s*\]",
    re.IGNORECASE)

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u201c\"(])")


@dataclass(frozen=True)
class Citation:
    source: str
    page: int
    page_end: int | None = None
    raw: str = ""

    def normalised(self) -> tuple[str, int]:
        return (self.source.strip().lower(), self.page)

    def __str__(self) -> str:
        if self.page_end and self.page_end != self.page:
            return f"[{self.source}, p. {self.page}-{self.page_end}]"
        return f"[{self.source}, p. {self.page}]"


@dataclass
class CitationReport:
    citations: list[Citation] = field(default_factory=list)
    valid: list[Citation] = field(default_factory=list)
    invalid: list[Citation] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)

    @property
    def has_citations(self) -> bool:
        return bool(self.citations)

    @property
    def all_valid(self) -> bool:
        return bool(self.citations) and not self.invalid

    @property
    def precision(self) -> float:
        return len(self.valid) / len(self.citations) if self.citations else 0.0


def parse_citations(text: str) -> list[Citation]:
    out: list[Citation] = []
    for m in CITATION_RE.finditer(text or ""):
        page_end = m.group("page_end")
        out.append(Citation(source=m.group("source").strip(),
                            page=int(m.group("page")),
                            page_end=int(page_end) if page_end else None,
                            raw=m.group(0)))
    return out


def allowed_citation_keys(chunks: Sequence[Chunk]) -> set[tuple[str, int]]:
    """Every (source, page) pair the model was actually shown."""
    keys: set[tuple[str, int]] = set()
    for c in chunks:
        for page in range(c.page, max(c.page, c.page_end) + 1):
            keys.add((c.source.strip().lower(), page))
    return keys


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split((text or "").strip()) if s.strip()]


def validate_citations(answer: str, chunks: Sequence[Chunk],
                       require_per_sentence: bool = True) -> CitationReport:
    """Check that every citation resolves to a retrieved (source, page)."""
    allowed = allowed_citation_keys(chunks)
    citations = parse_citations(answer)
    report = CitationReport(citations=citations)
    for cit in citations:
        (report.valid if cit.normalised() in allowed else report.invalid).append(cit)
    if require_per_sentence:
        for sentence in split_sentences(answer):
            if len(sentence) < 25:          # headings, "Yes.", list bullets
                continue
            if not CITATION_RE.search(sentence):
                report.uncited_sentences.append(sentence)
    return report


def strip_citations(text: str) -> str:
    """Remove citation markers -- used before lexical comparison in metrics."""
    return re.sub(r"\s{2,}", " ", CITATION_RE.sub(" ", text or "")).strip()


def format_citation(chunk: Chunk) -> str:
    return chunk.citation


def unique_sources(citations: Iterable[Citation]) -> list[str]:
    seen: list[str] = []
    for c in citations:
        if c.source not in seen:
            seen.append(c.source)
    return seen
