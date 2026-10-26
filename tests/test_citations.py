from __future__ import annotations

from ragevals.citations import (
    allowed_citation_keys,
    parse_citations,
    strip_citations,
    validate_citations,
)


def test_parse_standard_form():
    cites = parse_citations("The deadline is 72 hours [gdpr.pdf, p. 3].")
    assert len(cites) == 1
    assert cites[0].source == "gdpr.pdf"
    assert cites[0].page == 3


def test_parse_accepts_variants():
    text = ("[a.pdf, p. 1] [b.pdf, page 2] [c.pdf p.3] [d.pdf, pp. 4] "
            "[e.pdf, p. 5-6]")
    cites = parse_citations(text)
    assert [c.page for c in cites] == [1, 2, 3, 4, 5]
    assert cites[-1].page_end == 6


def test_parse_ignores_non_citations():
    assert parse_citations("See [note] and [2024] and [Annex III].") == []


def test_validate_flags_a_fabricated_page(chunks):
    report = validate_citations("Breaches must be notified [gdpr.pdf, p. 77].", chunks)
    assert report.invalid and not report.valid
    assert not report.all_valid


def test_validate_accepts_a_retrieved_page(chunks):
    report = validate_citations("Breaches must be notified in time [gdpr.pdf, p. 3].",
                                chunks)
    assert report.valid and not report.invalid
    assert report.precision == 1.0


def test_validate_flags_uncited_sentences(chunks):
    answer = ("The controller must notify within 72 hours [gdpr.pdf, p. 3]. "
              "Fines reach four per cent of global turnover.")
    report = validate_citations(answer, chunks, require_per_sentence=True)
    assert len(report.uncited_sentences) == 1


def test_page_ranges_are_expanded(chunks):
    from ragevals.ingest.chunking import Chunk

    spanning = [Chunk(id="x", text="t", source="doc.pdf", page=2, page_end=4)]
    keys = allowed_citation_keys(spanning)
    assert keys == {("doc.pdf", 2), ("doc.pdf", 3), ("doc.pdf", 4)}


def test_strip_citations():
    assert strip_citations("Text here [a.pdf, p. 1].").strip() == "Text here ."


def test_case_insensitive_source_match(chunks):
    report = validate_citations("Notified in time [GDPR.PDF, p. 3].", chunks)
    assert report.valid
