from __future__ import annotations

import pytest

from ragevals.ingest.chunking import (
    CHUNKERS,
    chunk_document,
    chunk_fixed,
    chunk_section,
    chunk_sentence,
    split_sentences,
)


def test_all_chunkers_registered():
    assert set(CHUNKERS) == {"fixed", "sentence", "section"}


@pytest.mark.parametrize("chunker", sorted(CHUNKERS))
def test_metadata_is_preserved(pages, chunker):
    chunks = chunk_document(pages, chunker=chunker, chunk_size=300, chunk_overlap=60)
    assert chunks
    for chunk in chunks:
        assert chunk.source == "demo.pdf"
        assert 1 <= chunk.page <= 2
        assert chunk.page_end >= chunk.page
        assert chunk.text.strip()
        assert chunk.id


def test_section_chunker_never_crosses_a_heading(pages):
    chunks = chunk_section(pages, source="demo.pdf", chunk_size=4000, chunk_overlap=0)
    assert len(chunks) >= 2, "a huge budget must still split at the heading"
    sections = [c.section for c in chunks]
    assert "Article 33 - Breach notification" in sections
    assert "Article 35 - Impact assessment" in sections
    for chunk in chunks:
        assert not ("72 hours" in chunk.text and "high risk" in chunk.text)


def test_sentence_chunker_does_not_split_sentences(pages):
    chunks = chunk_sentence(pages, source="demo.pdf", chunk_size=200, chunk_overlap=40)
    for chunk in chunks:
        assert not chunk.text.endswith(("whe", "th", "n ,"))
        assert chunk.text == chunk.text.strip()


def test_fixed_chunker_respects_size_and_overlap(pages):
    chunks = chunk_fixed(pages, source="demo.pdf", chunk_size=200, chunk_overlap=50)
    # Every window obeys the budget except the last, which absorbs a remainder
    # too short to stand on its own (see min_chunk_chars in chunking.py).
    assert all(len(c.text) <= 200 for c in chunks[:-1])
    assert len(chunks[-1].text) <= 200 + 120
    assert len(chunks) > 1
    joined = " ".join(c.text for c in chunks)
    assert "72 hours" in joined


def test_overlap_must_be_smaller_than_chunk_size(pages):
    with pytest.raises(ValueError):
        chunk_document(pages, chunk_size=100, chunk_overlap=100)


def test_unknown_chunker_raises(pages):
    with pytest.raises(ValueError, match="unknown chunker"):
        chunk_document(pages, chunker="magic")


def test_empty_input_is_handled():
    assert chunk_document([]) == []


def test_chunk_citation_format(chunks):
    assert chunks[0].citation == "[gdpr.pdf, p. 3]"


def test_split_sentences():
    out = split_sentences("First sentence. Second one? Third! Fourth.")
    assert len(out) == 4
