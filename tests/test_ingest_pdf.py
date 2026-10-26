"""Ingestion tests against the three real sample PDFs.

They are built on the fly from data/samples/*.md, so the test exercises the
same layout-aware path the pipeline uses -- headings by font size, running
headers dropped, page numbers preserved for citation.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pymupdf", reason="PyMuPDF is required for PDF ingestion")

from ragevals.ingest.pdf import extract_pdf, extract_text_file, iter_documents  # noqa: E402
from ragevals.samples import build_sample_corpus  # noqa: E402


@pytest.fixture(scope="module")
def built_corpus(tmp_path_factory, repo_root):
    out = tmp_path_factory.mktemp("corpus")
    paths = build_sample_corpus(repo_root / "data" / "samples", out)
    return paths


def test_three_sample_pdfs_are_built(built_corpus):
    assert len(built_corpus) == 3
    assert all(p.exists() and p.stat().st_size > 2000 for p in built_corpus)


def test_pages_and_text_are_extracted(built_corpus):
    pages = extract_pdf(built_corpus[0])
    assert len(pages) >= 2
    assert all(p.page == i for i, p in enumerate(pages, start=1))
    assert sum(len(p.text) for p in pages) > 3000


def test_headings_are_detected(built_corpus):
    pages = extract_pdf(built_corpus[0])
    headings = [b.text for p in pages for b in p.blocks if b.is_heading]
    assert headings, "layout-aware extraction must recover at least one heading"
    assert any(h.lower().startswith(("article", "section", "eu ai act"))
               for h in headings)


def test_running_headers_are_dropped(built_corpus):
    """Every page carries the document title as a running header; it must not
    reappear in the body of every page."""
    pages = extract_pdf(built_corpus[0], drop_repeated_headers=True)
    kept = extract_pdf(built_corpus[0], drop_repeated_headers=False)
    assert sum(len(p.blocks) for p in pages) < sum(len(p.blocks) for p in kept)


def test_page_numbers_support_citation(built_corpus):
    pages = extract_pdf(built_corpus[2])
    for page in pages:
        for block in page.blocks:
            assert block.page == page.page


def test_layout_blind_mode_still_works(built_corpus):
    pages = extract_pdf(built_corpus[1], layout_aware=False)
    assert pages and any(p.text for p in pages)


def test_text_fallback_paginates(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("\n\n".join(["ARTICLE 1"] + ["paragraph " * 80] * 6),
                    encoding="utf-8")
    pages = extract_text_file(path, chars_per_page=1200)
    assert len(pages) >= 2
    assert any(b.is_heading for p in pages for b in p.blocks)


def test_iter_documents_filters_by_suffix(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"\x89PNG")
    assert {p.name for p in iter_documents(tmp_path)} == {"a.pdf", "b.txt"}


def test_unsupported_type_raises(tmp_path):
    from ragevals.ingest.pdf import load_document

    bad = tmp_path / "x.docx"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        load_document(bad)
