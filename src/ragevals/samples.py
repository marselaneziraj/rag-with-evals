"""Build the demo corpus: structured PDFs from the bundled markdown sources.

The sample PDFs are generated rather than committed as binaries so that the
repository stays text-only and diffable, and so the ingestion tests exercise
the *layout* path: headings are drawn at a larger bold font, body text at 10pt,
and each page carries a running header and a page number -- exactly the
structures ``extract_pdf`` has to recognise and (for the header) discard.
"""
from __future__ import annotations

import re
from pathlib import Path

from .logging_utils import get_logger

log = get_logger(__name__)

PAGE_W, PAGE_H = 595, 842          # A4 in points
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 60, 72, 64
BODY_SIZE, HEAD_SIZE, H1_SIZE = 10.0, 12.5, 15.0
LEADING = 14.0


def _wrap(text: str, font, size: float, width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if font.text_length(trial, fontsize=size) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def build_pdf(markdown_path: str | Path, out_path: str | Path,
              title: str | None = None) -> Path:
    try:
        import pymupdf as fitz
    except ImportError:  # pragma: no cover - legacy package name
        import fitz  # type: ignore[no-redef]

    markdown_path, out_path = Path(markdown_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = title or markdown_path.stem.replace("_", " ").title()
    raw = markdown_path.read_text(encoding="utf-8")

    blocks: list[tuple[str, str]] = []
    for para in re.split(r"\n\s*\n", raw):
        para = para.strip()
        if not para:
            continue
        if para.startswith("# "):
            blocks.append(("h1", para[2:].strip()))
        elif para.startswith("## "):
            blocks.append(("h2", para[3:].strip()))
        else:
            blocks.append(("p", re.sub(r"\s*\n\s*", " ", para)))

    doc = fitz.open()
    regular = fitz.Font("helv")
    bold = fitz.Font("hebo")
    usable = PAGE_W - 2 * MARGIN_X
    page = None
    writer = None
    y = 0.0
    page_no = 0

    def new_page():
        nonlocal page, writer, y, page_no
        if writer is not None and page is not None:
            writer.write_text(page)
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        writer = fitz.TextWriter(page.rect)
        page_no += 1
        writer.append(fitz.Point(MARGIN_X, 46), title, font=regular, fontsize=8)
        writer.append(fitz.Point(PAGE_W - MARGIN_X - 40, PAGE_H - 40),
                      f"Page {page_no}", font=regular, fontsize=8)
        y = MARGIN_TOP

    new_page()
    for kind, text in blocks:
        size = {"h1": H1_SIZE, "h2": HEAD_SIZE, "p": BODY_SIZE}[kind]
        font = bold if kind in {"h1", "h2"} else regular
        lines = _wrap(text, font, size, usable)
        needed = len(lines) * (LEADING if kind == "p" else LEADING + 4) + 6
        if y + needed > PAGE_H - MARGIN_BOTTOM:
            new_page()
        if kind != "p":
            y += 8
        for line in lines:
            writer.append(fitz.Point(MARGIN_X, y), line, font=font, fontsize=size)
            y += LEADING if kind == "p" else LEADING + 4
        y += 6 if kind == "p" else 4

    writer.write_text(page)
    doc.save(out_path, deflate=True)
    doc.close()
    log.info("built %s (%d page(s))", out_path.name, page_no)
    return out_path


def build_sample_corpus(sample_dir: str | Path = "data/samples",
                        corpus_dir: str | Path = "data/corpus") -> list[Path]:
    sample_dir, corpus_dir = Path(sample_dir), Path(corpus_dir)
    sources = sorted(sample_dir.glob("*.md"))
    if not sources:
        raise FileNotFoundError(f"no .md sources in {sample_dir}")
    out: list[Path] = []
    for src in sources:
        out.append(build_pdf(src, corpus_dir / f"{src.stem}.pdf"))
    return out
