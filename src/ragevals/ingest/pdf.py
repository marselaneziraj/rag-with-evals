"""Layout-aware PDF -> text.

Why not ``page.get_text()`` and be done with it?
------------------------------------------------
Plain text extraction throws away the two signals that make legal/clinical PDFs
navigable: *font metrics* (which line is a heading) and *geometry* (which block
is a running header, which is a footnote). Losing them costs you at retrieval
time, because a chunk whose metadata says ``section: "Article 22 - Automated
individual decision-making"`` is far easier to rank and to cite than an
anonymous 900-character window.

So we read the structured dict from PyMuPDF, keep per-span font size and flags,
and reconstruct:
  * reading order (sort blocks by column, then by vertical position),
  * headings (font noticeably larger than the page's body size, or bold+short),
  * running headers/footers (identical text in the same band on many pages).
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..logging_utils import get_logger

log = get_logger(__name__)

_WS_RE = re.compile(r"[ \t\u00a0]+")
_HYPHEN_RE = re.compile(r"(\w)-\n(\w)")
_NUMBERED_HEADING_RE = re.compile(
    r"^(article|artikull|section|chapter|annex|appendix|recital|part|clause)\s+"
    r"[0-9ivxlc]+", re.IGNORECASE)


@dataclass
class Block:
    """One laid-out text block on a page."""
    text: str
    page: int                 # 1-based, as a human would cite it
    bbox: tuple[float, float, float, float]
    size: float = 0.0         # dominant font size
    bold: bool = False
    is_heading: bool = False

    @property
    def y0(self) -> float:
        return self.bbox[1]


@dataclass
class PageText:
    source: str
    page: int
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


def _clean(text: str) -> str:
    text = text.replace("\r", "")
    text = _HYPHEN_RE.sub(r"\1\2", text)        # de-hyphenate across line breaks
    text = text.replace("\u00ad", "")
    text = _WS_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_like_heading(text: str, size: float, body_size: float, bold: bool,
                        scale: float) -> bool:
    if not text or len(text) > 160:
        return False
    if _NUMBERED_HEADING_RE.match(text.strip()):
        return True
    if body_size and size >= body_size * scale:
        return True
    if bold and len(text) < 90 and not text.rstrip().endswith((".", ";", ":")):
        return True
    if text.isupper() and 3 < len(text) < 90:
        return True
    return False


def _page_blocks(page, page_no: int) -> list[Block]:
    data = page.get_text("dict")
    blocks: list[Block] = []
    for blk in data.get("blocks", []):
        if blk.get("type", 0) != 0:          # 1 == image
            continue
        lines_out: list[str] = []
        sizes: list[float] = []
        bold_votes = 0
        span_count = 0
        for line in blk.get("lines", []):
            parts: list[str] = []
            for span in line.get("spans", []):
                txt = span.get("text", "")
                if not txt.strip():
                    continue
                parts.append(txt)
                sizes.append(float(span.get("size", 0.0)))
                span_count += 1
                flags = int(span.get("flags", 0))
                if flags & 2 ** 4:            # bit 4 == bold in PyMuPDF
                    bold_votes += 1
            if parts:
                lines_out.append("".join(parts))
        if not lines_out:
            continue
        text = _clean("\n".join(lines_out))
        if not text:
            continue
        blocks.append(Block(
            text=text,
            page=page_no,
            bbox=tuple(float(v) for v in blk.get("bbox", (0, 0, 0, 0))),  # type: ignore[arg-type]
            size=statistics.median(sizes) if sizes else 0.0,
            bold=bool(span_count) and bold_votes / max(span_count, 1) > 0.6,
        ))
    # reading order: two-column aware -- left column top-to-bottom, then right
    if blocks:
        page_width = float(page.rect.width) or 1.0
        mid = page_width / 2
        multi_column = (
            sum(1 for b in blocks if b.bbox[0] > mid) >= 3
            and sum(1 for b in blocks if b.bbox[2] < mid) >= 3
        )
        if multi_column:
            blocks.sort(key=lambda b: (0 if b.bbox[0] < mid else 1, round(b.y0, 1)))
        else:
            blocks.sort(key=lambda b: (round(b.y0, 1), b.bbox[0]))
    return blocks


def _strip_running_text(pages: Sequence[PageText], min_pages: int = 3) -> None:
    """Remove headers/footers: short blocks repeated verbatim across pages."""
    if len(pages) < min_pages:
        return
    counts: Counter[str] = Counter()
    for pg in pages:
        for b in pg.blocks:
            if len(b.text) <= 120:
                counts[b.text.strip()] += 1
    threshold = max(min_pages, int(0.6 * len(pages)))
    repeated = {t for t, c in counts.items() if c >= threshold and t}
    if not repeated:
        return
    for pg in pages:
        pg.blocks = [b for b in pg.blocks if b.text.strip() not in repeated]
    log.debug("dropped %d repeated header/footer string(s)", len(repeated))


def extract_pdf(path: str | Path, *, layout_aware: bool = True,
                drop_repeated_headers: bool = True,
                heading_scale: float = 1.12) -> list[PageText]:
    """Extract one PDF into per-page blocks with heading flags."""
    try:
        import pymupdf as fitz
    except ImportError:  # PyMuPDF < 1.24 only exposes the legacy name
        try:
            import fitz  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ImportError(
                "PyMuPDF is required for PDF ingestion: pip install pymupdf"
            ) from exc

    path = Path(path)
    doc = fitz.open(path)
    pages: list[PageText] = []
    try:
        for i, page in enumerate(doc, start=1):
            if layout_aware:
                blocks = _page_blocks(page, i)
            else:
                text = _clean(page.get_text("text"))
                blocks = [Block(text=text, page=i, bbox=(0, 0, 0, 0))] if text else []
            pages.append(PageText(source=path.name, page=i, blocks=blocks))
    finally:
        doc.close()

    if drop_repeated_headers:
        _strip_running_text(pages)

    if layout_aware:
        sizes = [b.size for pg in pages for b in pg.blocks if b.size]
        body = statistics.median(sizes) if sizes else 0.0
        for pg in pages:
            for b in pg.blocks:
                b.is_heading = _looks_like_heading(b.text, b.size, body, b.bold,
                                                   heading_scale)
    log.info("extracted %s: %d page(s), %d block(s)", path.name, len(pages),
             sum(len(p.blocks) for p in pages))
    return pages


def extract_text_file(path: str | Path, chars_per_page: int = 2600) -> list[PageText]:
    """Fallback for .txt/.md sources, paginated so citations stay meaningful.

    A line that is ALL CAPS, or starts with 'Article'/'Section'/'#', is treated
    as a heading, which mirrors what ``extract_pdf`` infers from font size.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    pages: list[PageText] = []
    page_no = 1
    buf: list[Block] = []
    used = 0
    for para in re.split(r"\n\s*\n", raw):
        para = _clean(para)
        if not para:
            continue
        heading = bool(
            _NUMBERED_HEADING_RE.match(para)
            or para.startswith("#")
            or (para.isupper() and len(para) < 90)
        )
        buf.append(Block(text=para.lstrip("# ").strip(), page=page_no,
                         bbox=(0, 0, 0, 0), is_heading=heading))
        used += len(para)
        if used >= chars_per_page:
            pages.append(PageText(source=path.name, page=page_no, blocks=buf))
            page_no += 1
            buf, used = [], 0
    if buf:
        pages.append(PageText(source=path.name, page=page_no, blocks=buf))
    return pages


def load_document(path: str | Path, **kwargs) -> list[PageText]:
    """Dispatch on suffix: PDFs through PyMuPDF, text through the paginator."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, **kwargs)
    if suffix in {".txt", ".md"}:
        return extract_text_file(path)
    raise ValueError(f"unsupported document type: {path}")


def iter_documents(directory: str | Path) -> Iterable[Path]:
    for p in sorted(Path(directory).rglob("*")):
        if p.suffix.lower() in {".pdf", ".txt", ".md"} and p.is_file():
            yield p
