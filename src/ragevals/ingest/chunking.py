"""Chunking strategies.

The chunker is the single most consequential knob in a RAG system and the one
most often chosen by vibes, so all three strategies here are swappable from the
config and compared head-to-head by the experiment runner.

fixed     character window with overlap. Cheap, layout-blind, splits mid-clause.
sentence  pack whole sentences up to the budget. Never splits a sentence, so a
          retrieved chunk is always readable; boundaries still ignore structure.
section   pack sentences but *never cross a heading*, and stamp the heading onto
          every chunk. For legislation and guidelines this is usually the best
          trade-off: an article is a semantic unit, and the heading text is a
          free, high-precision keyword signal for the lexical retriever.

Overlap exists so a fact that straddles a boundary survives in at least one
chunk. It costs index size and introduces near-duplicates in the ranking, which
is why ``context_precision`` is measured per-configuration rather than assumed.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .pdf import PageText

_SENT_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9(\u201c\"'])|\n+")


@dataclass
class Chunk:
    """A retrievable unit plus everything needed to cite it."""
    id: str
    text: str
    source: str
    page: int
    page_end: int
    section: str = ""
    chunk_index: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        if self.page_end != self.page:
            return f"[{self.source}, p. {self.page}-{self.page_end}]"
        return f"[{self.source}, p. {self.page}]"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "source": self.source,
                "page": self.page, "page_end": self.page_end,
                "section": self.section, "chunk_index": self.chunk_index,
                "extra": self.extra}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(id=d["id"], text=d["text"], source=d["source"], page=int(d["page"]),
                   page_end=int(d.get("page_end", d["page"])),
                   section=d.get("section", ""), chunk_index=int(d.get("chunk_index", 0)),
                   extra=d.get("extra", {}) or {})


def split_sentences(text: str) -> list[str]:
    """Deliberately simple and dependency-free; good enough for prose and law."""
    parts = [p.strip() for p in _SENT_RE.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}|{index}|{text}".encode()).hexdigest()[:16]
    return f"{Path_stem(source)}-{index:04d}-{digest}"


def Path_stem(source: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", source.rsplit(".", 1)[0])[:40]


@dataclass
class _Unit:
    """A sentence (or block) with the page and section it came from."""
    text: str
    page: int
    section: str


def _units(pages: Sequence[PageText], *, respect_sections: bool) -> list[_Unit]:
    units: list[_Unit] = []
    section = ""
    for pg in pages:
        for blk in pg.blocks:
            if blk.is_heading:
                section = blk.text.strip()
                if respect_sections:
                    units.append(_Unit(text="\u0000HEADING\u0000" + blk.text,
                                       page=blk.page, section=section))
                    continue
            for sent in split_sentences(blk.text):
                units.append(_Unit(text=sent, page=blk.page, section=section))
    return units


def _emit(buf: list[_Unit], source: str, index: int, min_chars: int,
          out: list[Chunk]) -> int:
    text = " ".join(u.text for u in buf).strip()
    if len(text) < min_chars and out:
        # too small to stand alone: glue onto the previous chunk
        prev = out[-1]
        prev.text = (prev.text + " " + text).strip()
        prev.page_end = max(prev.page_end, buf[-1].page)
        return index
    if not text:
        return index
    out.append(Chunk(
        id=_chunk_id(source, index, text),
        text=text,
        source=source,
        page=buf[0].page,
        page_end=buf[-1].page,
        section=buf[0].section,
        chunk_index=index,
    ))
    return index + 1


def chunk_fixed(pages: Sequence[PageText], *, source: str, chunk_size: int,
                chunk_overlap: int, min_chunk_chars: int = 120) -> list[Chunk]:
    """Character windows over the page stream, with page bookkeeping."""
    spans: list[tuple[int, int, str, str]] = []   # (start, end, page, section)
    buf: list[str] = []
    cursor = 0
    section = ""
    for pg in pages:
        for blk in pg.blocks:
            if blk.is_heading:
                section = blk.text.strip()
            piece = blk.text.strip()
            if not piece:
                continue
            buf.append(piece)
            spans.append((cursor, cursor + len(piece), str(blk.page), section))
            cursor += len(piece) + 1
    text = " ".join(buf)
    if not text:
        return []

    def locate(pos: int) -> tuple[int, str]:
        for start, end, page, sec in spans:
            if start <= pos < end:
                return int(page), sec
        return int(spans[-1][2]), spans[-1][3]

    step = max(1, chunk_size - chunk_overlap)
    out: list[Chunk] = []
    index = 0
    for start in range(0, len(text), step):
        window = text[start:start + chunk_size]
        if not window.strip():
            continue
        if len(window) < min_chunk_chars and out:
            out[-1].text += " " + window.strip()
            continue
        page, sec = locate(start)
        page_end, _ = locate(min(start + len(window) - 1, len(text) - 1))
        out.append(Chunk(id=_chunk_id(source, index, window), text=window.strip(),
                         source=source, page=page, page_end=max(page, page_end),
                         section=sec, chunk_index=index))
        index += 1
        if start + chunk_size >= len(text):
            break
    return out


def _pack(units: Sequence[_Unit], *, source: str, chunk_size: int,
          chunk_overlap: int, min_chunk_chars: int,
          break_on_heading: bool) -> list[Chunk]:
    out: list[Chunk] = []
    buf: list[_Unit] = []
    size = 0
    index = 0
    for unit in units:
        is_heading = unit.text.startswith("\u0000HEADING\u0000")
        clean = unit.text.replace("\u0000HEADING\u0000", "")
        u = _Unit(text=clean, page=unit.page, section=unit.section)
        if is_heading and break_on_heading and buf:
            index = _emit(buf, source, index, min_chunk_chars, out)
            buf, size = [], 0
        if size + len(clean) > chunk_size and buf:
            index = _emit(buf, source, index, min_chunk_chars, out)
            # carry the tail forward as overlap
            tail: list[_Unit] = []
            carried = 0
            for prev in reversed(buf):
                if carried >= chunk_overlap:
                    break
                tail.insert(0, prev)
                carried += len(prev.text)
            buf = list(tail)
            size = sum(len(x.text) for x in buf)
        buf.append(u)
        size += len(clean)
    if buf:
        _emit(buf, source, index, min_chunk_chars, out)
    return out


def chunk_sentence(pages: Sequence[PageText], *, source: str, chunk_size: int,
                   chunk_overlap: int, min_chunk_chars: int = 120) -> list[Chunk]:
    return _pack(_units(pages, respect_sections=False), source=source,
                 chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                 min_chunk_chars=min_chunk_chars, break_on_heading=False)


def chunk_section(pages: Sequence[PageText], *, source: str, chunk_size: int,
                  chunk_overlap: int, min_chunk_chars: int = 120) -> list[Chunk]:
    return _pack(_units(pages, respect_sections=True), source=source,
                 chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                 min_chunk_chars=min_chunk_chars, break_on_heading=True)


CHUNKERS: dict[str, Callable[..., list[Chunk]]] = {
    "fixed": chunk_fixed,
    "sentence": chunk_sentence,
    "section": chunk_section,
}


def chunk_document(pages: Sequence[PageText], *, chunker: str = "section",
                   chunk_size: int = 900, chunk_overlap: int = 150,
                   min_chunk_chars: int = 120, source: str | None = None) -> list[Chunk]:
    if chunker not in CHUNKERS:
        raise ValueError(f"unknown chunker {chunker!r}; choose from {sorted(CHUNKERS)}")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not pages:
        return []
    source = source or pages[0].source
    return CHUNKERS[chunker](pages, source=source, chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap,
                             min_chunk_chars=min_chunk_chars)
