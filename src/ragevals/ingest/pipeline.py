"""Corpus -> chunks -> embeddings -> vector store."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..embeddings import get_embedder
from ..logging_utils import get_logger
from ..store import get_store
from .chunking import Chunk, chunk_document
from .pdf import iter_documents, load_document

log = get_logger(__name__)


@dataclass
class IngestReport:
    documents: int = 0
    pages: int = 0
    chunks: int = 0
    seconds: float = 0.0
    chars: int = 0
    per_document: dict = field(default_factory=dict)
    index_dir: str = ""

    @property
    def mean_chunk_chars(self) -> float:
        return self.chars / self.chunks if self.chunks else 0.0

    def summary(self) -> str:
        return (f"{self.documents} document(s), {self.pages} page(s), "
                f"{self.chunks} chunk(s), mean {self.mean_chunk_chars:.0f} chars, "
                f"{self.seconds:.1f}s")


def build_chunks(cfg: Config, corpus_dir: str | Path | None = None) -> tuple[list[Chunk], IngestReport]:
    corpus_dir = Path(corpus_dir or cfg.paths.corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"corpus directory {corpus_dir} does not exist -- run 'make samples' "
            f"or 'python scripts/fetch_corpus.py' first")
    started = time.perf_counter()
    report = IngestReport()
    chunks: list[Chunk] = []
    for path in iter_documents(corpus_dir):
        pages = load_document(
            path,
            **({"layout_aware": cfg.ingest.layout_aware,
                "drop_repeated_headers": cfg.ingest.drop_repeated_headers,
                "heading_scale": cfg.ingest.heading_scale}
               if path.suffix.lower() == ".pdf" else {}),
        )
        doc_chunks = chunk_document(
            pages,
            chunker=cfg.ingest.chunker,
            chunk_size=cfg.ingest.chunk_size,
            chunk_overlap=cfg.ingest.chunk_overlap,
            min_chunk_chars=cfg.ingest.min_chunk_chars,
            source=path.name,
        )
        chunks.extend(doc_chunks)
        report.documents += 1
        report.pages += len(pages)
        report.chunks += len(doc_chunks)
        report.chars += sum(len(c.text) for c in doc_chunks)
        report.per_document[path.name] = {"pages": len(pages), "chunks": len(doc_chunks)}
        log.info("%s -> %d chunk(s) from %d page(s)", path.name, len(doc_chunks), len(pages))
    if not chunks:
        raise RuntimeError(f"no text extracted from {corpus_dir}: is the corpus empty "
                           f"or are the PDFs scanned images (OCR needed)?")
    report.seconds = time.perf_counter() - started
    return chunks, report


def ingest_corpus(cfg: Config, corpus_dir: str | Path | None = None,
                  index_dir: str | Path | None = None) -> IngestReport:
    """Full ingestion: extract, chunk, embed, persist. Idempotent."""
    chunks, report = build_chunks(cfg, corpus_dir)
    index_dir = Path(index_dir or cfg.paths.index_dir) / cfg.fingerprint()
    index_dir.mkdir(parents=True, exist_ok=True)

    embedder = get_embedder(cfg)
    vectors = embedder.encode([c.text for c in chunks])
    store = get_store(cfg, dim=vectors.shape[1], path=index_dir)
    store.reset()
    store.add(chunks, vectors)
    store.persist()

    (index_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks),
        encoding="utf-8")
    manifest = {
        "config_fingerprint": cfg.fingerprint(),
        "config": cfg.to_dict(),
        "embedder": embedder.describe(),
        "store": store.describe(),
        "report": {"documents": report.documents, "pages": report.pages,
                   "chunks": report.chunks, "seconds": round(report.seconds, 3),
                   "mean_chunk_chars": round(report.mean_chunk_chars, 1),
                   "per_document": report.per_document},
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report.index_dir = str(index_dir)
    log.info("index written to %s (%s)", index_dir, report.summary())
    return report


def load_chunks(cfg: Config, index_dir: str | Path | None = None) -> list[Chunk]:
    index_dir = Path(index_dir or cfg.paths.index_dir) / cfg.fingerprint()
    path = index_dir / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"no index for config fingerprint {cfg.fingerprint()} at {path}; "
            f"run: python -m ragevals.cli ingest --config <your.yaml>")
    return [Chunk.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
