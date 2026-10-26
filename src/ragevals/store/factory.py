"""Pick a vector store: configured backend, or the best one installed."""
from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..logging_utils import get_logger
from .base import VectorStore
from .numpy_store import NumpyStore

log = get_logger(__name__)


def get_store(cfg: Config, dim: int, path: str | Path | None = None) -> VectorStore:
    path = Path(path or Path(cfg.paths.index_dir) / cfg.fingerprint())
    backend = cfg.store.backend
    order = [backend] if backend != "auto" else ["chroma", "faiss", "numpy"]
    errors: list[str] = []
    for name in order:
        try:
            if name == "chroma":
                from .chroma_store import ChromaStore
                return ChromaStore(path, dim, cfg.store.collection)
            if name == "faiss":
                from .faiss_store import FaissStore
                return FaissStore(path, dim, cfg.store.collection)
            if name == "numpy":
                return NumpyStore(path, dim, cfg.store.collection)
            raise ValueError(f"unknown store backend {name!r}")
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
            if backend != "auto":
                raise
    log.info("vector store: numpy (optional backends unavailable: %s)", "; ".join(errors))
    return NumpyStore(path, dim, cfg.store.collection)
