"""FAISS backend (optional dependency: ``pip install faiss-cpu``).

Inner product on L2-normalised vectors == cosine similarity, so an IndexFlatIP
gives exactly the same ranking as the numpy store while scaling further.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..ingest.chunking import Chunk
from .base import SearchHit, VectorStore


class FaissStore(VectorStore):
    name = "faiss"

    def __init__(self, path, dim, collection="ragevals") -> None:
        super().__init__(path, dim, collection)
        import faiss  # noqa: WPS433

        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    @property
    def _index_path(self) -> Path:
        return self.path / f"{self.collection}.faiss"

    @property
    def _meta_path(self) -> Path:
        return self.path / f"{self.collection}.meta.jsonl"

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self._faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[SearchHit]:
        if not self._chunks:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1).copy()
        self._faiss.normalize_L2(q)
        scores, idx = self._index.search(q, min(k, len(self._chunks)))
        return [SearchHit(chunk=self._chunks[int(i)], score=float(s), rank=rank,
                          source_ranker="dense")
                for rank, (s, i) in enumerate(zip(scores[0], idx[0], strict=False), start=1) if i >= 0]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def persist(self) -> None:
        self._faiss.write_index(self._index, str(self._index_path))
        self._meta_path.write_text(
            "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in self._chunks),
            encoding="utf-8")

    def load(self) -> bool:
        if not (self._index_path.exists() and self._meta_path.exists()):
            return False
        self._index = self._faiss.read_index(str(self._index_path))
        self._chunks = [Chunk.from_dict(json.loads(line))
                        for line in self._meta_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()]
        return True

    def reset(self) -> None:
        self._index = self._faiss.IndexFlatIP(self.dim)
        self._chunks = []
        for p in (self._index_path, self._meta_path):
            if p.exists():
                p.unlink()
