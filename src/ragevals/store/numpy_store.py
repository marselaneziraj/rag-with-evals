"""Flat numpy store: exact cosine search, zero extra dependencies.

A corpus of a few hundred pages is tens of thousands of vectors at most; an
exact brute-force scan over that is sub-millisecond and removes an entire class
of "is my ANN index lying to me?" questions from the evaluation. Chroma and
FAISS are available for larger corpora and are drop-in via the same interface.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..ingest.chunking import Chunk
from .base import SearchHit, VectorStore


class NumpyStore(VectorStore):
    name = "numpy"

    def __init__(self, path, dim, collection="ragevals") -> None:
        super().__init__(path, dim, collection)
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._chunks: list[Chunk] = []

    @property
    def _vec_path(self) -> Path:
        return self.path / f"{self.collection}.vectors.npy"

    @property
    def _meta_path(self) -> Path:
        return self.path / f"{self.collection}.meta.jsonl"

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[0] != len(chunks):
            raise ValueError("vectors and chunks must align")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        self._vectors = np.vstack([self._vectors, vectors])
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[SearchHit]:
        if not self._chunks:
            return []
        q = np.asarray(query_vector, dtype=np.float32).ravel()
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = self._vectors @ q
        k = min(k, len(self._chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [SearchHit(chunk=self._chunks[int(i)], score=float(scores[int(i)]),
                          rank=rank, source_ranker="dense")
                for rank, i in enumerate(top, start=1)]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def persist(self) -> None:
        np.save(self._vec_path, self._vectors)
        self._meta_path.write_text(
            "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in self._chunks),
            encoding="utf-8")

    def load(self) -> bool:
        if not (self._vec_path.exists() and self._meta_path.exists()):
            return False
        self._vectors = np.load(self._vec_path).astype(np.float32)
        self._chunks = [Chunk.from_dict(json.loads(line))
                        for line in self._meta_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()]
        self.dim = self._vectors.shape[1] if self._vectors.size else self.dim
        return True

    def reset(self) -> None:
        self._vectors = np.zeros((0, self.dim), dtype=np.float32)
        self._chunks = []
        for p in (self._vec_path, self._meta_path):
            if p.exists():
                p.unlink()
