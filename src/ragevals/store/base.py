"""Vector-store interface shared by the numpy, Chroma and FAISS backends."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..ingest.chunking import Chunk


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    rank: int = 0
    source_ranker: str = ""

    def to_dict(self) -> dict[str, object]:
        return {**self.chunk.to_dict(), "score": round(float(self.score), 6),
                "rank": self.rank, "ranker": self.source_ranker}


class VectorStore:
    name = "base"

    def __init__(self, path: str | Path, dim: int, collection: str = "ragevals") -> None:
        self.path = Path(path)
        self.dim = dim
        self.collection = collection
        self.path.mkdir(parents=True, exist_ok=True)

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[SearchHit]:
        raise NotImplementedError

    def all_chunks(self) -> list[Chunk]:
        raise NotImplementedError

    def persist(self) -> None:
        raise NotImplementedError

    def load(self) -> bool:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.all_chunks())

    def describe(self) -> dict[str, object]:
        return {"backend": self.name, "dim": self.dim, "size": len(self),
                "path": str(self.path)}
