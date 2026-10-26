"""Chroma backend (optional dependency: ``pip install chromadb``)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..ingest.chunking import Chunk
from .base import SearchHit, VectorStore


class ChromaStore(VectorStore):
    name = "chroma"

    def __init__(self, path, dim, collection="ragevals") -> None:
        super().__init__(path, dim, collection)
        import chromadb  # noqa: WPS433
        from chromadb.config import Settings  # noqa: WPS433

        self._client = chromadb.PersistentClient(
            path=str(self.path), settings=Settings(anonymized_telemetry=False))
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=[v.tolist() for v in np.asarray(vectors, dtype=np.float32)],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page, "page_end": c.page_end,
                        "section": c.section, "chunk_index": c.chunk_index}
                       for c in chunks],
        )

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[SearchHit]:
        if self._collection.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32).ravel().tolist()],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        for rank, (cid, doc, meta, dist) in enumerate(
                zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                    res["distances"][0], strict=False), start=1):
            chunk = Chunk(id=cid, text=doc, source=str(meta.get("source", "")),
                          page=int(meta.get("page", 1)),
                          page_end=int(meta.get("page_end", meta.get("page", 1))),
                          section=str(meta.get("section", "")),
                          chunk_index=int(meta.get("chunk_index", 0)))
            hits.append(SearchHit(chunk=chunk, score=1.0 - float(dist), rank=rank,
                                  source_ranker="dense"))
        return hits

    def all_chunks(self) -> list[Chunk]:
        if self._collection.count() == 0:
            return []
        res = self._collection.get(include=["documents", "metadatas"])
        return [Chunk(id=cid, text=doc, source=str(meta.get("source", "")),
                      page=int(meta.get("page", 1)),
                      page_end=int(meta.get("page_end", meta.get("page", 1))),
                      section=str(meta.get("section", "")),
                      chunk_index=int(meta.get("chunk_index", 0)))
                for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=False)]

    def persist(self) -> None:          # PersistentClient writes through
        return None

    def load(self) -> bool:
        return self._collection.count() > 0

    def reset(self) -> None:
        self._client.delete_collection(self.collection)
        self._collection = self._client.get_or_create_collection(
            name=self.collection, metadata={"hnsw:space": "cosine"})
