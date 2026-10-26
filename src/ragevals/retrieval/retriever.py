"""The retrieval layer: dense, BM25, or both fused with RRF.

Why reciprocal rank fusion rather than score addition?
------------------------------------------------------
Cosine similarity and BM25 live on incomparable scales -- cosine is bounded in
[-1, 1] and tightly clustered, BM25 is unbounded and corpus-dependent. Adding
them (even after min-max normalisation) makes the mixture weight depend on the
score *distribution* of the particular query, which is exactly the instability
you do not want in an evaluation. RRF throws away magnitudes and keeps only
ranks:

    score(d) = sum_r  w_r / (K + rank_r(d))

with K = 60 by default (Cormack et al., 2009). It is scale-free, needs no
tuning, and degrades gracefully when one retriever returns nothing useful.
``dense_weight`` / ``bm25_weight`` still let you lean on one side.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..embeddings import Embedder, get_embedder
from ..ingest.chunking import Chunk
from ..logging_utils import get_logger
from ..store.base import SearchHit, VectorStore
from ..store.factory import get_store
from .bm25 import BM25Index
from .cache import RetrievalCache
from .rerank import Reranker, build_reranker

log = get_logger(__name__)


@dataclass
class RetrievalResult:
    query: str
    hits: list[SearchHit] = field(default_factory=list)
    latency_s: float = 0.0
    cached: bool = False

    @property
    def contexts(self) -> list[str]:
        return [h.chunk.text for h in self.hits]

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0


def corpus_fingerprint(chunks: Sequence[Chunk]) -> str:
    """Short stable hash of the indexed chunk ids."""
    digest = hashlib.sha1()
    for cid in sorted(c.id for c in chunks):
        digest.update(cid.encode("utf-8"))
    return digest.hexdigest()[:10]


def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[SearchHit]],
                           weights: Sequence[float], rrf_k: int = 60) -> list[SearchHit]:
    fused: dict[str, float] = {}
    best: dict[str, SearchHit] = {}
    rankers: dict[str, set] = {}
    for hits, weight in zip(ranked_lists, weights, strict=False):
        for rank, hit in enumerate(hits, start=1):
            cid = hit.chunk.id
            fused[cid] = fused.get(cid, 0.0) + weight / (rrf_k + rank)
            rankers.setdefault(cid, set()).add(hit.source_ranker or "?")
            if cid not in best or hit.rank < best[cid].rank:
                best[cid] = hit
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    out: list[SearchHit] = []
    for rank, (cid, score) in enumerate(ordered, start=1):
        hit = best[cid]
        out.append(SearchHit(chunk=hit.chunk, score=float(score), rank=rank,
                             source_ranker="+".join(sorted(rankers[cid]))))
    return out


class Retriever:
    """Owns the index, the lexical index, the reranker and the cache."""

    def __init__(self, cfg: Config, chunks: Sequence[Chunk], store: VectorStore,
                 embedder: Embedder, reranker: Reranker | None = None,
                 cache: RetrievalCache | None = None) -> None:
        self.cfg = cfg
        self.chunks = list(chunks)
        self.by_id = {c.id: c for c in self.chunks}
        self.store = store
        self.embedder = embedder
        self.reranker = reranker or build_reranker(cfg.retrieval.reranker,
                                                   cfg.retrieval.rerank_model)
        self.bm25 = (BM25Index(self.chunks)
                     if cfg.retrieval.mode in {"bm25", "hybrid"} else None)
        # The cache namespace must bind the *corpus* as well as the config: two
        # runs can share a fingerprint and still index different documents (this
        # is exactly what the injection suite does when it poisons the corpus),
        # and a cache that ignored that would serve answers from the wrong index.
        self.corpus_hash = corpus_fingerprint(self.chunks)
        self.cache = cache or RetrievalCache(
            cfg.paths.cache_dir,
            namespace=f"retrieval-{cfg.fingerprint()}-{self.corpus_hash}",
            enabled=cfg.retrieval.cache)

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or self.cfg.retrieval.top_k
        started = time.perf_counter()
        cached = self.cache.get(query)
        if cached is not None:
            hits = [SearchHit(chunk=self.by_id[d["id"]] if d["id"] in self.by_id
                              else Chunk.from_dict(d),
                              score=float(d["score"]), rank=int(d["rank"]),
                              source_ranker=str(d.get("ranker", "")))
                    for d in cached][:top_k]
            return RetrievalResult(query=query, hits=hits,
                                   latency_s=time.perf_counter() - started, cached=True)

        candidate_k = max(self.cfg.retrieval.candidate_k, top_k)
        mode = self.cfg.retrieval.mode
        lists: list[list[SearchHit]] = []
        weights: list[float] = []

        if mode in {"dense", "hybrid"}:
            qvec = self.embedder.encode_one(query)
            dense_hits = self.store.search(qvec, k=candidate_k)
            lists.append(dense_hits)
            weights.append(self.cfg.retrieval.dense_weight if mode == "hybrid" else 1.0)
        if mode in {"bm25", "hybrid"} and self.bm25 is not None:
            lexical_hits = self.bm25.search(query, k=candidate_k)
            lists.append(lexical_hits)
            weights.append(self.cfg.retrieval.bm25_weight if mode == "hybrid" else 1.0)
        if not lists:
            raise ValueError(f"unknown retrieval mode {mode!r}")

        if len(lists) == 1:
            fused = list(lists[0])
        else:
            fused = reciprocal_rank_fusion(lists, weights, self.cfg.retrieval.rrf_k)

        hits = self.reranker.rerank(query, fused[:candidate_k], top_k)
        if self.cfg.retrieval.min_score > 0:
            hits = [h for h in hits if h.score >= self.cfg.retrieval.min_score]
        for rank, hit in enumerate(hits, start=1):
            hit.rank = rank
        self.cache.set(query, [h.to_dict() for h in hits])
        return RetrievalResult(query=query, hits=hits,
                               latency_s=time.perf_counter() - started, cached=False)

    def describe(self) -> dict[str, object]:
        return {"mode": self.cfg.retrieval.mode, "top_k": self.cfg.retrieval.top_k,
                "candidate_k": self.cfg.retrieval.candidate_k,
                "reranker": self.reranker.name, "chunks": len(self.chunks),
                "store": self.store.name, "embedder": self.embedder.name,
                "corpus_hash": self.corpus_hash,
                "cache": self.cache.stats()}


def build_retriever(cfg: Config, chunks: Sequence[Chunk] | None = None,
                    index_dir: str | Path | None = None) -> Retriever:
    """Load (or rebuild in memory) everything the retriever needs."""
    from ..ingest.pipeline import load_chunks

    index_dir = Path(index_dir or Path(cfg.paths.index_dir) / cfg.fingerprint())
    embedder = get_embedder(cfg)
    if chunks is None:
        chunks = load_chunks(cfg, Path(cfg.paths.index_dir))
    chunks = list(chunks)
    store = get_store(cfg, dim=embedder.dim, path=index_dir)
    if not store.load():
        log.info("vector store empty at %s; embedding %d chunk(s) in memory",
                 index_dir, len(chunks))
        store.add(chunks, embedder.encode([c.text for c in chunks]))
    else:
        stored = store.all_chunks()
        if stored:
            chunks = stored
    return Retriever(cfg, chunks, store, embedder)
