from __future__ import annotations

import pytest

from ragevals.embeddings import HashingEmbedder
from ragevals.retrieval.bm25 import BM25Index
from ragevals.retrieval.cache import RetrievalCache
from ragevals.retrieval.rerank import LexicalReranker, build_reranker
from ragevals.retrieval.retriever import Retriever, corpus_fingerprint, reciprocal_rank_fusion
from ragevals.store.base import SearchHit
from ragevals.store.numpy_store import NumpyStore


def _retriever(cfg, chunks, tmp_path, mode="hybrid"):
    cfg = cfg.merged({"retrieval": {"mode": mode, "top_k": 2, "candidate_k": 5}})
    embedder = HashingEmbedder(dim=256)
    store = NumpyStore(tmp_path / "idx", dim=256)
    store.add(chunks, embedder.encode([c.text for c in chunks]))
    return Retriever(cfg, chunks, store, embedder)


def test_bm25_ranks_exact_terms_first(chunks):
    index = BM25Index(chunks)
    hits = index.search("20 000 000 EUR administrative fines", k=2)
    assert hits and hits[0].chunk.id == "c3"


def test_bm25_empty_query(chunks):
    assert BM25Index(chunks).search("", k=3) == []


def test_bm25_indexes_section_headings(chunks):
    index = BM25Index(chunks)
    hits = index.search("Article 83", k=1)
    assert hits and hits[0].chunk.id == "c3"


def test_rrf_is_scale_free():
    a = [SearchHit(chunk=c, score=s, rank=i, source_ranker="dense")
         for i, (c, s) in enumerate([("x", 0.9), ("y", 0.8)], start=1)
         for c in [type("C", (), {"id": c})()]]
    b = [SearchHit(chunk=type("C", (), {"id": "y"})(), score=900.0, rank=1,
                   source_ranker="bm25")]
    fused = reciprocal_rank_fusion([a, b], [0.5, 0.5])
    assert fused[0].chunk.id == "y", "agreement across rankers must win"
    assert all(h.score < 1.0 for h in fused), "fused scores are rank-based, not raw"


def test_hybrid_retrieval_returns_top_k(cfg, chunks, tmp_path):
    retriever = _retriever(cfg, chunks, tmp_path)
    result = retriever.retrieve("How long do I have to report a breach?")
    assert len(result.hits) == 2
    assert result.hits[0].rank == 1
    assert "+" in result.hits[0].source_ranker or result.hits[0].source_ranker


def test_dense_and_bm25_modes_work(cfg, chunks, tmp_path):
    for mode in ("dense", "bm25"):
        result = _retriever(cfg, chunks, tmp_path, mode=mode).retrieve("impact assessment")
        assert result.hits


def test_lexical_reranker_reorders(chunks):
    hits = [SearchHit(chunk=c, score=0.5, rank=i, source_ranker="dense")
            for i, c in enumerate(chunks, start=1)]
    reranked = LexicalReranker().rerank("administrative fines turnover", hits, top_k=1)
    assert reranked[0].chunk.id == "c3"


def test_build_reranker_names():
    assert build_reranker("none", "").name == "none"
    assert build_reranker("lexical", "").name == "lexical"
    with pytest.raises(ValueError):
        build_reranker("telepathy", "")


def test_cache_roundtrip(tmp_path):
    cache = RetrievalCache(tmp_path, namespace="ns", enabled=True)
    assert cache.get("q") is None
    cache.set("q", [{"id": "c1", "score": 1.0, "rank": 1, "ranker": "dense"}])
    assert cache.get("q")[0]["id"] == "c1"


def test_cache_can_be_disabled(tmp_path):
    cache = RetrievalCache(tmp_path, namespace="ns", enabled=False)
    cache.set("q", [{"id": "c1"}])
    assert cache.get("q") is None


def test_cache_namespace_binds_the_corpus(cfg, chunks, tmp_path):
    """Regression: a poisoned corpus must not read the clean corpus's cache."""
    from ragevals.ingest.chunking import Chunk

    extra = Chunk(id="evil", text="unrelated", source="x.pdf", page=1, page_end=1)
    assert corpus_fingerprint(chunks) != corpus_fingerprint(list(chunks) + [extra])
