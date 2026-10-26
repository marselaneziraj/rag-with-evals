"""Rerankers.

A bi-encoder scores query and chunk independently, so it cannot model term
interaction; a cross-encoder reads the pair jointly and is markedly more
accurate at the top of the list -- at a cost of one forward pass per candidate.
The standard pattern, implemented here, is retrieve ~20 cheaply then rerank to
top-5. ``lexical`` is an offline stand-in that rescores candidates by token
overlap and section-title match; it is weaker than a cross-encoder but keeps
the rerank *stage* testable without downloading weights.
"""
from __future__ import annotations

from collections.abc import Sequence

from ..embeddings import tokenize
from ..logging_utils import get_logger
from ..store.base import SearchHit

log = get_logger(__name__)


class Reranker:
    name = "none"

    def rerank(self, query: str, hits: Sequence[SearchHit], top_k: int) -> list[SearchHit]:
        return list(hits)[:top_k]


class LexicalReranker(Reranker):
    name = "lexical"

    def rerank(self, query: str, hits: Sequence[SearchHit], top_k: int) -> list[SearchHit]:
        q = set(tokenize(query))
        if not q:
            return list(hits)[:top_k]

        def score(hit: SearchHit) -> float:
            toks = set(tokenize(hit.chunk.text))
            overlap = len(q & toks) / len(q)
            title = len(q & set(tokenize(hit.chunk.section))) / len(q) if hit.chunk.section else 0.0
            return 0.70 * overlap + 0.20 * title + 0.10 * hit.score

        ordered = sorted(hits, key=score, reverse=True)[:top_k]
        for rank, hit in enumerate(ordered, start=1):
            hit.rank = rank
            hit.score = score(hit)
        return ordered


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"

    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder  # noqa: WPS433

        self.model_name = model
        self._model = CrossEncoder(model)

    def rerank(self, query: str, hits: Sequence[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        scores = self._model.predict([(query, h.chunk.text) for h in hits])
        ordered = [h for _, h in sorted(zip(scores, hits, strict=False), key=lambda p: -float(p[0]))]
        for rank, (hit, sc) in enumerate(zip(ordered, sorted(scores, reverse=True), strict=False), start=1):
            hit.rank = rank
            hit.score = float(sc)
        return ordered[:top_k]


def build_reranker(name: str, model: str) -> Reranker:
    if name in {"none", "", None}:
        return Reranker()
    if name == "lexical":
        return LexicalReranker()
    if name == "cross-encoder":
        try:
            return CrossEncoderReranker(model)
        except Exception as exc:  # noqa: BLE001
            log.warning("cross-encoder unavailable (%s); falling back to lexical rerank",
                        type(exc).__name__)
            return LexicalReranker()
    raise ValueError(f"unknown reranker {name!r}")
