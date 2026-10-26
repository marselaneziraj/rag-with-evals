"""Okapi BM25, implemented here rather than pulled in as a dependency.

Dense retrieval alone is unreliable on this corpus class: statutes are full of
rare, high-signal tokens ("Article 22", "DPIA", "Annex III") that an embedding
model smooths away, and a user's question often quotes them verbatim. BM25 is
the right tool for exact-term matching -- hence the hybrid default.

k1 controls term-frequency saturation (1.5 is the standard setting); b controls
length normalisation (0.75 standard). Section headings are indexed with the
body text, which is why the section-aware chunker helps the lexical side too.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from ..embeddings import tokenize
from ..ingest.chunking import Chunk
from ..store.base import SearchHit


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = list(chunks)
        self.docs: list[Counter] = []
        self.lengths: list[int] = []
        df: Counter = Counter()
        for chunk in self.chunks:
            text = f"{chunk.section} {chunk.text}" if chunk.section else chunk.text
            tokens = tokenize(text)
            counts = Counter(tokens)
            self.docs.append(counts)
            self.lengths.append(len(tokens))
            df.update(counts.keys())
        self.n = len(self.chunks)
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        # Robertson/Sparck-Jones idf with the +0.5 smoothing, floored at 0
        self.idf: dict[str, float] = {
            term: max(math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5)), 0.0)
            for term, freq in df.items()
        }

    def score_all(self, query: str) -> list[float]:
        terms = tokenize(query)
        scores = [0.0] * self.n
        if not terms or not self.n:
            return scores
        for term in terms:
            idf = self.idf.get(term)
            if not idf:
                continue
            for i, counts in enumerate(self.docs):
                tf = counts.get(term, 0)
                if not tf:
                    continue
                dl = self.lengths[i] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return scores

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        scores = self.score_all(query)
        order = sorted(range(self.n), key=lambda i: -scores[i])[:k]
        return [SearchHit(chunk=self.chunks[i], score=float(scores[i]), rank=rank,
                          source_ranker="bm25")
                for rank, i in enumerate(order, start=1) if scores[i] > 0]
