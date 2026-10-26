"""Embedding backends.

``auto`` prefers sentence-transformers when it is installed and the weights are
already cached, and otherwise falls back to a deterministic hashing embedder.
The fallback is not a toy: it is a signed random-projection of word unigrams,
bigrams and character 4-grams with sub-linear term frequency and L2
normalisation -- i.e. a hashed TF-IDF-style vector. It is weaker than a trained
encoder on paraphrase, but it is *reproducible, offline and free*, which is what
lets the whole evaluation suite run in CI with no API key and no model download.

Every result row records which embedder produced it (``embedder`` column), so a
hashing-backend number is never silently compared with a transformer number.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

import numpy as np

from .config import Config
from .logging_utils import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder:
    name = "base"
    dim = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "dim": self.dim}


class HashingEmbedder(Embedder):
    """Deterministic, offline, dependency-light."""

    name = "hashing"

    def __init__(self, dim: int = 512, char_ngram: int = 4) -> None:
        self.dim = int(dim)
        self.char_ngram = char_ngram

    @staticmethod
    def _bucket(feature: str, dim: int) -> tuple[int, float]:
        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(h, "big")
        return value % dim, 1.0 if (value >> 63) & 1 else -1.0

    def _features(self, text: str) -> dict[str, float]:
        tokens = tokenize(text)
        counts: dict[str, float] = {}
        for tok in tokens:
            counts[f"w:{tok}"] = counts.get(f"w:{tok}", 0.0) + 1.0
        for a, b in zip(tokens, tokens[1:], strict=False):
            key = f"b:{a}_{b}"
            counts[key] = counts.get(key, 0.0) + 1.0
        compact = re.sub(r"\s+", " ", text.lower())
        n = self.char_ngram
        for i in range(0, max(0, len(compact) - n + 1)):
            key = f"c:{compact[i:i + n]}"
            counts[key] = counts.get(key, 0.0) + 1.0
        # sub-linear tf: one mention of "fine" should not dominate ten of "data"
        return {k: 1.0 + math.log(v) for k, v in counts.items()}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for feature, weight in self._features(text or "").items():
                idx, sign = self._bucket(feature, self.dim)
                out[i, idx] += sign * weight
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "dim": self.dim, "char_ngram": self.char_ngram,
                "trained": False}


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: WPS433

        self.model_name = model
        self._model = SentenceTransformer(model)
        self.batch_size = batch_size
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(list(texts), batch_size=self.batch_size,
                                  convert_to_numpy=True, normalize_embeddings=True,
                                  show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "model": self.model_name, "dim": self.dim,
                "trained": True}


def get_embedder(cfg: Config) -> Embedder:
    backend = cfg.embedding.backend
    if backend in {"auto", "sentence-transformers"}:
        try:
            emb = SentenceTransformerEmbedder(cfg.embedding.model,
                                              cfg.embedding.batch_size)
            log.info("embedder: sentence-transformers/%s (dim=%d)",
                     cfg.embedding.model, emb.dim)
            return emb
        except Exception as exc:  # noqa: BLE001 - any failure falls back
            if backend == "sentence-transformers":
                raise
            log.info("sentence-transformers unavailable (%s); using hashing embedder",
                     type(exc).__name__)
    if backend in {"auto", "hashing"}:
        return HashingEmbedder(cfg.embedding.dim)
    raise ValueError(f"unknown embedding backend {backend!r}")


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of a (d,) query against a (n, d) matrix."""
    if b.ndim == 1:
        b = b[None, :]
    denom = (np.linalg.norm(a) * np.linalg.norm(b, axis=1)) + 1e-12
    return (b @ a) / denom
