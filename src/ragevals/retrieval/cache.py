"""Disk cache for retrieval results.

Evaluation re-runs the same ~80 questions many times while you iterate on the
*generation* side. Caching retrieval makes those loops cheap and, more
importantly, removes retrieval jitter as a confound: the cache key is
sha1(query + retrieval settings + index fingerprint), so a cache hit is only
ever returned for a byte-identical configuration.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any


class RetrievalCache:
    def __init__(self, directory: str | Path, namespace: str = "default",
                 enabled: bool = True) -> None:
        self.enabled = enabled
        self.namespace = namespace
        self.dir = Path(directory) / namespace
        self._lock = threading.Lock()
        self._memory: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def key(self, query: str) -> str:
        return hashlib.sha1(f"{self.namespace}||{query}".encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, query: str) -> list[dict[str, Any]] | None:
        if not self.enabled:
            return None
        key = self.key(query)
        with self._lock:
            if key in self._memory:
                self.hits += 1
                return self._memory[key]
            path = self._path(key)
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    return None
                self._memory[key] = payload["hits"]
                self.hits += 1
                return payload["hits"]
        self.misses += 1
        return None

    def set(self, query: str, hits: list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        key = self.key(query)
        with self._lock:
            self._memory[key] = hits
            try:
                self._path(key).write_text(
                    json.dumps({"query": query, "hits": hits}, ensure_ascii=False),
                    encoding="utf-8")
            except OSError:
                pass

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}
