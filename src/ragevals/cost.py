"""Price table and latency helpers.

Cost per query is reported because it is half of every real deployment
decision: a configuration that wins faithfulness by two points and costs four
times as much is not obviously the winner, and you cannot have that argument
without the number. Prices are *config values*, not hard-coded constants, so
they can be corrected without touching code -- vendor pricing moves.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# USD per 1000 tokens (input, output). Reference values only -- override in YAML.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "o4-mini": (0.0011, 0.0044),
    "offline-extractive": (0.0, 0.0),
}


def lookup_price(model: str) -> tuple[float, float]:
    """Exact match first, then longest known prefix; unknown models cost 0."""
    if model in PRICE_TABLE:
        return PRICE_TABLE[model]
    candidates = [k for k in PRICE_TABLE if model.startswith(k)]
    if candidates:
        return PRICE_TABLE[max(candidates, key=len)]
    return (0.0, 0.0)


def percentile(values: Sequence[float], q: float) -> float:
    """p95 with linear interpolation; empty input is 0.0.

    p95 rather than the mean because latency distributions in RAG are
    right-skewed (a cache miss plus a long generation is a different animal
    from the median query) and users feel the tail, not the average.
    """
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(list(values), dtype=float), q))


def summarise_latency(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean_s": 0.0, "p50_s": 0.0, "p95_s": 0.0, "max_s": 0.0}
    arr = np.asarray(list(values), dtype=float)
    return {"mean_s": float(arr.mean()), "p50_s": float(np.percentile(arr, 50)),
            "p95_s": float(np.percentile(arr, 95)), "max_s": float(arr.max())}
