"""Optional RAGAS integration.

RAGAS is the reference implementation of faithfulness / context precision /
context recall, and when an API key is present it is worth running: its
LLM-based claim decomposition is finer-grained than the sentence-level
decomposition used here. It is *optional* because it pulls a large dependency
tree, needs a key and a network, and is the single most common reason a RAG
repo cannot be reproduced six months later.

Contract: ``ragas_scores`` returns ``None`` whenever RAGAS cannot run, and the
harness falls back to the built-in metrics. The results table always records
which engine produced the numbers (``metrics_engine`` column), because a RAGAS
faithfulness and a lexical faithfulness are not the same quantity and must
never be averaged together.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..logging_utils import get_logger

log = get_logger(__name__)


def ragas_available() -> bool:
    try:
        import datasets  # noqa: F401
        import ragas  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def ragas_scores(questions: Sequence[str], answers: Sequence[str],
                 contexts: Sequence[Sequence[str]], references: Sequence[str],
                 ) -> dict[str, Any] | None:
    """Per-item RAGAS scores, or None if RAGAS is unusable in this environment."""
    if not ragas_available():
        log.info("ragas not installed; using built-in metrics")
        return None
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": list(questions),
            "answer": list(answers),
            "contexts": [list(c) for c in contexts],
            "ground_truth": list(references),
        })
        result = evaluate(dataset, metrics=[faithfulness, context_precision,
                                            context_recall, answer_correctness])
        frame = result.to_pandas()
        per_item: list[dict[str, float]] = []
        for _, row in frame.iterrows():
            per_item.append({
                "faithfulness": float(row.get("faithfulness", float("nan"))),
                "context_precision": float(row.get("context_precision", float("nan"))),
                "context_recall": float(row.get("context_recall", float("nan"))),
                "answer_correctness": float(row.get("answer_correctness", float("nan"))),
            })
        return {"engine": "ragas", "per_item": per_item}
    except Exception as exc:  # noqa: BLE001
        log.warning("ragas evaluation failed (%s: %s); falling back to built-in metrics",
                    type(exc).__name__, exc)
        return None
