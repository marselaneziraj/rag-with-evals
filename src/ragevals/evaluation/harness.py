"""The evaluation harness: gold set in, results table out.

One run produces three artefacts, and all three are needed to defend a number:

``<run>/records.jsonl``   one row per gold item: the answer, the retrieved
                          chunk ids, every per-item metric, latency, tokens,
                          cost, and the validator's verdict. This is the
                          evidence. Any aggregate can be recomputed from it.
``<run>/summary.json``    aggregates plus the full config, the embedder, the
                          judge's identity and rubric hash, and the metrics
                          engine. This is the provenance.
``<run>/records.csv``     the same rows, for people who open things in Excel.

Aggregates are means over the applicable subset, not over everything: context
metrics are averaged over answerable items only (they are undefined when there
is no reference answer), while refusal metrics are computed over the whole set.
Mixing those two populations is the most common way a RAG evaluation ends up
reporting a number that cannot be reproduced.
"""
from __future__ import annotations

import json
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Config
from ..cost import summarise_latency
from ..logging_utils import get_logger
from ..pipeline import AnswerResult, RAGPipeline
from .goldset import GoldItem, gold_set_stats, load_gold_set
from .judge import Judge, build_judge
from .metrics import (
    aggregate,
    citation_metrics,
    context_precision,
    context_recall,
    faithfulness,
    refusal_metrics,
)
from .ragas_adapter import ragas_scores

log = get_logger(__name__)


@dataclass
class EvaluationReport:
    config_name: str
    fingerprint: str
    records: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    run_dir: Path | None = None

    def headline(self) -> dict[str, float]:
        keys = ["faithfulness", "context_precision", "context_recall",
                "answer_correctness", "refusal_accuracy", "citation_validity",
                "p95_latency_s", "cost_per_query_usd"]
        return {k: float(self.summary.get(k, 0.0)) for k in keys}

    def to_markdown_row(self) -> str:
        h = self.headline()
        return ("| {name} | {faithfulness:.3f} | {context_precision:.3f} | "
                "{context_recall:.3f} | {answer_correctness:.3f} | "
                "{refusal_accuracy:.3f} | {citation_validity:.3f} | "
                "{p95_latency_s:.3f} | {cost_per_query_usd:.5f} |").format(
            name=self.config_name, **h)


class EvaluationHarness:
    def __init__(self, cfg: Config, pipeline: RAGPipeline,
                 judge: Judge | None = None) -> None:
        self.cfg = cfg
        self.pipeline = pipeline
        self.judge = judge if judge is not None else build_judge(cfg, pipeline.backend)

    # ------------------------------------------------------------------ #
    def _score_item(self, item: GoldItem, result: AnswerResult) -> dict[str, Any]:
        contexts = result.contexts
        hit_keys = [(h.chunk.source.strip().lower(), h.chunk.page) for h in result.hits]
        gold_keys = item.gold_keys or None

        row: dict[str, Any] = {
            "id": item.id, "question": item.question, "answerable": item.answerable,
            "difficulty": item.difficulty, "tags": item.tags,
            "reference_answer": item.reference_answer,
            "answer": result.answer, "refused": result.refused,
            "expected_refusal": not item.answerable,
            "refusal_correct": result.refused == (not item.answerable),
            "retrieved_ids": [h.chunk.id for h in result.hits],
            "retrieved_keys": [f"{s}#{p}" for s, p in hit_keys],
            "contexts": contexts,
            "valid_citations": result.valid_citations,
            "invalid_citations": result.invalid_citations,
            "retrieval_latency_s": result.retrieval_latency_s,
            "generation_latency_s": result.generation_latency_s,
            "total_latency_s": result.total_latency_s,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cost_usd": result.cost_usd,
            "blocked_by_validator": result.blocked_by_validator,
            "validation_reasons": result.validation_reasons,
            "sanitizer_flags": result.sanitizer_flags,
        }

        faith = faithfulness(result.answer, contexts) if not result.refused \
            else None
        row["faithfulness"] = float(faith.value) if faith else None
        row["faithfulness_detail"] = faith.detail if faith else None

        if item.answerable:
            cp = context_precision(contexts, item.reference_answer,
                                   gold_keys=gold_keys, hit_keys=hit_keys)
            cr = context_recall(item.reference_answer, contexts,
                                gold_keys=gold_keys, hit_keys=hit_keys)
            correctness = self.judge.score("answer_correctness", item.question,
                                           item.reference_answer, result.answer,
                                           "\n".join(contexts)[:4000])
            row["context_precision"] = float(cp.value)
            row["context_recall"] = float(cr.value)
            row["answer_correctness"] = float(correctness.value)
            row["answer_correctness_detail"] = correctness.detail
            row["context_precision_detail"] = cp.detail
            row["context_recall_detail"] = cr.detail
        else:
            row["context_precision"] = None
            row["context_recall"] = None
            # On an unanswerable item the only correct behaviour is refusal, so
            # correctness is the refusal decision itself -- scoring the text of
            # a refusal against an empty reference would be meaningless.
            row["answer_correctness"] = 1.0 if result.refused else 0.0
        return row

    # ------------------------------------------------------------------ #
    def run(self, items: Sequence[GoldItem], run_dir: str | Path | None = None,
            progress: bool = True) -> EvaluationReport:
        started = time.perf_counter()
        records: list[dict[str, Any]] = []
        for i, item in enumerate(items, start=1):
            result = self.pipeline.answer(item.question)
            records.append(self._score_item(item, result))
            if progress and (i % 10 == 0 or i == len(items)):
                log.info("evaluated %d/%d", i, len(items))

        answerable = [r for r in records if r["answerable"]]
        answered = [r for r in records if not r["refused"]]
        metrics_engine = "builtin"

        if self.cfg.evaluation.use_ragas and answerable:
            rag = ragas_scores(
                [r["question"] for r in answerable],
                [r["answer"] for r in answerable],
                [[c for c in r.get("contexts", [])] or [""] for r in answerable],
                [r["reference_answer"] for r in answerable])
            if rag:
                metrics_engine = "ragas"
                for row, scores in zip(answerable, rag["per_item"], strict=False):
                    row.update({f"ragas_{k}": v for k, v in scores.items()})

        latencies = [r["total_latency_s"] for r in records]
        latency = summarise_latency(latencies)
        refusal = refusal_metrics([r["expected_refusal"] for r in records],
                                  [bool(r["refused"]) for r in records])
        citations = citation_metrics(records)

        summary: dict[str, Any] = {
            "config_name": self.cfg.name,
            "config_fingerprint": self.cfg.fingerprint(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_items": len(records),
            "n_answerable": len(answerable),
            "n_unanswerable": len(records) - len(answerable),
            "metrics_engine": metrics_engine,
            "faithfulness": aggregate([r["faithfulness"] for r in records
                                       if r["faithfulness"] is not None]),
            "context_precision": aggregate([r["context_precision"] for r in answerable]),
            "context_recall": aggregate([r["context_recall"] for r in answerable]),
            "answer_correctness": aggregate([r["answer_correctness"] for r in records]),
            "answer_correctness_answerable_only": aggregate(
                [r["answer_correctness"] for r in answerable]),
            **refusal,
            **citations,
            "mean_latency_s": latency["mean_s"],
            "p50_latency_s": latency["p50_s"],
            "p95_latency_s": latency["p95_s"],
            "max_latency_s": latency["max_s"],
            "cost_per_query_usd": (sum(r["cost_usd"] for r in records) / len(records)
                                   if records else 0.0),
            "total_cost_usd": sum(r["cost_usd"] for r in records),
            "judge_cost_usd": self.judge.stats().get("cost_usd", 0.0),
            "mean_prompt_tokens": aggregate([r["prompt_tokens"] for r in records]),
            "mean_completion_tokens": aggregate([r["completion_tokens"] for r in records]),
            "answered": len(answered),
            "blocked_by_validator": sum(1 for r in records if r["blocked_by_validator"]),
            "wall_seconds": round(time.perf_counter() - started, 2),
            "gold_set": gold_set_stats(items),
            "pipeline": self.pipeline.describe(),
            "judge": self.judge.stats(),
            "config": self.cfg.to_dict(),
            "environment": {"python": platform.python_version(),
                            "platform": platform.platform()},
        }

        report = EvaluationReport(self.cfg.name, self.cfg.fingerprint(), records, summary)
        if run_dir:
            report.run_dir = _write_outputs(report, Path(run_dir))
        return report


def _write_outputs(report: EvaluationReport, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as fh:
        for row in report.records:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    (run_dir / "summary.json").write_text(
        json.dumps(report.summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    try:
        import pandas as pd

        flat = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))}
                for row in report.records]
        pd.DataFrame(flat).to_csv(run_dir / "records.csv", index=False,
                                  encoding="utf-8")
    except ImportError:  # pragma: no cover
        pass
    log.info("evaluation written to %s", run_dir)
    return run_dir


def evaluate_config(cfg: Config, gold_path: str | Path | None = None,
                    run_dir: str | Path | None = None,
                    pipeline: RAGPipeline | None = None) -> EvaluationReport:
    """Convenience entry point used by the CLI and the experiment runner."""
    items = load_gold_set(gold_path or cfg.paths.gold_set)
    pipeline = pipeline or RAGPipeline.from_config(cfg)
    harness = EvaluationHarness(cfg, pipeline)
    run_dir = Path(run_dir or Path(cfg.paths.results_dir) / "runs" /
                   f"{cfg.name}-{cfg.fingerprint()}")
    return harness.run(items, run_dir=run_dir)
