"""Run every configuration in a sweep file and compare them.

The runner exists so that "which chunk size is better?" is answered by a
command rather than by memory. Two details make the comparison trustworthy:

* **The gold set and the judge are held fixed across configurations.** Only the
  patched keys differ, and the patch is recorded per row, so the table is a
  controlled comparison rather than a collection of unrelated runs.
* **Each configuration gets its own index.** Chunking changes the corpus
  representation, so reusing one index across chunkers would silently evaluate
  the wrong thing. Indexes are keyed by the config fingerprint and rebuilt only
  when that fingerprint changes.
"""
from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config, load_sweep
from ..evaluation.goldset import GoldItem, load_gold_set
from ..evaluation.harness import EvaluationHarness, EvaluationReport
from ..evaluation.judge import build_judge
from ..ingest.pipeline import build_chunks
from ..logging_utils import get_logger
from ..pipeline import RAGPipeline

log = get_logger(__name__)

COMPARISON_COLUMNS: list[tuple[str, str]] = [
    ("config", "Configuration"),
    ("faithfulness", "Faithful."),
    ("context_precision", "Ctx prec."),
    ("context_recall", "Ctx recall"),
    ("answer_correctness", "Correct."),
    ("refusal_accuracy", "Refusal acc."),
    ("over_refusal_rate", "Over-refusal"),
    ("citation_validity", "Cite valid."),
    ("p95_latency_s", "p95 lat. (s)"),
    ("cost_per_query_usd", "USD/query"),
]


@dataclass
class ExperimentRunner:
    base: Config
    experiments: list[tuple[str, dict[str, Any]]]
    gold_path: str | None = None
    results_dir: str | None = None
    reports: list[EvaluationReport] = field(default_factory=list)

    def run(self, progress: bool = True) -> list[EvaluationReport]:
        items: Sequence[GoldItem] = load_gold_set(self.gold_path or self.base.paths.gold_set)
        results_dir = Path(self.results_dir or self.base.paths.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        log.info("sweep: %d configuration(s) x %d gold item(s)",
                 len(self.experiments), len(items))

        # Chunking is the only stage that must be redone per configuration;
        # cache by the ingest settings so configs that share them share work.
        chunk_cache: dict[str, list] = {}

        for name, patch in self.experiments:
            cfg = self.base.merged(patch)
            cfg.name = name
            started = time.perf_counter()
            ingest_key = json.dumps(cfg.ingest.__dict__, sort_keys=True)
            if ingest_key not in chunk_cache:
                chunks, report = build_chunks(cfg)
                chunk_cache[ingest_key] = chunks
                log.info("[%s] ingestion: %s", name, report.summary())
            chunks = chunk_cache[ingest_key]

            pipeline = RAGPipeline.from_config(cfg, chunks=chunks)
            judge = build_judge(cfg, pipeline.backend)
            harness = EvaluationHarness(cfg, pipeline, judge)
            run_dir = results_dir / "runs" / f"{name}-{cfg.fingerprint()}"
            report = harness.run(items, run_dir=run_dir, progress=progress)
            report.summary["patch"] = patch
            report.summary["n_chunks"] = len(chunks)
            (run_dir / "summary.json").write_text(
                json.dumps(report.summary, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
            self.reports.append(report)
            log.info("[%s] done in %.1fs -- faithfulness %.3f, correctness %.3f, "
                     "refusal acc %.3f", name, time.perf_counter() - started,
                     report.summary["faithfulness"],
                     report.summary["answer_correctness"],
                     report.summary["refusal_accuracy"])
        return self.reports

    # ------------------------------------------------------------------ #
    def comparison_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for report in self.reports:
            s = report.summary
            row = {"config": report.config_name}
            for key, _ in COMPARISON_COLUMNS[1:]:
                row[key] = float(s.get(key, 0.0))
            row["n_chunks"] = s.get("n_chunks", 0)
            row["fingerprint"] = report.fingerprint
            rows.append(row)
        return rows

    def to_markdown(self) -> str:
        header = "| " + " | ".join(label for _, label in COMPARISON_COLUMNS) + " |"
        divider = "|" + "|".join(["---"] * len(COMPARISON_COLUMNS)) + "|"
        lines = [header, divider]
        for row in self.comparison_rows():
            cells = [str(row["config"])]
            for key, _ in COMPARISON_COLUMNS[1:]:
                value = row[key]
                cells.append(f"{value:.5f}" if key == "cost_per_query_usd"
                             else f"{value:.3f}")
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def save(self, results_dir: str | Path | None = None) -> dict[str, Path]:
        results_dir = Path(results_dir or self.results_dir or self.base.paths.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        rows = self.comparison_rows()
        out: dict[str, Path] = {}

        csv_path = results_dir / "comparison.csv"
        try:
            import pandas as pd

            pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
        except ImportError:  # pragma: no cover
            import csv

            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        out["csv"] = csv_path

        md_path = results_dir / "comparison.md"
        md_path.write_text("# Configuration comparison\n\n" + self.to_markdown() + "\n",
                           encoding="utf-8")
        out["markdown"] = md_path

        json_path = results_dir / "comparison.json"
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        out["json"] = json_path

        from .plots import plot_comparison

        out["figure"] = plot_comparison(rows, results_dir / "comparison.png")
        log.info("comparison written: %s", ", ".join(str(p) for p in out.values()))
        return out


def run_sweep(sweep_path: str | Path, gold_path: str | Path | None = None,
              results_dir: str | Path | None = None) -> ExperimentRunner:
    base, experiments = load_sweep(sweep_path)
    runner = ExperimentRunner(base=base, experiments=experiments,
                              gold_path=str(gold_path) if gold_path else None,
                              results_dir=str(results_dir) if results_dir else None)
    runner.run()
    runner.save()
    return runner
