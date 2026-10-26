"""Command-line interface -- every Makefile target lands here.

    python -m ragevals.cli samples
    python -m ragevals.cli ingest     --config configs/default.yaml
    python -m ragevals.cli ask        --config configs/default.yaml "your question"
    python -m ragevals.cli propose    --config configs/default.yaml -n 60
    python -m ragevals.cli evaluate   --config configs/default.yaml
    python -m ragevals.cli experiment --sweep configs/experiments/sweep.yaml
    python -m ragevals.cli injection  --config configs/default.yaml
    python -m ragevals.cli report
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import Config, load_config
from .logging_utils import get_logger, setup_logging

log = get_logger(__name__)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", "-c", default="configs/default.yaml",
                        help="path to a YAML config (default: configs/default.yaml)")
    parser.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE",
                        help="dotted override, e.g. -s retrieval.top_k=8")


def _apply_overrides(cfg: Config, assignments: Sequence[str]) -> Config:
    patch: dict = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise SystemExit(f"--set expects KEY=VALUE, got {assignment!r}")
        key, value = assignment.split("=", 1)
        node = patch
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)
    return cfg.merged(patch) if patch else cfg


def _coerce(value: str):
    low = value.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load(args) -> Config:
    cfg = load_config(args.config)
    return _apply_overrides(cfg, args.set)


# --------------------------------------------------------------------------- #
def cmd_samples(args) -> int:
    from .samples import build_sample_corpus

    paths = build_sample_corpus(args.sample_dir, args.corpus_dir)
    print(f"built {len(paths)} sample PDF(s) in {args.corpus_dir}:")
    for p in paths:
        print(f"  {p}")
    return 0


def cmd_ingest(args) -> int:
    from .ingest.pipeline import ingest_corpus

    cfg = _load(args)
    report = ingest_corpus(cfg, corpus_dir=args.corpus_dir)
    print(f"[{cfg.name}] {report.summary()}")
    print(f"index: {report.index_dir}")
    return 0


def cmd_ask(args) -> int:
    from .pipeline import RAGPipeline

    cfg = _load(args)
    pipeline = RAGPipeline.from_config(cfg)
    result = pipeline.answer(args.question)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str))
        return 0
    print("\n=== ANSWER ===")
    print(result.answer)
    print("\n=== RETRIEVED ===")
    for hit in result.hits:
        head = f"  [{hit.rank}] {hit.chunk.source} p.{hit.chunk.page} "
        head += f"({hit.source_ranker}, score={hit.score:.4f})"
        print(head)
        if hit.chunk.section:
            print(f"      section: {hit.chunk.section}")
        print(f"      {hit.chunk.text[:200].strip()}...")
    print(f"\nlatency {result.total_latency_s:.3f}s | tokens "
          f"{result.prompt_tokens}+{result.completion_tokens} | "
          f"cost ${result.cost_usd:.5f} | citations "
          f"{result.valid_citations} valid / {result.invalid_citations} invalid")
    if result.validation_reasons:
        print(f"validator: {', '.join(result.validation_reasons)}")
    return 0


def cmd_propose(args) -> int:
    from .tools.propose import propose_questions

    cfg = _load(args)
    out = Path(args.out)
    items = propose_questions(cfg, n=args.n, seed=args.seed, out_path=out)
    print(f"proposed {len(items)} candidate question(s) -> {out}")
    print("Curate by hand: fix wording, write reference answers, set 'answerable',")
    print("and keep roughly 25% unanswerable. Nothing here is a gold item until a")
    print("human has read the cited page and signed off on the answer.")
    return 0


def cmd_evaluate(args) -> int:
    from .evaluation.harness import evaluate_config

    cfg = _load(args)
    report = evaluate_config(cfg, gold_path=args.gold, run_dir=args.out)
    summary = report.summary
    print(f"\n=== {cfg.name} ({cfg.fingerprint()}) -- {summary['n_items']} items "
          f"({summary['n_unanswerable']} unanswerable) ===")
    for key in ["faithfulness", "context_precision", "context_recall",
                "answer_correctness", "refusal_accuracy", "over_refusal_rate",
                "hallucination_rate_on_unanswerable", "citation_coverage",
                "citation_validity", "p95_latency_s", "cost_per_query_usd"]:
        print(f"  {key:38s} {float(summary.get(key, 0.0)):.4f}")
    print(f"\nrecords: {report.run_dir}")
    return 0


def cmd_experiment(args) -> int:
    from .experiments.runner import run_sweep

    runner = run_sweep(args.sweep, gold_path=args.gold, results_dir=args.out)
    print("\n" + runner.to_markdown())
    return 0


def cmd_injection(args) -> int:
    from .ingest.pipeline import build_chunks
    from .safety.injection_suite import load_injection_set, run_injection_suite

    cfg = _load(args)
    docs = load_injection_set(args.set_path or cfg.paths.injection_set)
    chunks, _ = build_chunks(cfg)
    out = Path(args.out or Path(cfg.paths.results_dir) / "injection")
    report = run_injection_suite(cfg, docs, chunks, results_dir=out)
    print("\n" + report.to_markdown())
    print(f"\ninjection success {100 * report.before:.1f}% -> {100 * report.after:.1f}% "
          f"({100 * report.reduction:.1f}% relative reduction)")
    if cfg.generation.backend == "offline":
        print("NOTE: offline backend -- susceptibility is simulated, not measured. "
              "Re-run with generation.backend=ollama or openai before quoting these.")
    return 0


def cmd_gold_stats(args) -> int:
    from .evaluation.goldset import gold_set_stats, load_gold_set, validate_gold_set

    items = load_gold_set(args.gold, strict=False)
    problems = validate_gold_set(items)
    print(json.dumps(gold_set_stats(items), indent=2))
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\ngold set OK")
    return 0


def cmd_report(args) -> int:
    from .tools.report import build_report

    path = build_report(Path(args.results), Path(args.out))
    print(f"report written to {path}")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragevals",
                                     description="RAG with a rigorous evaluation suite")
    parser.add_argument("--log", default="INFO", help="log level")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("samples", help="build the demo PDFs from data/samples/*.md")
    p.add_argument("--sample-dir", default="data/samples")
    p.add_argument("--corpus-dir", default="data/corpus")
    p.set_defaults(func=cmd_samples)

    p = sub.add_parser("ingest", help="extract, chunk, embed and index the corpus")
    _add_common(p)
    p.add_argument("--corpus-dir", default=None)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="answer one question")
    _add_common(p)
    p.add_argument("question")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("propose", help="propose candidate gold questions for curation")
    _add_common(p)
    p.add_argument("-n", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="data/gold/candidates.yaml")
    p.set_defaults(func=cmd_propose)

    p = sub.add_parser("evaluate", help="run the gold set through the pipeline")
    _add_common(p)
    p.add_argument("--gold", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("experiment", help="run a sweep and emit the comparison table")
    p.add_argument("--sweep", default="configs/experiments/sweep.yaml")
    p.add_argument("--gold", default=None)
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser("injection", help="run the prompt-injection suite")
    _add_common(p)
    p.add_argument("--set-path", default=None, dest="set_path")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_injection)

    p = sub.add_parser("gold-stats", help="validate and summarise a gold set")
    p.add_argument("--gold", default="data/gold/gold_set.yaml")
    p.set_defaults(func=cmd_gold_stats)

    p = sub.add_parser("report", help="assemble RESULTS.md from results/")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="RESULTS.md")
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
