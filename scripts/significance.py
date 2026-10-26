#!/usr/bin/env python
"""How much of a difference between two configurations is real?

An 80-item gold set cannot resolve small differences, and a comparison table
invites you to forget that. This script puts an interval on a single run and a
*paired* interval on the difference between two runs, which is the comparison
you actually care about: both configurations answer the same questions, so
pairing removes the between-question variance that dominates the spread.

    python scripts/significance.py --results results                 # one run
    python scripts/significance.py --results results \\
        --compare hybrid-k5 dense-only --metric answer_correctness

Method: Wilson score interval for proportions (refusal accuracy), percentile
bootstrap with 20 000 resamples for means and for paired differences. No
normality assumption, which matters because per-item metric distributions here
are bounded and lumpy.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

PROPORTIONS = {"refusal_accuracy", "citation_coverage"}


def load_records(results_dir: Path, name: str) -> dict[str, dict]:
    matches = sorted(glob.glob(str(results_dir / "runs" / f"{name}-*" / "records.jsonl")))
    if not matches:
        raise SystemExit(f"no run named {name!r} under {results_dir}/runs/")
    rows = [json.loads(line) for line in
            Path(matches[-1]).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["id"]: row for row in rows}


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def bootstrap_mean(values: Sequence[float], rounds: int = 20000,
                   seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    idx = rng.integers(0, arr.size, size=(rounds, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _values(records: dict[str, dict], metric: str,
            ids: Sequence[str] | None = None) -> list[float]:
    ids = ids if ids is not None else list(records)
    return [float(records[i][metric]) for i in ids
            if records[i].get(metric) is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results", type=Path)
    parser.add_argument("--run", default=None, help="run name for a single-run interval")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None)
    parser.add_argument("--metric", default="answer_correctness")
    args = parser.parse_args()

    if args.compare:
        a_name, b_name = args.compare
        a, b = load_records(args.results, a_name), load_records(args.results, b_name)
        shared = [i for i in a if i in b
                  and a[i].get(args.metric) is not None
                  and b[i].get(args.metric) is not None]
        diffs = np.array([float(a[i][args.metric]) - float(b[i][args.metric])
                          for i in shared])
        lo, hi = bootstrap_mean(diffs)
        verdict = ("the interval excludes zero: the difference is resolvable "
                   "at this sample size"
                   if lo > 0 or hi < 0 else
                   "the interval spans zero: this gold set cannot tell these two "
                   "configurations apart")
        print(f"{args.metric}: {a_name} - {b_name} on {len(shared)} paired items")
        print(f"  mean difference {diffs.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  {verdict}")
        return 0

    name = args.run
    if name is None:
        runs = sorted(glob.glob(str(args.results / "runs" / "*")))
        if not runs:
            raise SystemExit(f"no runs under {args.results}/runs/")
        name = Path(runs[-1]).name.rsplit("-", 1)[0]
    records = load_records(args.results, name)
    n = len(records)
    print(f"run {name}: {n} gold items")

    correct = sum(1 for r in records.values() if r["refusal_correct"])
    lo, hi = wilson(correct, n)
    print(f"  refusal_accuracy      {correct}/{n} = {correct / n:.3f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]  (Wilson)")

    for metric in ["faithfulness", "context_precision", "context_recall",
                   "answer_correctness"]:
        values = _values(records, metric)
        if not values:
            continue
        lo, hi = bootstrap_mean(values)
        print(f"  {metric:21s} n={len(values):3d} mean {np.mean(values):.3f}  "
              f"95% CI [{lo:.3f}, {hi:.3f}]  (bootstrap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
