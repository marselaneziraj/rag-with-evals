from __future__ import annotations

import json

from ragevals.config import Config
from ragevals.experiments.plots import plot_comparison
from ragevals.experiments.runner import ExperimentRunner


def _base(cfg, repo_root, tmp_path) -> Config:
    base = cfg.merged({"paths": {"corpus_dir": str(repo_root / "data" / "samples"),
                                 "gold_set": str(tmp_path / "gold.yaml"),
                                 "results_dir": str(tmp_path / "results")},
                       "retrieval": {"top_k": 3, "candidate_k": 6}})
    (tmp_path / "gold.yaml").write_text(
        "items:\n"
        "  - id: g1\n"
        "    question: Within how many hours must a breach be notified?\n"
        "    reference_answer: Within 72 hours of becoming aware of it.\n"
        "    answerable: true\n"
        "    gold_sources: [{source: gdpr_essentials.md, page: 3}]\n"
        "  - id: g2\n"
        "    question: What is the retention period for CCTV at the depot?\n"
        "    reference_answer: ''\n"
        "    answerable: false\n", encoding="utf-8")
    return base


def test_runner_compares_two_configurations(cfg, repo_root, tmp_path):
    base = _base(cfg, repo_root, tmp_path)
    runner = ExperimentRunner(base=base, experiments=[
        ("bm25", {"retrieval": {"mode": "bm25"}}),
        ("hybrid", {"retrieval": {"mode": "hybrid"}}),
    ])
    runner.run(progress=False)
    assert len(runner.reports) == 2
    rows = runner.comparison_rows()
    assert {r["config"] for r in rows} == {"bm25", "hybrid"}
    assert len({r["fingerprint"] for r in rows}) == 2, "configs must differ"
    assert all("faithfulness" in r for r in rows)


def test_runner_writes_table_figure_and_json(cfg, repo_root, tmp_path):
    base = _base(cfg, repo_root, tmp_path)
    runner = ExperimentRunner(base=base,
                              experiments=[("only", {"retrieval": {"mode": "hybrid"}})])
    runner.run(progress=False)
    out = runner.save(tmp_path / "results")
    for key in ("csv", "markdown", "json", "figure"):
        assert out[key].exists()
    table = out["markdown"].read_text(encoding="utf-8")
    assert "| only |" in table
    assert json.loads(out["json"].read_text(encoding="utf-8"))


def test_markdown_table_shape(cfg, repo_root, tmp_path):
    base = _base(cfg, repo_root, tmp_path)
    runner = ExperimentRunner(base=base,
                              experiments=[("a", {}), ("b", {"retrieval": {"top_k": 2}})])
    runner.run(progress=False)
    lines = runner.to_markdown().splitlines()
    assert len(lines) == 4                       # header, divider, two rows
    assert lines[0].count("|") == lines[2].count("|")


def test_plot_handles_zero_cost(tmp_path):
    rows = [{"config": "a", "faithfulness": 1.0, "context_precision": 0.5,
             "context_recall": 0.5, "answer_correctness": 0.5, "refusal_accuracy": 0.5,
             "p95_latency_s": 0.1, "cost_per_query_usd": 0.0}]
    path = plot_comparison(rows, tmp_path / "cmp.png")
    assert path.exists() and path.stat().st_size > 5000
