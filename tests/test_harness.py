from __future__ import annotations

import json

from ragevals.embeddings import HashingEmbedder
from ragevals.evaluation.goldset import GoldItem
from ragevals.evaluation.harness import EvaluationHarness
from ragevals.evaluation.judge import Judge, build_judge
from ragevals.generation.offline import OfflineExtractiveBackend
from ragevals.pipeline import RAGPipeline
from ragevals.retrieval.retriever import Retriever
from ragevals.store.numpy_store import NumpyStore


def _harness(cfg, chunks, tmp_path):
    cfg = cfg.merged({"retrieval": {"top_k": 3, "candidate_k": 5}})
    embedder = HashingEmbedder(dim=256)
    store = NumpyStore(tmp_path / "idx", dim=256)
    store.add(chunks, embedder.encode([c.text for c in chunks]))
    pipeline = RAGPipeline(cfg, Retriever(cfg, chunks, store, embedder),
                           OfflineExtractiveBackend())
    return cfg, EvaluationHarness(cfg, pipeline, Judge(None, enabled=False))


GOLD = [
    GoldItem(id="g1", question="Within how many hours must a breach be notified?",
             reference_answer="Within 72 hours of becoming aware of it.",
             answerable=True, gold_sources=[{"source": "gdpr.pdf", "page": 3}]),
    GoldItem(id="g2", question="What is the maximum administrative fine?",
             reference_answer="Up to 20 000 000 EUR or 4 per cent of turnover.",
             answerable=True, gold_sources=[{"source": "gdpr.pdf", "page": 9}]),
    GoldItem(id="g3", question="What is the retention period for CCTV at the depot?",
             reference_answer="", answerable=False),
]


def test_harness_produces_one_record_per_item(cfg, chunks, tmp_path):
    _, harness = _harness(cfg, chunks, tmp_path)
    report = harness.run(GOLD, progress=False)
    assert len(report.records) == 3
    assert report.summary["n_items"] == 3
    assert report.summary["n_unanswerable"] == 1


def test_context_metrics_are_averaged_over_answerable_items_only(cfg, chunks, tmp_path):
    _, harness = _harness(cfg, chunks, tmp_path)
    report = harness.run(GOLD, progress=False)
    assert report.records[2]["context_precision"] is None
    assert 0.0 <= report.summary["context_precision"] <= 1.0


def test_summary_carries_full_provenance(cfg, chunks, tmp_path):
    _, harness = _harness(cfg, chunks, tmp_path)
    summary = harness.run(GOLD, progress=False).summary
    for key in ["config", "config_fingerprint", "judge", "pipeline", "metrics_engine",
                "environment", "timestamp_utc", "gold_set"]:
        assert key in summary
    assert summary["metrics_engine"] == "builtin"


def test_outputs_are_written(cfg, chunks, tmp_path):
    _, harness = _harness(cfg, chunks, tmp_path)
    run_dir = tmp_path / "run"
    report = harness.run(GOLD, run_dir=run_dir, progress=False)
    assert (run_dir / "records.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    rows = [json.loads(line) for line in
            (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert report.run_dir == run_dir


def test_refusal_decision_is_scored(cfg, chunks, tmp_path):
    _, harness = _harness(cfg, chunks, tmp_path)
    report = harness.run(GOLD, progress=False)
    unanswerable = report.records[2]
    assert unanswerable["expected_refusal"] is True
    assert unanswerable["answer_correctness"] in (0.0, 1.0)


def test_heuristic_judge_is_used_without_a_model(cfg):
    judge = build_judge(cfg, OfflineExtractiveBackend())
    assert not judge.enabled
    score = judge.score("answer_correctness", "q", "72 hours", "72 hours")
    assert score.value > 0.8


def test_judge_parses_strict_json():
    assert Judge._parse('{"score": 3, "reason": "ok"}') == (3, "ok")
    assert Judge._parse("prose with {\"score\": 4} inside")[0] == 4
    assert Judge._parse("no json here") is None
    assert Judge._parse('{"score": 99}')[0] == 4, "scores are clamped to the rubric"


def test_judge_rubric_hash_changes_with_the_rubric():
    a = Judge(None, rubric={"answer_correctness": "x"})
    b = Judge(None, rubric={"answer_correctness": "y"})
    assert a.rubric_hash != b.rubric_hash
