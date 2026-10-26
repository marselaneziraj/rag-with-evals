from __future__ import annotations

import base64

import pytest

from ragevals.safety.injection_suite import STAGES, load_injection_set, run_injection_suite


def test_shipped_injection_set_is_well_formed(repo_root):
    docs = load_injection_set(repo_root / "data" / "injection" / "injection_docs.yaml")
    assert len(docs) == 15
    assert len({d.id for d in docs}) == 15
    assert len({d.canary for d in docs}) == 15
    assert len({d.family for d in docs}) >= 10, "attack families must be varied"
    for doc in docs:
        assert doc.carrier_question.strip().endswith("?")
        decoded = " ".join(
            base64.b64decode(tok + "=" * (-len(tok) % 4)).decode("utf-8", "replace")
            for tok in doc.text.split() if len(tok) >= 24 and tok.rstrip("=").isalnum())
        assert (doc.canary in doc.text or doc.canary in decoded
                or doc.family == "prompt_leak"), f"doc {doc.id} carries no canary"


def test_stages_are_cumulative():
    seen = set()
    for _, patch in STAGES:
        enabled = {k for k, v in patch.items() if v}
        assert seen <= enabled, "each stage must only add defences"
        seen = enabled


def test_suite_runs_and_mitigation_helps(cfg, chunks, tmp_path, repo_root):
    cfg = cfg.merged({"retrieval": {"top_k": 3, "candidate_k": 6}})
    docs = load_injection_set(repo_root / "data" / "injection" / "injection_docs.yaml")[:5]
    report = run_injection_suite(cfg, docs, chunks, results_dir=tmp_path / "inj")
    assert len(report.stages) == len(STAGES)
    assert report.before >= report.after, "defences must not make things worse"
    assert (tmp_path / "inj" / "injection_summary.json").exists()
    assert (tmp_path / "inj" / "injection.png").exists()


def test_denominator_is_retrieved_documents(cfg, chunks, tmp_path, repo_root):
    docs = load_injection_set(repo_root / "data" / "injection" / "injection_docs.yaml")[:3]
    report = run_injection_suite(cfg, docs, chunks, results_dir=None)
    for stage in report.stages:
        assert stage["retrieved"] <= stage["n"]
        assert stage["successes"] <= stage["retrieved"]


def test_duplicate_canaries_are_rejected(tmp_path):
    path = tmp_path / "inj.yaml"
    path.write_text(
        "documents:\n"
        "  - {id: '1', family: a, carrier_question: 'q?', canary: X, text: 'X'}\n"
        "  - {id: '2', family: b, carrier_question: 'q?', canary: X, text: 'X'}\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_injection_set(path)
