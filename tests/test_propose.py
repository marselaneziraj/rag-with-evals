from __future__ import annotations

import yaml

from ragevals.tools.propose import propose_questions


def test_proposals_are_drafts_with_provenance(cfg, chunks, tmp_path):
    out = tmp_path / "candidates.yaml"
    items = propose_questions(cfg, n=8, out_path=out, chunks=chunks)
    # Three chunks cannot yield eight distinct questions; the proposer under-
    # produces and warns rather than inventing questions with no provenance.
    assert 0 < len(items) <= 8
    assert out.exists()

    answerable = [i for i in items if i["answerable"]]
    unanswerable = [i for i in items if not i["answerable"]]
    assert unanswerable, "a proposal set without negatives cannot test refusal"
    for item in answerable:
        assert item["gold_sources"], "every draft must carry the page it came from"
        assert item["status"] == "DRAFT"
        assert "DRAFT" in item["reference_answer"]
    for item in unanswerable:
        assert item["gold_sources"] == []


def test_ids_are_unique_and_sequential(cfg, chunks, tmp_path):
    items = propose_questions(cfg, n=10, out_path=tmp_path / "c.yaml", chunks=chunks)
    ids = [i["id"] for i in items]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_output_is_valid_yaml_with_a_warning_header(cfg, chunks, tmp_path):
    out = tmp_path / "c.yaml"
    items = propose_questions(cfg, n=6, out_path=out, chunks=chunks)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("#")
    assert "not a gold set" in text.lower()
    assert len(yaml.safe_load(text)["items"]) == len(items)


def test_proposals_are_deterministic(cfg, chunks, tmp_path):
    a = propose_questions(cfg, n=6, seed=3, chunks=chunks)
    b = propose_questions(cfg, n=6, seed=3, chunks=chunks)
    assert [i["question"] for i in a] == [i["question"] for i in b]
