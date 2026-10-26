from __future__ import annotations

import pytest
import yaml

from ragevals.evaluation.goldset import (
    GoldItem,
    gold_set_stats,
    load_gold_set,
    save_gold_set,
    validate_gold_set,
)


def _item(**kw):
    base = dict(id="g1", question="q?", reference_answer="a", answerable=True)
    base.update(kw)
    return GoldItem(**base)


def test_valid_set_has_no_problems():
    items = [_item(id="g1"), _item(id="g2", answerable=False, reference_answer="")]
    assert validate_gold_set(items) == []


def test_duplicate_ids_are_caught():
    problems = validate_gold_set([_item(id="g1"), _item(id="g1", answerable=False,
                                                        reference_answer="")])
    assert any("duplicate id" in p for p in problems)


def test_answerable_without_reference_is_caught():
    problems = validate_gold_set([_item(reference_answer=""),
                                  _item(id="g2", answerable=False, reference_answer="")])
    assert any("reference_answer" in p for p in problems)


def test_unanswerable_with_gold_sources_is_caught():
    bad = _item(id="g2", answerable=False, reference_answer="",
                gold_sources=[{"source": "a.pdf", "page": 1}])
    assert any("must not list gold_sources" in p for p in validate_gold_set([bad]))


def test_a_set_with_no_negatives_is_rejected():
    problems = validate_gold_set([_item()])
    assert any("no unanswerable items" in p for p in problems)


def test_bad_difficulty_is_caught():
    problems = validate_gold_set([_item(difficulty="brutal"),
                                  _item(id="g2", answerable=False, reference_answer="")])
    assert any("difficulty" in p for p in problems)


def test_gold_keys_are_normalised():
    item = _item(gold_sources=[{"source": "GDPR.pdf", "page": "3"}])
    assert item.gold_keys == {("gdpr.pdf", 3)}


def test_roundtrip(tmp_path):
    items = [_item(id="g1"), _item(id="g2", answerable=False, reference_answer="")]
    path = tmp_path / "gold.yaml"
    save_gold_set(items, path, header="# test")
    loaded = load_gold_set(path)
    assert [i.id for i in loaded] == ["g1", "g2"]


def test_strict_loading_raises(tmp_path):
    path = tmp_path / "gold.yaml"
    path.write_text(yaml.safe_dump({"items": [{"id": "g1", "question": "q?",
                                               "answerable": True}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_gold_set(path, strict=True)


# --------------------------------------------------------------------------- #
# The shipped gold set is part of the product and is tested as such.
# --------------------------------------------------------------------------- #
def test_shipped_gold_set_is_well_formed(repo_root):
    items = load_gold_set(repo_root / "data" / "gold" / "gold_set.yaml")
    stats = gold_set_stats(items)
    assert stats["n"] == 80
    assert stats["unanswerable"] == 20, "25% negatives by design"
    assert stats["annotated_with_gold_sources"] == stats["answerable"]
    assert len({i.question for i in items}) == stats["n"], "no duplicate questions"


def test_shipped_gold_annotations_point_at_real_pages(repo_root):
    items = load_gold_set(repo_root / "data" / "gold" / "gold_set.yaml")
    corpus = {p.stem for p in (repo_root / "data" / "samples").glob("*.md")}
    for item in items:
        for ref in item.gold_sources:
            assert ref["source"].rsplit(".", 1)[0] in corpus
            assert int(ref["page"]) >= 1
