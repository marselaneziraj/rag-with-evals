"""Gold-set schema, loading and validation.

A gold set is a dataset and deserves dataset discipline: a stable id per item,
a provenance field, an explicit ``answerable`` flag, and -- where you were
willing to annotate it -- the (source, page) pairs that contain the answer.
Annotated ``gold_sources`` turn context precision/recall from a noisy lexical
proxy into a direct measurement, which is the single highest-leverage hour of
manual work in the whole project.

Validation is strict and fails loudly: a silent typo in ``answerable`` corrupts
refusal accuracy, which is the metric most likely to be quoted in a review.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOWED_DIFFICULTY = {"easy", "medium", "hard"}


@dataclass
class GoldItem:
    id: str
    question: str
    reference_answer: str = ""
    answerable: bool = True
    gold_sources: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    notes: str = ""

    @property
    def gold_keys(self) -> set[tuple[str, int]]:
        keys: set[tuple[str, int]] = set()
        for ref in self.gold_sources:
            source = str(ref.get("source", "")).strip().lower()
            page = ref.get("page")
            if source and page is not None:
                keys.add((source, int(page)))
        return keys

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "question": self.question,
                "reference_answer": self.reference_answer,
                "answerable": self.answerable, "gold_sources": self.gold_sources,
                "tags": self.tags, "difficulty": self.difficulty, "notes": self.notes}


def validate_gold_set(items: Sequence[GoldItem]) -> list[str]:
    """Return a list of problems; empty means the set is well-formed."""
    problems: list[str] = []
    seen: set[str] = set()
    for i, item in enumerate(items):
        where = f"item {i} (id={item.id!r})"
        if not item.id:
            problems.append(f"{where}: missing id")
        elif item.id in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(item.id)
        if not item.question.strip():
            problems.append(f"{where}: empty question")
        if item.answerable and not item.reference_answer.strip():
            problems.append(f"{where}: answerable item needs a reference_answer")
        if not item.answerable and item.gold_sources:
            problems.append(f"{where}: unanswerable item must not list gold_sources")
        if item.difficulty not in ALLOWED_DIFFICULTY:
            problems.append(f"{where}: difficulty must be one of {sorted(ALLOWED_DIFFICULTY)}")
        for ref in item.gold_sources:
            if "source" not in ref or "page" not in ref:
                problems.append(f"{where}: gold_sources entries need 'source' and 'page'")
    unanswerable = sum(1 for i in items if not i.answerable)
    if items and unanswerable == 0:
        problems.append("gold set contains no unanswerable items: refusal metrics "
                        "will be undefined and the system cannot be shown to refuse")
    return problems


def load_gold_set(path: str | Path, strict: bool = True) -> list[GoldItem]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    rows = raw.get("items", raw if isinstance(raw, list) else [])
    items: list[GoldItem] = []
    for row in rows:
        items.append(GoldItem(
            id=str(row.get("id", "")),
            question=str(row.get("question", "")),
            reference_answer=str(row.get("reference_answer", "") or ""),
            answerable=bool(row.get("answerable", True)),
            gold_sources=list(row.get("gold_sources", []) or []),
            tags=list(row.get("tags", []) or []),
            difficulty=str(row.get("difficulty", "medium")),
            notes=str(row.get("notes", "") or ""),
        ))
    problems = validate_gold_set(items)
    if problems and strict:
        raise ValueError("gold set validation failed:\n  - " + "\n  - ".join(problems))
    return items


def gold_set_stats(items: Sequence[GoldItem]) -> dict[str, Any]:
    tags: dict[str, int] = {}
    for item in items:
        for tag in item.tags:
            tags[tag] = tags.get(tag, 0) + 1
    return {
        "n": len(items),
        "answerable": sum(1 for i in items if i.answerable),
        "unanswerable": sum(1 for i in items if not i.answerable),
        "annotated_with_gold_sources": sum(1 for i in items if i.gold_sources),
        "by_difficulty": {d: sum(1 for i in items if i.difficulty == d)
                          for d in sorted(ALLOWED_DIFFICULTY)},
        "by_tag": dict(sorted(tags.items())),
    }


def save_gold_set(items: Sequence[GoldItem], path: str | Path, header: str = "") -> None:
    payload = {"items": [i.to_dict() for i in items]}
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)
    Path(path).write_text((header + "\n" if header else "") + text, encoding="utf-8")
