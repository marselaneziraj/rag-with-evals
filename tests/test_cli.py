from __future__ import annotations

import pytest

from ragevals.cli import _coerce, build_parser, main


def test_parser_exposes_every_command():
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # noqa: SLF001
    assert {"samples", "ingest", "ask", "propose", "evaluate", "experiment",
            "injection", "gold-stats", "report"} <= set(commands)


@pytest.mark.parametrize("raw,expected", [
    ("5", 5), ("0.5", 0.5), ("true", True), ("false", False), ("none", None),
    ("hybrid", "hybrid"),
])
def test_override_coercion(raw, expected):
    assert _coerce(raw) == expected


def test_gold_stats_command(repo_root, capsys):
    code = main(["gold-stats", "--gold", str(repo_root / "data" / "gold" / "gold_set.yaml")])
    assert code == 0
    assert "\"unanswerable\": 20" in capsys.readouterr().out


def test_missing_corpus_exits_cleanly(tmp_path, repo_root):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"name: t\npaths:\n  corpus_dir: {tmp_path / 'nope'}\n", encoding="utf-8")
    assert main(["ingest", "-c", str(cfg)]) == 2
