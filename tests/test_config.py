from __future__ import annotations

import pytest

from ragevals.config import Config, load_config, load_sweep


def test_defaults_are_complete():
    cfg = Config()
    assert cfg.ingest.chunker == "section"
    assert cfg.retrieval.mode == "hybrid"
    assert cfg.generation.backend == "offline"


def test_fingerprint_is_stable_and_sensitive():
    a, b = Config(), Config()
    assert a.fingerprint() == b.fingerprint()
    c = a.merged({"retrieval": {"top_k": 9}})
    assert c.fingerprint() != a.fingerprint()
    assert c.retrieval.top_k == 9
    assert a.retrieval.top_k == 5, "merged() must not mutate the original"


def test_merge_is_deep():
    cfg = Config().merged({"retrieval": {"top_k": 7}})
    assert cfg.retrieval.top_k == 7
    assert cfg.retrieval.mode == "hybrid", "unpatched sibling keys must survive"


def test_unknown_key_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("retrieval:\n  top_kk: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(path)


def test_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGEVALS_TEST_MODEL", "my-model")
    path = tmp_path / "c.yaml"
    path.write_text("generation:\n  model: ${RAGEVALS_TEST_MODEL}\n"
                    "  api_base: ${RAGEVALS_MISSING:https://fallback}\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.generation.model == "my-model"
    assert cfg.generation.api_base == "https://fallback"


def test_real_configs_load(repo_root):
    for name in ["default.yaml", "ollama.yaml", "openai.yaml"]:
        cfg = load_config(repo_root / "configs" / name)
        assert cfg.name


def test_sweep_file_parses(repo_root):
    base, experiments = load_sweep(repo_root / "configs" / "experiments" / "sweep.yaml")
    assert len(experiments) >= 5
    names = [n for n, _ in experiments]
    assert len(names) == len(set(names)), "experiment names must be unique"
    for _, patch in experiments:
        base.merged(patch)          # every patch must produce a valid config
