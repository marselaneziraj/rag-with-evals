"""Typed configuration.

Design notes
------------
* One YAML file fully describes an experiment: ingestion, retrieval, generation,
  safety and evaluation. That is what makes a run reproducible -- the config is
  hashed into the results so a number can always be traced back to its settings.
* ``${VAR}`` / ``${VAR:default}`` in any string is expanded from the environment,
  so secrets never live in the repo.
* Unknown keys raise. A typo in a sweep file would otherwise silently evaluate
  the default configuration and you would compare a config against itself.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
@dataclass
class PathsConfig:
    corpus_dir: str = "data/corpus"
    sample_dir: str = "data/samples"
    index_dir: str = "artifacts/index"
    cache_dir: str = "artifacts/cache"
    results_dir: str = "results"
    gold_set: str = "data/gold/gold_set.yaml"
    injection_set: str = "data/injection/injection_docs.yaml"


@dataclass
class IngestConfig:
    chunker: str = "section"           # fixed | sentence | section
    chunk_size: int = 900              # characters (see docs/METHODOLOGY.md)
    chunk_overlap: int = 150
    min_chunk_chars: int = 120
    layout_aware: bool = True          # use font metrics to recover headings
    drop_repeated_headers: bool = True # kill running headers/footers
    heading_scale: float = 1.12        # font size ratio that marks a heading


@dataclass
class EmbeddingConfig:
    backend: str = "auto"              # auto | sentence-transformers | hashing
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 512                     # only used by the hashing fallback
    batch_size: int = 32


@dataclass
class StoreConfig:
    backend: str = "auto"              # auto | chroma | faiss | numpy
    collection: str = "ragevals"


@dataclass
class RetrievalConfig:
    mode: str = "hybrid"               # dense | bm25 | hybrid
    top_k: int = 5                     # chunks handed to the generator
    candidate_k: int = 20              # chunks pulled before reranking
    rrf_k: int = 60                    # reciprocal-rank-fusion damping
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    reranker: str = "none"             # none | lexical | cross-encoder
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cache: bool = True
    min_score: float = 0.0             # below this, context is treated as empty


@dataclass
class GenerationConfig:
    backend: str = "offline"           # offline | ollama | openai
    model: str = "offline-extractive"
    temperature: float = 0.0
    max_tokens: int = 700
    base_url: str = "${OLLAMA_HOST:http://localhost:11434}"
    api_key_env: str = "OPENAI_API_KEY"
    api_base: str = "${OPENAI_BASE_URL:https://api.openai.com/v1}"
    timeout_s: float = 120.0
    price_per_1k_input: float = 0.0
    price_per_1k_output: float = 0.0
    simulate_injection_susceptibility: bool = True
    coverage_threshold: float = 0.34   # offline backend: abstain below this
    max_sentences: int = 3             # offline backend: answer length cap


@dataclass
class SafetyConfig:
    delimit_context: bool = True       # wrap chunks in typed, id-tagged blocks
    instruction_hierarchy: bool = True # system prompt states data != instructions
    sanitize_context: bool = True      # neutralise imperative text inside chunks
    validate_output: bool = True       # reject unsupported / uncited answers
    support_threshold: float = 0.25    # lexical support a sentence must reach
    max_context_chars: int = 12000


@dataclass
class JudgeConfig:
    backend: str = "heuristic"         # heuristic | same-as-generator | openai | ollama
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    rubric: str = "configs/judge_rubric.yaml"
    cache: bool = True


@dataclass
class EvalConfig:
    use_ragas: bool = False            # opt-in; fallback metrics are the default
    faithfulness_threshold: float = 0.30
    correctness_threshold: float = 0.60
    refusal_marker: str = "INSUFFICIENT_CONTEXT"
    judge: JudgeConfig = field(default_factory=JudgeConfig)


@dataclass
class Config:
    name: str = "default"
    seed: int = 20260101
    paths: PathsConfig = field(default_factory=PathsConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # ---------------- helpers ---------------- #
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        """Stable short hash of the whole config: stamped onto every result row."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def merged(self, patch: Mapping[str, Any]) -> Config:
        base = self.to_dict()
        return _from_dict(Config, _deep_merge(base, dict(patch)))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")
        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _deep_merge(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _from_dict(cls: type[T], data: Mapping[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"unknown config key(s) for {cls.__name__}: {sorted(unknown)}; "
            f"allowed: {sorted(known)}"
        )
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        if is_dataclass(f.type) or (isinstance(f.type, type) and is_dataclass(f.type)):
            kwargs[name] = _from_dict(f.type, value or {})
        elif name in {"paths", "ingest", "embedding", "store", "retrieval",
                      "generation", "safety", "evaluation", "judge"}:
            sub = {"paths": PathsConfig, "ingest": IngestConfig, "embedding": EmbeddingConfig,
                   "store": StoreConfig, "retrieval": RetrievalConfig,
                   "generation": GenerationConfig, "safety": SafetyConfig,
                   "evaluation": EvalConfig, "judge": JudgeConfig}[name]
            kwargs[name] = _from_dict(sub, value or {})
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[arg-type]


def load_config(path: str | Path | None = None,
                overrides: Mapping[str, Any] | None = None) -> Config:
    """Load ``path`` (YAML) over the dataclass defaults, then apply ``overrides``."""
    data: dict[str, Any] = {}
    if path is not None:
        raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top level of a config must be a mapping")
    if overrides:
        data = _deep_merge(data, overrides)
    data = _expand(data)
    data.pop("experiments", None)   # sweep files may carry this; handled elsewhere
    data.pop("extends", None)
    return _from_dict(Config, data)


def load_sweep(path: str | Path) -> tuple[Config, list[tuple[str, dict[str, Any]]]]:
    """Read a sweep file: a base config plus a list of named patches."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    patches = raw.pop("experiments", []) or []
    base = load_config(None, raw)
    out: list[tuple[str, dict[str, Any]]] = []
    for item in patches:
        item = dict(item)
        name = item.pop("name", None)
        if not name:
            raise ValueError(f"{path}: every entry under 'experiments' needs a 'name'")
        out.append((name, _expand(item)))
    if not out:
        raise ValueError(f"{path}: no 'experiments' entries found")
    return base, out
