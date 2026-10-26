"""LLM-as-judge with a fixed rubric at temperature 0.

Why a judge at all
------------------
Lexical overlap cannot tell you that "the controller must notify within 72
hours" and "notification is due no later than three days" agree. A judge can.
That capability costs you determinism, so every design choice here is about
buying it back:

* **Fixed rubric, loaded from YAML and hashed into the results.** A score is
  meaningless without the rubric that produced it; changing the rubric is a new
  experiment, not a re-run.
* **Temperature 0 and a fixed seed.** Not a guarantee of determinism with a
  hosted model, but it removes the sampling variance you can control.
* **Integer 0-4, not a float.** Judges are poor at fine-grained scoring and
  good at ordering; a coarse scale with explicit anchors is more reproducible.
  The value is normalised to [0, 1] only at aggregation time.
* **JSON-only output, parsed strictly, with a lexical fallback on parse
  failure.** A judge that occasionally returns prose must not crash a 4-hour
  sweep.
* **Disk cache keyed by (metric, rubric hash, model, question, answer).** Judge
  calls dominate evaluation cost; re-running a sweep where only the retriever
  changed should not re-pay for unchanged pairs.

Known limitation, stated up front: an LLM judge is biased towards verbose and
self-confident answers, and a judge from the same family as the generator
flatters it. Keep the judge model fixed across configurations you intend to
compare, report which model judged, and spot-check 10-20 items by hand.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from ..config import Config
from ..generation.base import LLMBackend
from ..generation.prompts import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from ..logging_utils import get_logger
from .metrics import MetricResult, answer_correctness_lexical, token_recall

log = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

DEFAULT_RUBRIC: dict[str, str] = {
    "answer_correctness": (
        "4 = states every fact in the reference, no contradictions, no additions\n"
        "3 = states the main fact correctly; a minor detail missing or imprecise\n"
        "2 = partially correct; a material omission or a vague hedge\n"
        "1 = related to the question but the key fact is wrong or absent\n"
        "0 = contradicts the reference, is off-topic, or refuses when the "
        "reference answers"
    ),
    "faithfulness": (
        "4 = every claim is directly supported by the retrieved context\n"
        "3 = one claim is a reasonable paraphrase that slightly overreaches\n"
        "2 = a claim is not in the context but is plausible\n"
        "1 = a claim contradicts the context\n"
        "0 = the answer is largely unsupported by the context"
    ),
}


class Judge:
    """Scores (question, reference, candidate, context) -> [0, 1]."""

    def __init__(self, backend: LLMBackend | None = None,
                 rubric: dict[str, str] | None = None,
                 cache_dir: str | Path | None = None, enabled: bool = True,
                 model_label: str = "heuristic") -> None:
        self.backend = backend
        self.rubric = {**DEFAULT_RUBRIC, **(rubric or {})}
        self.enabled = enabled and backend is not None
        self.model_label = model_label
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.cache_hits = 0
        self.parse_failures = 0
        self.cost_usd = 0.0

    # ------------------------------------------------------------------ #
    @property
    def rubric_hash(self) -> str:
        payload = json.dumps(self.rubric, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

    def _cache_key(self, metric: str, question: str, candidate: str,
                   reference: str) -> str:
        blob = f"{metric}|{self.rubric_hash}|{self.model_label}|{question}|{reference}|{candidate}"
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def _cached(self, key: str) -> dict[str, Any] | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                self.cache_hits += 1
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _store(self, key: str, payload: dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        try:
            (self.cache_dir / f"{key}.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    def score(self, metric: str, question: str, reference: str, candidate: str,
              context: str = "") -> MetricResult:
        if not self.enabled:
            if metric == "faithfulness":
                value = token_recall(candidate, context)
                return MetricResult(value, {"mode": "heuristic"})
            return answer_correctness_lexical(candidate, reference)

        key = self._cache_key(metric, question, candidate, reference)
        cached = self._cached(key)
        if cached is not None:
            return MetricResult(float(cached["value"]),
                                {**cached.get("detail", {}), "cached": True})

        prompt = build_judge_prompt(metric, self.rubric.get(metric, ""), question,
                                    reference, candidate, context)
        try:
            response = self.backend.generate(prompt, system=JUDGE_SYSTEM_PROMPT,
                                             temperature=0.0)
            self.calls += 1
            self.cost_usd += response.cost_usd
            parsed = self._parse(response.text)
        except Exception as exc:  # noqa: BLE001 - never let the judge kill a sweep
            log.warning("judge call failed (%s); falling back to lexical scoring",
                        type(exc).__name__)
            return answer_correctness_lexical(candidate, reference)

        if parsed is None:
            self.parse_failures += 1
            log.warning("judge returned unparseable output; falling back to lexical")
            return answer_correctness_lexical(candidate, reference)

        score, reason = parsed
        value = max(0.0, min(1.0, score / 4.0))
        detail = {"raw_score": score, "reason": reason, "mode": "llm-judge",
                  "judge_model": self.model_label, "rubric_hash": self.rubric_hash,
                  "cost_usd": response.cost_usd}
        self._store(key, {"value": value, "detail": detail})
        return MetricResult(value, detail)

    @staticmethod
    def _parse(text: str) -> tuple[int, str] | None:
        match = _JSON_RE.search(text or "")
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            score = int(round(float(data["score"])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        return max(0, min(4, score)), str(data.get("reason", ""))[:200]

    def stats(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "model": self.model_label,
                "rubric_hash": self.rubric_hash, "calls": self.calls,
                "cache_hits": self.cache_hits, "parse_failures": self.parse_failures,
                "cost_usd": round(self.cost_usd, 6)}


def load_rubric(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in (data.get("rubrics", data) or {}).items()}


def build_judge(cfg: Config, generator: LLMBackend | None = None) -> Judge:
    """Construct the judge described by ``evaluation.judge``."""
    jc = cfg.evaluation.judge
    rubric = load_rubric(jc.rubric)
    cache_dir = Path(cfg.paths.cache_dir) / "judge" if jc.cache else None

    if jc.backend == "heuristic":
        return Judge(None, rubric, cache_dir, enabled=False, model_label="heuristic")
    if jc.backend == "same-as-generator":
        if generator is None or generator.name == "offline":
            log.info("judge: generator backend is offline/absent -> heuristic judge")
            return Judge(None, rubric, cache_dir, enabled=False, model_label="heuristic")
        return Judge(generator, rubric, cache_dir, enabled=True,
                     model_label=f"{generator.name}:{generator.model}")

    judge_cfg = cfg.merged({"generation": {"backend": jc.backend, "model": jc.model,
                                           "temperature": 0.0}})
    from ..generation.factory import build_backend

    backend = build_backend(judge_cfg)
    if backend.name == "offline":
        return Judge(None, rubric, cache_dir, enabled=False, model_label="heuristic")
    return Judge(backend, rubric, cache_dir, enabled=True,
                 model_label=f"{backend.name}:{backend.model}")
