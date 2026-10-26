"""The end-to-end RAG pipeline: retrieve -> sanitise -> generate -> validate.

The ordering matters and is the whole design:

* **Refuse before generating.** If retrieval returns nothing above
  ``retrieval.min_score``, the pipeline refuses without calling the model. That
  is a correctness property (no context, no answer), a cost property (an
  unanswerable question should cost nothing) and a latency property.
* **Sanitise between retrieval and the prompt**, because that is the only point
  where untrusted text is still a string we control rather than tokens the
  model has already read.
* **Validate after generation**, because the generator is the component we
  trust least. An answer that fails validation is downgraded to a refusal --
  the system's failure mode is silence, not confident invention.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .citations import Citation, validate_citations
from .config import Config
from .generation.base import LLMBackend
from .generation.factory import build_backend
from .generation.prompts import ANSWER_SYSTEM_PROMPT, REFUSAL_MARKER, build_answer_prompt
from .ingest.chunking import Chunk
from .logging_utils import get_logger
from .retrieval.retriever import Retriever, build_retriever
from .safety.output_validation import validate_answer
from .safety.sanitize import sanitize_chunks
from .store.base import SearchHit

log = get_logger(__name__)


@dataclass
class AnswerResult:
    question: str
    answer: str
    refused: bool = False
    hits: list[SearchHit] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    valid_citations: int = 0
    invalid_citations: int = 0
    retrieval_latency_s: float = 0.0
    generation_latency_s: float = 0.0
    total_latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    backend: str = ""
    model: str = ""
    sanitizer_flags: list[str] = field(default_factory=list)
    validation_reasons: list[str] = field(default_factory=list)
    blocked_by_validator: bool = False
    raw_answer: str = ""
    config_fingerprint: str = ""

    @property
    def contexts(self) -> list[str]:
        return [h.chunk.text for h in self.hits]

    @property
    def chunks(self) -> list[Chunk]:
        return [h.chunk for h in self.hits]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question, "answer": self.answer, "refused": self.refused,
            "citations": [str(c) for c in self.citations],
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "retrieval_latency_s": round(self.retrieval_latency_s, 4),
            "generation_latency_s": round(self.generation_latency_s, 4),
            "total_latency_s": round(self.total_latency_s, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd, "backend": self.backend, "model": self.model,
            "sanitizer_flags": self.sanitizer_flags,
            "validation_reasons": self.validation_reasons,
            "blocked_by_validator": self.blocked_by_validator,
            "raw_answer": self.raw_answer,
            "config_fingerprint": self.config_fingerprint,
            "retrieved": [h.to_dict() for h in self.hits],
        }


class RAGPipeline:
    def __init__(self, cfg: Config, retriever: Retriever, backend: LLMBackend) -> None:
        self.cfg = cfg
        self.retriever = retriever
        self.backend = backend
        self.system_prompt = (ANSWER_SYSTEM_PROMPT if cfg.safety.instruction_hierarchy
                              else _minimal_system_prompt())

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: Config, chunks: Sequence[Chunk] | None = None,
                    retriever: Retriever | None = None,
                    backend: LLMBackend | None = None) -> RAGPipeline:
        retriever = retriever or build_retriever(cfg, chunks)
        backend = backend or build_backend(cfg)
        return cls(cfg, retriever, backend)

    # ------------------------------------------------------------------ #
    def answer(self, question: str) -> AnswerResult:
        started = time.perf_counter()
        fingerprint = self.cfg.fingerprint()
        retrieval = self.retriever.retrieve(question)
        result = AnswerResult(question=question, answer="", hits=list(retrieval.hits),
                              retrieval_latency_s=retrieval.latency_s,
                              backend=self.backend.name, model=self.backend.model,
                              config_fingerprint=fingerprint)

        if not retrieval.hits:
            result.answer = REFUSAL_MARKER
            result.refused = True
            result.validation_reasons = ["no_context_retrieved"]
            result.total_latency_s = time.perf_counter() - started
            return result

        chunks = [h.chunk for h in retrieval.hits]
        if self.cfg.safety.sanitize_context:
            chunks, flags = sanitize_chunks(chunks)
            result.sanitizer_flags = flags

        prompt = build_answer_prompt(
            question, chunks,
            max_context_chars=self.cfg.safety.max_context_chars,
            delimit=self.cfg.safety.delimit_context)

        response = self.backend.generate(prompt, system=self.system_prompt)
        result.generation_latency_s = response.latency_s
        result.prompt_tokens = response.prompt_tokens
        result.completion_tokens = response.completion_tokens
        result.cost_usd = response.cost_usd
        result.raw_answer = response.text
        answer = response.text.strip()

        if self.cfg.safety.validate_output:
            validation = validate_answer(
                answer, chunks, system_prompt=self.system_prompt,
                support_threshold=self.cfg.safety.support_threshold,
                require_citations=True)
            if not validation.ok:
                result.blocked_by_validator = True
                result.validation_reasons = validation.reasons
                answer = validation.answer

        result.answer = answer
        result.refused = answer.strip().startswith(REFUSAL_MARKER)
        report = validate_citations(answer, chunks, require_per_sentence=False)
        result.citations = report.citations
        result.valid_citations = len(report.valid)
        result.invalid_citations = len(report.invalid)
        result.total_latency_s = time.perf_counter() - started
        return result

    # ------------------------------------------------------------------ #
    def describe(self) -> dict[str, Any]:
        return {"config": self.cfg.name, "fingerprint": self.cfg.fingerprint(),
                "retriever": self.retriever.describe(),
                "generator": self.backend.describe(),
                "safety": {"delimit": self.cfg.safety.delimit_context,
                           "hierarchy": self.cfg.safety.instruction_hierarchy,
                           "sanitize": self.cfg.safety.sanitize_context,
                           "validate": self.cfg.safety.validate_output}}


def _minimal_system_prompt() -> str:
    """Ablation: the citation contract without the instruction hierarchy.

    Used by the injection experiment to isolate how much the hierarchy itself
    is worth, separately from sanitisation and output validation.
    """
    return ("You are a careful question-answering assistant for regulatory and "
            "policy documents.\nAnswer the question using the supplied documents. "
            f"Cite as [source, p. N]. If the documents are insufficient, reply "
            f"exactly {REFUSAL_MARKER}.")
