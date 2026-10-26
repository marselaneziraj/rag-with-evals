"""Offline deterministic backend -- the reason this repo runs with no API key.

What it is
----------
A scripted extractive "model". It parses the same prompt a real LLM would see,
scores each sentence of each retrieved document against the question with an
IDF-weighted overlap, emits the best two or three sentences with a correctly
formatted citation, and emits the refusal marker when coverage is too low.

Why it exists
-------------
Three reasons, in order of importance:
  1. CI must be able to run the *whole* evaluation -- metrics, experiment
     runner, injection suite -- on every commit, deterministically, for free.
  2. It is a floor. Any real generator that scores *below* this on
     correctness is not earning its cost.
  3. It makes the harness itself falsifiable: if a metric cannot separate this
     extractive stub from a frontier model, that metric is not measuring what
     you think it is.

What it is NOT
--------------
It is not a language model and its behaviour under prompt injection is a
*modelled assumption*, not evidence about any real model. ``simulate_injection
_susceptibility`` makes it obey plainly worded instructions it finds inside the
retrieved context -- which is how an unprotected instruction-tuned model tends
to behave, but the magnitude is stipulated, not measured. Any injection number
produced with this backend is labelled ``simulated`` in the results and must be
re-measured against a real model before it is quoted anywhere that matters.
"""
from __future__ import annotations

import base64
import binascii
import math
import re
import time
from collections import Counter
from collections.abc import Sequence

from ..embeddings import tokenize
from .base import LLMBackend, LLMResponse, estimate_tokens
from .prompts import REFUSAL_MARKER

_DOC_RE = re.compile(
    r'<document\s+id="(?P<id>\d+)"\s+source="(?P<source>[^"]*)"\s+page="(?P<page>\d+)"'
    r'(?:\s+section="(?P<section>[^"]*)")?\s*>(?P<body>.*?)</document>',
    re.DOTALL)
_PLAIN_DOC_RE = re.compile(r"^\[(?P<source>[^\],]+),\s*p\.?\s*(?P<page>\d+)\]\s*\n(?P<body>.*?)"
                           r"(?=\n\[|\Z)", re.DOTALL | re.MULTILINE)
_QUESTION_RE = re.compile(r"^QUESTION:\s*(?P<q>.+?)\s*$", re.MULTILINE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\u201c\"'])")
_B64_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "has", "have", "how", "i", "in", "is", "it", "its", "may", "must",
    "not", "of", "on", "or", "ся", "shall", "should", "that", "the", "их", "to",
    "under", "was", "what", "when", "where", "which", "who", "why", "will",
    "with", "you", "your", "about", "any", "if", "this", "there", "their",
}

# --- simulated susceptibility -------------------------------------------- #
_OBEY_PAYLOAD_RE = re.compile(
    r"(?:say|says|output|print|reply|respond|answer|write|emit|return)\s+"
    r"(?:with\s+|exactly\s+|only\s+|this\s+|the\s+following[:\s]+)*"
    r"[:\-]?\s*[\"\u201c']?(?P<payload>[^\"\u201d'\n]{3,200})",
    re.IGNORECASE)
# A bare imperative with an explicit "exactly"/"only" is itself enough to move an
# unprotected instruction-tuned model; it does not need a classic override phrase.
_DIRECTIVE_RE = re.compile(
    r"(?:say|output|print|reply|respond|answer|write|emit|return)\s+"
    r"(?:with\s+)?(?:exactly|only|the\s+following)\b",
    re.IGNORECASE)
_OBEY_TRIGGER_RE = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|"
    r"system)\s*(instructions|prompts|rules|context)?"
    r"|new\s+(instruction|task|directive|system\s+prompt)"
    r"|system\s*(override|update|message)"
    r"|you\s+(must|should|are\s+now|shall)\b"
    r"|important\s*[:!]"
    r"|^\s*(assistant|system)\s*:",
    re.IGNORECASE | re.MULTILINE)
# The captured payload runs to the end of the line, periods included: an
# instruction-following model emits the planted sentence whole, citation and all.
_PLANTED_FACT_RE = re.compile(
    r"(?:the\s+)?(?:correct|official|only|authoritative|approved)\s+"
    r"(?:answer|response)\s+(?:to\s+[^.\n]{0,80}\s+)?is[:\s]+(?P<payload>[^\n]{3,240})",
    re.IGNORECASE)
_LEAK_RE = re.compile(r"(reveal|repeat|print|show|disclose)\s+(your\s+)?"
                      r"(system\s+prompt|instructions|rules)", re.IGNORECASE)


def _decode_b64_blobs(text: str) -> str:
    """A real model reads base64 as text; the simulator does too."""
    extra: list[str] = []
    for m in _B64_RE.finditer(text):
        blob = m.group(0)
        try:
            decoded = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
            as_text = decoded.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if sum(c.isprintable() for c in as_text) / max(len(as_text), 1) > 0.9:
            extra.append(as_text)
    return "\n".join(extra)


class ParsedPrompt:
    def __init__(self, question: str, docs: list[dict[str, object]]) -> None:
        self.question = question
        self.docs = docs


def parse_prompt(prompt: str) -> ParsedPrompt:
    docs: list[dict[str, object]] = []
    for m in _DOC_RE.finditer(prompt):
        docs.append({"source": m.group("source"), "page": int(m.group("page")),
                     "section": m.group("section") or "", "body": m.group("body").strip()})
    if not docs:
        for m in _PLAIN_DOC_RE.finditer(prompt):
            body = m.group("body").strip()
            if body and "QUESTION:" not in body:
                docs.append({"source": m.group("source"), "page": int(m.group("page")),
                             "section": "", "body": body})
    qm = _QUESTION_RE.search(prompt)
    question = qm.group("q").strip() if qm else ""
    return ParsedPrompt(question, docs)


class OfflineExtractiveBackend(LLMBackend):
    name = "offline"
    supports_usage = False

    def __init__(self, model: str = "offline-extractive", temperature: float = 0.0,
                 max_tokens: int = 700, price_per_1k_input: float = 0.0,
                 price_per_1k_output: float = 0.0,
                 simulate_injection_susceptibility: bool = True,
                 coverage_threshold: float = 0.34, max_sentences: int = 3,
                 **kwargs) -> None:
        super().__init__(model, temperature, max_tokens, price_per_1k_input,
                         price_per_1k_output)
        self.simulate_injection = simulate_injection_susceptibility
        self.coverage_threshold = coverage_threshold
        self.max_sentences = max_sentences

    # ------------------------------------------------------------------ #
    def _idf(self, docs: Sequence[dict[str, object]]) -> dict[str, float]:
        df: Counter = Counter()
        for d in docs:
            df.update(set(tokenize(str(d["body"]))))
        n = max(len(docs), 1)
        return {t: math.log(1 + n / (1 + c)) + 0.1 for t, c in df.items()}

    def _obeyed_injection(self, docs: Sequence[dict[str, object]],
                          system: str | None) -> str | None:
        """Scripted susceptibility: follow plainly worded in-context commands."""
        if not self.simulate_injection:
            return None
        from ..safety.sanitize import detection_view

        for d in docs:
            # A real model reads through homoglyphs and base64; so does the stub.
            body = detection_view(str(d["body"]))
            body = body + "\n" + _decode_b64_blobs(body)
            if _LEAK_RE.search(body) and system:
                return system.strip().splitlines()[0][:200]
            planted = _PLANTED_FACT_RE.search(body)
            if planted:
                return planted.group("payload").strip()
            if _OBEY_TRIGGER_RE.search(body) or _DIRECTIVE_RE.search(body):
                payload = _OBEY_PAYLOAD_RE.search(body)
                if payload:
                    return payload.group("payload").strip()
        return None

    def _answer_from_docs(self, question: str,
                          docs: Sequence[dict[str, object]]) -> tuple[str, float]:
        q_tokens = [t for t in tokenize(question) if t not in STOPWORDS and len(t) > 1]
        if not q_tokens or not docs:
            return REFUSAL_MARKER, 0.0
        q_set = set(q_tokens)
        idf = self._idf(docs)
        scored: list[tuple[float, float, str, dict[str, object]]] = []
        for d in docs:
            sentences = [s.strip() for s in _SENT_SPLIT_RE.split(str(d["body"])) if s.strip()]
            header = str(d.get("section") or "")
            for sent in sentences:
                toks = set(tokenize(sent)) | set(tokenize(header))
                matched = q_set & toks
                if not matched:
                    continue
                weight = sum(idf.get(t, 0.5) for t in matched)
                coverage = len(matched) / len(q_set)
                length_penalty = 1.0 / (1.0 + math.log(1 + len(sent) / 400))
                scored.append((weight * coverage * length_penalty, coverage, sent, d))
        if not scored:
            return REFUSAL_MARKER, 0.0
        scored.sort(key=lambda t: -t[0])
        best_coverage = max(c for _, c, _, _ in scored[:5])
        if best_coverage < self.coverage_threshold:
            return REFUSAL_MARKER, best_coverage
        chosen: list[tuple[str, dict[str, object]]] = []
        seen: set[str] = set()
        for _, _, sent, d in scored:
            if sent in seen:
                continue
            seen.add(sent)
            chosen.append((sent, d))
            if len(chosen) >= self.max_sentences:
                break
        parts = []
        for sent, d in chosen:
            sent = sent.rstrip()
            if not sent.endswith((".", ";", ":")):
                sent += "."
            parts.append(f"{sent} [{d['source']}, p. {d['page']}]")
        return " ".join(parts), best_coverage

    def generate(self, prompt: str, system: str | None = None,
                 temperature: float | None = None) -> LLMResponse:
        started = time.perf_counter()
        parsed = parse_prompt(prompt)
        obeyed = self._obeyed_injection(parsed.docs, system)
        if obeyed is not None:
            text = obeyed
        else:
            text, _ = self._answer_from_docs(parsed.question, parsed.docs)
        latency = time.perf_counter() - started
        pt = estimate_tokens((system or "") + prompt)
        ct = estimate_tokens(text)
        return LLMResponse(text=text, model=self.model, backend=self.name,
                           prompt_tokens=pt, completion_tokens=ct,
                           latency_s=latency, cost_usd=self.price(pt, ct),
                           raw={"documents": len(parsed.docs),
                                "obeyed_injection": obeyed is not None})
