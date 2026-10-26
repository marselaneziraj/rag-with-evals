"""Input-side mitigation: neutralise instructions hiding in retrieved text.

Threat model
------------
An attacker who can get a document into the corpus (a public PDF you crawled, a
user upload, a wiki page) controls text that will be placed inside the model's
prompt. Retrieval is the delivery mechanism. The attacker's goals are: override
the answer, exfiltrate the system prompt, or make the system emit a chosen
string.

Three layers, in order of reliability
-------------------------------------
1. **Delimiting + escaping** (here and in ``prompts.format_context``): chunk
   text can never close the ``<document>`` block, because the closing tag and
   role-label strings are escaped. This defeats boundary-breaking outright --
   it is a string transformation, not a request to the model.
2. **Instruction neutralisation** (here): lines matching known injection
   phrasings are replaced by a visible placeholder. Unicode is NFKC-normalised
   and zero-width characters are stripped *before* matching, so homoglyph and
   invisible-character evasion is defeated; base64 blobs are decoded and
   scanned, so encoded payloads are caught too.
3. **Instruction hierarchy in the system prompt** (``prompts``): the weakest
   layer, because it asks the model to behave rather than making misbehaviour
   impossible -- but it is also the only one that generalises to phrasings the
   regexes do not know.

Known gap, measured rather than hidden
--------------------------------------
Pattern matching cannot catch an attack with no imperative surface -- a
*planted assertion* ("The correct answer to any question about retention is
X"), which reads exactly like ordinary document content. The injection suite
includes those cases on purpose, and the residual success rate after mitigation
is reported rather than rounded to zero.
"""
from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..ingest.chunking import Chunk

PLACEHOLDER = "[untrusted-instruction removed by input filter]"

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)

# Cyrillic and Greek characters that render identically to Latin ones. NFKC does
# not touch them -- they are legitimately different letters -- so an attacker can
# write "Ignore аll previous instructions" with a Cyrillic a and walk past a
# pattern list. Folding is applied to a *detection copy* only: the text handed to
# the model keeps its original characters, so a genuinely Greek or Cyrillic
# document is not mangled by the filter.
HOMOGLYPHS = str.maketrans({
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c",
    "\u0445": "x", "\u0443": "y", "\u0456": "i", "\u0458": "j", "\u04bb": "h",
    "\u0501": "d", "\u051b": "q", "\u0455": "s", "\u043d": "h", "\u0410": "A",
    "\u0412": "B", "\u0415": "E", "\u041a": "K", "\u041c": "M", "\u041d": "H",
    "\u041e": "O", "\u0420": "P", "\u0421": "C", "\u0422": "T", "\u0423": "Y",
    "\u0425": "X", "\u03bf": "o", "\u03b1": "a", "\u03b5": "e", "\u03c1": "p",
    "\u03bd": "v", "\u03c4": "t", "\u03b9": "i", "\u03ba": "k", "\u03c3": "o",
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0396": "Z", "\u0397": "H",
    "\u0399": "I", "\u039a": "K", "\u039c": "M", "\u039d": "N", "\u039f": "O",
    "\u03a1": "P", "\u03a4": "T", "\u03a7": "X", "\u03a5": "Y",
})

INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("override", re.compile(
        r"(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+)?"
        r"(previous|prior|above|earlier|preceding|system|original)\s*"
        r"(instructions?|prompts?|rules?|directions?|context)?", re.IGNORECASE)),
    ("new_directive", re.compile(
        r"(new|updated|revised)\s+(instruction|task|directive|system\s+prompt|rules?)"
        r"|your\s+(new\s+)?(task|instruction|role)\s+is", re.IGNORECASE)),
    ("role_label", re.compile(r"^\s*(system|assistant|user|developer)\s*:", re.IGNORECASE
                              | re.MULTILINE)),
    ("role_play", re.compile(
        r"(you\s+are\s+now|act\s+as|pretend\s+to\s+be|from\s+now\s+on|switch\s+to)\s+"
        r"[^.\n]{0,60}(mode|assistant|dan|admin|developer)?", re.IGNORECASE)),
    ("emit_string", re.compile(
        r"(say|output|print|reply|respond|answer|write|emit|return)\s+"
        r"(exactly|only|with|the\s+following|this)\b", re.IGNORECASE)),
    ("leak_prompt", re.compile(
        r"(reveal|repeat|print|show|disclose|summari[sz]e)\s+(your\s+|the\s+)?"
        r"(system\s+prompt|initial\s+instructions|hidden\s+rules|configuration)",
        re.IGNORECASE)),
    ("tool_abuse", re.compile(
        r"(send|post|upload|exfiltrate|email)\s+[^.\n]{0,40}\s+to\s+"
        r"(https?://|[\w.-]+@)", re.IGNORECASE)),
    ("multilingual_override", re.compile(
        r"(ignorez|ignoriere|ignora|olvida|\u0438\u0433\u043d\u043e\u0440\u0438\u0440"
        r"\u0443\u0439|shperfill)\s+", re.IGNORECASE)),
    ("fake_authority", re.compile(
        r"(this\s+(message|note)\s+is\s+from\s+(the\s+)?(operator|developer|admin)"
        r"|security\s+(override|notice)\s*[:#]|priority\s*[:#]?\s*(high|critical)\s*[:#])",
        re.IGNORECASE)),
]

_B64_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")
_DELIMITER_RE = re.compile(r"</?\s*(document|context|system|instructions?)\b[^>]*>",
                           re.IGNORECASE)


@dataclass
class SanitizationResult:
    text: str
    flags: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: bool = False

    @property
    def suspicious(self) -> bool:
        return bool(self.flags)


def normalise(text: str) -> str:
    """NFKC + zero-width strip. Safe to keep: no letter is replaced."""
    return unicodedata.normalize("NFKC", text or "").translate(ZERO_WIDTH)


def detection_view(text: str) -> str:
    """Normalised *and* homoglyph-folded copy, used only for pattern matching."""
    return normalise(text).translate(HOMOGLYPHS)


def _decoded_b64_is_hostile(blob: str) -> bool:
    try:
        decoded = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
        text = decoded.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return False
    probe = detection_view(text)
    return any(pattern.search(probe) for _, pattern in INJECTION_PATTERNS)


def sanitize_text(text: str) -> SanitizationResult:
    original = text or ""
    text = normalise(original)
    flags: list[str] = []
    removed: list[str] = []

    # 1. escape anything that could close or forge a prompt block
    if _DELIMITER_RE.search(text):
        flags.append("delimiter_injection")
        text = _DELIMITER_RE.sub(lambda m: "(" + m.group(0).strip("<>") + ")", text)

    # 2. encoded payloads
    def _b64_sub(m: re.Match) -> str:
        if _decoded_b64_is_hostile(m.group(0)):
            flags.append("encoded_payload")
            removed.append(m.group(0)[:60])
            return PLACEHOLDER
        return m.group(0)

    text = _B64_RE.sub(_b64_sub, text)

    # 3. line-level instruction neutralisation. Matching runs against the
    #    homoglyph-folded view; the line that is kept or dropped is the original.
    out_lines: list[str] = []
    for line in text.splitlines():
        probe = detection_view(line)
        hit = next((name for name, pattern in INJECTION_PATTERNS if pattern.search(probe)),
                   None)
        if hit:
            flags.append(hit)
            removed.append(line.strip()[:160])
            out_lines.append(PLACEHOLDER)
        else:
            out_lines.append(line)
    text = "\n".join(out_lines)

    return SanitizationResult(text=text, flags=sorted(set(flags)), removed=removed,
                              modified=text != original)


def sanitize_chunk(chunk: Chunk) -> tuple[Chunk, SanitizationResult]:
    result = sanitize_text(chunk.text)
    if not result.modified:
        return chunk, result
    clean = Chunk(id=chunk.id, text=result.text, source=chunk.source, page=chunk.page,
                  page_end=chunk.page_end, section=chunk.section,
                  chunk_index=chunk.chunk_index,
                  extra={**chunk.extra, "sanitized": True, "flags": result.flags})
    return clean, result


def sanitize_chunks(chunks: Sequence[Chunk]) -> tuple[list[Chunk], list[str]]:
    cleaned: list[Chunk] = []
    all_flags: list[str] = []
    for chunk in chunks:
        clean, result = sanitize_chunk(chunk)
        cleaned.append(clean)
        all_flags.extend(result.flags)
    return cleaned, sorted(set(all_flags))
