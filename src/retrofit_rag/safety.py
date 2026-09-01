"""Input and output controls for a citation-first RAG boundary."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .provenance import normalise_text

MAX_QUESTION_CHARS = 1_000
MAX_ANSWER_CHARS = 4_000
INJECTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore\s+(all\s+)?(previous|prior|system|developer)\s+instructions?",
        r"reveal\s+(the\s+)?(system|developer)\s+prompt",
        r"print\s+(the\s+)?(secret|api\s*key|environment\s+variables?)",
        r"override\s+(the\s+)?(safety|guardrails?|instructions?)",
        r"act\s+as\s+(dan|an?\s+unrestricted)",
    )
)
ELIGIBILITY_DECISION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(am|are)\s+i\s+(definitely\s+)?eligible\b",
        r"\bdo\s+i\s+(definitely\s+)?qualify\b",
        r"\bguarantee(d)?\s+(me|my|that\s+i)\b",
        r"\bexact(ly)?\s+how\s+much\s+(will|can)\s+i\s+(get|receive)\b",
    )
)
UNSAFE_ANSWER_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\byou\s+(are|will\s+be)\s+(definitely\s+)?eligible\b",
        r"\byou\s+(definitely\s+)?qualify\b",
        r"\byou\s+are\s+guaranteed\b",
        r"\byou\s+will\s+receive\s+[£$€]?\d",
    )
)
CITATION = re.compile(r"\[(\d+)\]")


class OutputValidationError(ValueError):
    """Generated content did not satisfy the grounding contract."""


@dataclass(frozen=True)
class ValidatedGeneration:
    answer: str
    citation_ids: tuple[int, ...]


def unsafe_question_reason(question: str) -> str | None:
    normalized = normalise_text(question)
    if not normalized:
        return "empty_question"
    if len(normalized) > MAX_QUESTION_CHARS:
        return "question_too_long"
    if any(pattern.search(normalized) for pattern in INJECTION_PATTERNS):
        return "prompt_injection"
    if any(pattern.search(normalized) for pattern in ELIGIBILITY_DECISION_PATTERNS):
        return "eligibility_decision_request"
    return None


def context_is_safe(text: str) -> bool:
    normalized = normalise_text(text)
    return not any(pattern.search(normalized) for pattern in INJECTION_PATTERNS)


def _json_object(raw: str) -> dict[str, object]:
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise OutputValidationError("model_output_not_json") from exc
    if not isinstance(payload, dict):
        raise OutputValidationError("model_output_not_object")
    return payload


def _factual_sentences(answer: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer)
        if len(re.findall(r"[A-Za-z]+", sentence)) >= 4
    ]


def validate_generation(raw: str, allowed_ids: set[int]) -> ValidatedGeneration:
    payload = _json_object(raw)
    if set(payload) != {"answer", "citations"}:
        raise OutputValidationError("unexpected_output_fields")
    answer = payload.get("answer")
    citation_ids = payload.get("citations")
    if not isinstance(answer, str) or not answer.strip():
        raise OutputValidationError("missing_answer")
    answer = normalise_text(answer)
    if len(answer) > MAX_ANSWER_CHARS:
        raise OutputValidationError("answer_too_long")
    if not isinstance(citation_ids, list) or not citation_ids:
        raise OutputValidationError("missing_citations")
    if any(type(item) is not int for item in citation_ids):
        raise OutputValidationError("invalid_citation_type")
    cited = set(citation_ids)
    bracketed = {int(item) for item in CITATION.findall(answer)}
    if not cited.issubset(allowed_ids) or not bracketed.issubset(allowed_ids):
        raise OutputValidationError("unknown_citation")
    if cited != bracketed:
        raise OutputValidationError("citation_contract_mismatch")
    sentences = _factual_sentences(answer)
    if not sentences or any(not CITATION.search(sentence) for sentence in sentences):
        raise OutputValidationError("uncited_claim")
    if any(pattern.search(answer) for pattern in UNSAFE_ANSWER_PATTERNS):
        raise OutputValidationError("unsafe_eligibility_claim")
    return ValidatedGeneration(answer=answer, citation_ids=tuple(sorted(cited)))
