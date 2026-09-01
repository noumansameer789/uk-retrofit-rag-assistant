"""Deterministic BM25 retrieval over provenance-checked guidance."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .provenance import content_sha256, is_trusted_source_url, normalise_text

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "what", "when", "where", "which", "who", "with",
}


@dataclass(frozen=True)
class Document:
    title: str
    url: str
    text: str
    document_id: str | None = None
    source_id: str | None = None
    checked_at: str | None = None
    content_sha256: str | None = None
    chunk_id: str | None = None


def tokens(text: str) -> list[str]:
    return [
        token
        for token in TOKEN.findall(normalise_text(text).lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


class Retriever:
    """Small-corpus Okapi BM25 implementation with stable tie-breaking."""

    def __init__(
        self,
        documents: list[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not documents:
            raise ValueError("At least one document is required")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(tokens(f"{doc.title} {doc.text}")) for doc in documents]
        self.doc_freq = Counter(term for counts in self.term_counts for term in counts)
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = sum(self.lengths) / len(self.lengths) or 1.0

    def search(
        self,
        query: str,
        k: int = 3,
        min_score: float = 0.0,
    ) -> list[tuple[Document, float]]:
        if k < 1:
            raise ValueError("k must be at least 1")
        query_terms = Counter(tokens(query))
        if not query_terms:
            return []
        scores: list[tuple[Document, float]] = []
        total = len(self.documents)
        for doc, counts, length in zip(
            self.documents, self.term_counts, self.lengths, strict=True
        ):
            score = 0.0
            for term, qtf in query_terms.items():
                frequency = counts[term]
                if not frequency:
                    continue
                document_frequency = self.doc_freq[term]
                idf = math.log(
                    1 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                normaliser = self.k1 * (
                    1 - self.b + self.b * length / self.average_length
                )
                query_weight = 1 + math.log(qtf)
                score += (
                    idf
                    * (frequency * (self.k1 + 1) / (frequency + normaliser))
                    * query_weight
                )
            if score > min_score:
                scores.append((doc, score))
        return sorted(
            scores,
            key=lambda item: (-item[1], item[0].url, item[0].title),
        )[:k]


def load_catalogue(path: str | Path) -> list[Document]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Catalogue must be a non-empty JSON array")
    documents: list[Document] = []
    identifiers: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Every catalogue entry must be an object")
        try:
            document = Document(**item)
        except TypeError as exc:
            raise ValueError("Catalogue entry has missing or unknown fields") from exc
        if not all(normalise_text(value) for value in (document.title, document.text)):
            raise ValueError("Catalogue title and text cannot be empty")
        if not is_trusted_source_url(document.url):
            raise ValueError(f"Untrusted catalogue URL: {document.url}")
        if document.document_id:
            if document.document_id in identifiers:
                raise ValueError(f"Duplicate document_id: {document.document_id}")
            identifiers.add(document.document_id)
        if document.checked_at:
            try:
                date.fromisoformat(document.checked_at)
            except ValueError as exc:
                raise ValueError("checked_at must use YYYY-MM-DD") from exc
        if document.content_sha256:
            if document.content_sha256 != content_sha256(document.text):
                raise ValueError(f"Content digest mismatch: {document.document_id or document.url}")
        documents.append(document)
    return documents
