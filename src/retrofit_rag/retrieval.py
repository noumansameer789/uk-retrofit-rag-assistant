"""Citation-first sparse retriever for UK home-retrofit guidance."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Document:
    title: str
    url: str
    text: str


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class Retriever:
    def __init__(self, documents: list[Document]):
        if not documents:
            raise ValueError("At least one document is required")
        self.documents = documents
        self.term_counts = [Counter(tokens(doc.text)) for doc in documents]
        self.doc_freq = Counter(term for counts in self.term_counts for term in counts)

    def search(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        query_terms = Counter(tokens(query))
        scores: list[tuple[Document, float]] = []
        total = len(self.documents)
        for doc, counts in zip(self.documents, self.term_counts):
            length = sum(counts.values()) or 1
            score = 0.0
            for term, qtf in query_terms.items():
                idf = math.log((1 + total) / (1 + self.doc_freq[term])) + 1
                score += qtf * counts[term] / length * idf
            if score:
                scores.append((doc, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:k]


def load_catalogue(path: str | Path) -> list[Document]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Document(**item) for item in payload]
