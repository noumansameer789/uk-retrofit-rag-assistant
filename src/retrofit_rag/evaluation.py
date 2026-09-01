"""Deterministic retrieval evaluation for the included labelled smoke set."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .provenance import is_trusted_source_url, normalise_text
from .retrieval import Retriever, load_catalogue

ROOT = Path(__file__).parents[2]


@dataclass(frozen=True)
class RetrievalMetrics:
    answerable_cases: int
    unanswerable_cases: int
    hit_rate_at_k: float
    mean_reciprocal_rank: float
    abstention_accuracy: float


def load_cases(path: str | Path) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Evaluation data must be a non-empty list")
    questions: set[str] = set()
    for case in payload:
        if not isinstance(case, dict):
            raise ValueError("Every evaluation case must be an object")
        question = case.get("question")
        relevant_urls = case.get("relevant_urls")
        if not isinstance(question, str) or not normalise_text(question):
            raise ValueError("Every evaluation case requires a question")
        if question in questions:
            raise ValueError(f"Duplicate evaluation question: {question}")
        if not isinstance(relevant_urls, list) or any(
            not isinstance(url, str) or not is_trusted_source_url(url)
            for url in relevant_urls
        ):
            raise ValueError("relevant_urls must contain only allowlisted HTTPS sources")
        questions.add(question)
    return payload


def evaluate_retrieval(
    retriever: Retriever,
    cases: list[dict[str, object]],
    k: int = 3,
) -> RetrievalMetrics:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not cases:
        raise ValueError("At least one evaluation case is required")
    hits, reciprocal_ranks, abstentions = [], [], []
    for case in cases:
        question = str(case["question"])
        expected = {str(url) for url in case.get("relevant_urls", [])}
        returned = [document.url for document, _ in retriever.search(question, k=k)]
        if expected:
            ranks = [index for index, url in enumerate(returned, start=1) if url in expected]
            hits.append(bool(ranks))
            reciprocal_ranks.append(1 / min(ranks) if ranks else 0.0)
        else:
            abstentions.append(not returned)
    return RetrievalMetrics(
        answerable_cases=len(hits),
        unanswerable_cases=len(abstentions),
        hit_rate_at_k=sum(hits) / len(hits) if hits else 0.0,
        mean_reciprocal_rank=(
            sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        ),
        abstention_accuracy=(
            sum(abstentions) / len(abstentions) if abstentions else 0.0
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", default=str(ROOT / "data" / "guidance.json"))
    parser.add_argument("--cases", default=str(ROOT / "data" / "evaluation.json"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    parser.add_argument("--min-mrr", type=float, default=0.0)
    parser.add_argument("--min-abstention", type=float, default=0.0)
    args = parser.parse_args()
    metrics = evaluate_retrieval(
        Retriever(load_catalogue(args.catalogue)),
        load_cases(args.cases),
        k=args.k,
    )
    print(json.dumps(asdict(metrics), indent=2, sort_keys=True))
    if (
        metrics.hit_rate_at_k < args.min_hit_rate
        or metrics.mean_reciprocal_rank < args.min_mrr
        or metrics.abstention_accuracy < args.min_abstention
    ):
        raise SystemExit("Evaluation threshold failed")


if __name__ == "__main__":
    main()
