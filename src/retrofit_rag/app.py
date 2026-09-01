"""CLI for retrieval inspection or a fully generated, citation-checked answer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .providers import build_llm
from .rag import RAGService
from .retrieval import Retriever, load_catalogue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--catalogue", default=str(Path(__file__).parents[2] / "data" / "guidance.json"))
    parser.add_argument(
        "--provider",
        choices=("retrieval", "openai", "ollama"),
        default="retrieval",
        help="Use retrieval-only output or a configured LLM provider.",
    )
    args = parser.parse_args()
    retriever = Retriever(load_catalogue(args.catalogue))
    if args.provider != "retrieval":
        result = RAGService(retriever, build_llm(args.provider)).ask(args.query)
        print(json.dumps(result.to_dict(), indent=2))
        return
    hits = retriever.search(args.query)
    if not hits:
        print("No supported guidance found; refine the question.")
        return
    for rank, (doc, score) in enumerate(hits, 1):
        print(f"{rank}. {doc.title} ({score:.3f})\n   {doc.url}\n   {doc.text}")


if __name__ == "__main__":
    main()
