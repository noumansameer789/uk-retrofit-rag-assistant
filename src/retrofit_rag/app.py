"""CLI demo. Generation is deliberately separate so every answer stays cited."""

from __future__ import annotations

import argparse
from pathlib import Path

from .retrieval import Retriever, load_catalogue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--catalogue", default=str(Path(__file__).parents[2] / "data" / "guidance.json"))
    args = parser.parse_args()
    hits = Retriever(load_catalogue(args.catalogue)).search(args.query)
    if not hits:
        print("No supported guidance found; refine the question.")
        return
    for rank, (doc, score) in enumerate(hits, 1):
        print(f"{rank}. {doc.title} ({score:.3f})\n   {doc.url}\n   {doc.text}")


if __name__ == "__main__":
    main()
