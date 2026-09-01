# UK retrofit guidance assistant

A citation-first retrieval prototype over official UK home-energy guidance.
Built independently to demonstrate a small, testable RAG foundation without
letting a language model invent eligibility or grant advice.

## What it does

- loads a source catalogue with title, URL and a concise evidence chunk
- ranks chunks using length-normalised TF-IDF-style sparse retrieval
- returns the source URL with every hit
- abstains when no catalogue evidence matches
- keeps retrieval separate from optional generation for easier evaluation

```bash
python -m unittest discover -s tests -v
PYTHONPATH=src python -m retrofit_rag.app "What support exists for a heat pump?"
```

## Why this design

Eligibility changes and the system is not a benefits calculator. A production
version should crawl versioned official pages, record retrieval recall on a
labelled question set, add an LLM only behind citation/abstention checks, and
monitor stale sources. The included text is a short project-authored synopsis;
the URLs are the authority.

## Container

```bash
docker build -t retrofit-rag .
docker run --rm retrofit-rag "energy performance certificate"
```
