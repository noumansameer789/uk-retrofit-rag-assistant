"""Evaluate a real Ollama-backed FastAPI service; never substitute mock answers."""

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen


def request(url, body=None):
    payload = json.dumps(body).encode() if body is not None else None
    with urlopen(Request(url, data=payload, headers={"Content-Type": "application/json"}), timeout=300) as response:
        return json.load(response)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(base, ollama, cases_path, output):
    root = Path(__file__).resolve().parents[1]
    catalogue = root / "data/guidance.json"
    sources = {doc["url"]: doc for doc in json.loads(catalogue.read_text())}
    cases = json.loads(cases_path.read_text())
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case identifiers")
    if request(base + "/health").get("provider") != "ollama":
        raise RuntimeError("This benchmark requires the real Ollama provider")
    if request(base + "/ready").get("status") != "ready":
        raise RuntimeError("Model is not ready")
    metadata = request(ollama + "/api/tags")
    started = datetime.now(UTC).isoformat()
    source_hashes = {str(p.relative_to(root)): digest(p)
                     for p in sorted((root / "src").rglob("*.py"))}
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        start = time.monotonic()
        error = None
        try:
            result = request(base + "/ask", {"question": case["question"], "top_k": 3})
        except Exception as exc:
            error = str(exc)
            result = {"status": "error", "answer": "", "citations": []}
        citations = result.get("citations", [])
        urls = {citation.get("url") for citation in citations}
        status_ok = result.get("status") == case["expected_status"]
        citation_ok = (bool(urls & set(case["relevant_urls"]))
                       if case["relevant_urls"] else not citations)
        provenance_ok = all(c.get("url") in sources and
                            c.get("content_sha256") == sources[c["url"]]["content_sha256"]
                            for c in citations)
        fact_ok = (bool(re.search(case["expected_pattern"], result.get("answer", ""), re.I))
                   if case.get("expected_pattern") else True)
        row = {**case, "response": result, "latency_seconds": time.monotonic()-start,
               "status_correct": status_ok, "citation_correct": citation_ok,
               "provenance_correct": provenance_ok, "reference_keyword_match": fact_ok,
               "passed": bool(status_ok and citation_ok and provenance_ok and fact_ok),
               "error": error}
        rows.append(row)
        with (output / "responses.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"{case['id']} {case['category']}: {result['status']} pass={row['passed']}", flush=True)
    categories = sorted({row["category"] for row in rows})
    summary = {"started_at": started, "completed_at": datetime.now(UTC).isoformat(),
        "provider": "ollama", "model_inventory": metadata,
        "runtime": platform.platform(), "python": platform.python_version(),
        "source_commit": os.getenv("GITHUB_SHA", "local working tree; see source_file_sha256"),
        "source_file_sha256": source_hashes,
        "dataset_sha256": digest(cases_path), "catalogue_sha256": digest(catalogue),
        "cases": len(rows), "passed": sum(r["passed"] for r in rows),
        "errors": sum(r["error"] is not None for r in rows),
        "by_category": {cat: {"cases": sum(r["category"] == cat for r in rows),
                              "passed": sum(r["passed"] for r in rows if r["category"] == cat)}
                        for cat in categories},
        "answered": sum(r["response"]["status"] == "answered" for r in rows),
        "latency_p50_seconds": statistics.median(r["latency_seconds"] for r in rows),
        "latency_max_seconds": max(r["latency_seconds"] for r in rows),
        "limitations": ["Authored regression suite over four curated summaries; not independent user traffic.",
                        "Keyword matching is a limited answer-correctness proxy, not semantic adjudication.",
                        "Exact quotation prevents novel claims but cannot prove relevance or source truth.",
                        "CPU hardware, model digest and runtime affect latency and reproducibility."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"source_file_sha256", "model_inventory"}}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--cases", type=Path, default=Path("data/generation_evaluation.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/live-evaluation"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Choose a new output directory; previous evidence is immutable.")
    result = evaluate(args.base_url, args.ollama_url, args.cases, args.output)
    if result["errors"]:
        raise SystemExit("Provider errors occurred; inspect the saved report.")
    if result["passed"] != result["cases"]:
        raise SystemExit("Regression cases failed; inspect the saved responses before release.")
