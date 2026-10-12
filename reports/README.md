# Real model evaluation record

The generation evaluation uses a real local Llama 3.2 3B quantised model through Ollama 0.33.2 and the FastAPI HTTP endpoint. No provider responses were mocked in these runs. The separate unit tests continue to use mocks for deterministic failure-path coverage.

| Run | Passed | Provider errors | Interpretation |
| --- | ---: | ---: | --- |
| `live-evaluation-2026-10-12` | 36/64 | 0 | Initial diagnostic; source files changed during the run and its end-of-run source hashes do not identify the loaded code. Do not use as a frozen reproducibility claim. |
| `live-evaluation-final-2026-10-12` | 62/64 | 0 | Frozen pre-correction run; source hashes captured at start. A10 was refused for missing per-sentence citations; N04 returned a vague quote instead of the requested numerical threshold. |
| `live-evaluation-release-2026-10-12` | 64/64 | 0 | Frozen release regression run after correcting those two defects. All answers and refusals are retained. |

The release changes normalise sentence markers only when the entire text already matches a single explicitly cited source. A separate check rejects a numerical-detail question if the quote contains no numerical value. Neither control invents or supplies missing eligibility information. Unit cases cover multi-sentence support, vague-quantity rejection and acceptance of an actually supported numerical quote.

## Boundaries

There are 64 authored cases over four short project-authored summaries of GOV.UK/Ofgem guidance. These cases were used during development; they are not an untouched test or representative user traffic. Scoring checks status, relevant source, source hash and reference keywords. Keyword matching cannot establish full semantic correctness. Known injection patterns are a limited regression set, not a security guarantee.

The summaries retain their recorded 1 September 2026 check dates; the evaluation does not revalidate current public policy. Exact quotation verifies textual support, not truth, relevance or freshness. CPU performance and quantised-model output may vary between runtimes. The supplied report records the model digest, environment, timing and source hashes so a reviewer can inspect what ran.

The evaluator refuses to overwrite existing reports. The real-model CI job fails on any regression case or provider error and uploads responses even if it fails. A green job means this narrow contract passed; it is not a claim of production readiness.
