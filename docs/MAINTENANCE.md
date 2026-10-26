# Dependency maintenance

Reviewed 26 October 2026. Runtime roots and the transitive lock must change together. CI installs both `requirements.txt` and `requirements.lock`, then runs `pip check`; contradictory pins now fail instead of silently testing the previous runtime.

A clean Python 3.12 environment resolved FastAPI 0.143.0, Pydantic 2.14.0 and Uvicorn 0.54.0. Local validation passed all 46 unit/contract tests, the 12-case retrieval check (8 answerable and 4 unanswerable) and Ruff. These mocked provider tests do not rerun the separate historical 64-case Ollama evaluation.

Updates remain weekly, with grouped Python and GitHub Actions changes. Major Python image and action upgrades require their own compatibility checks.

The resolved runtime dependency scan reported no known vulnerabilities on 26 October 2026; see `dependency-audit-2026-10-26.json`.
