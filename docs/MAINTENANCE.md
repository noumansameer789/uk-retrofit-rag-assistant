# Dependency maintenance

Reviewed 28 October 2026. Runtime roots and the transitive lock must change together. CI installs both `requirements.txt` and `requirements.lock`, then runs `pip check`; contradictory pins fail instead of silently testing the previous runtime.

A clean Python 3.12 environment resolved FastAPI 0.162.0, Pydantic 2.14.1 and Uvicorn 0.54.0, with Pydantic Core 2.50.1 in the 14-package runtime closure. Ruff is pinned to 0.17.2. Local validation passed all 46 unit/contract tests, the 12-case retrieval check (8 answerable and 4 unanswerable), the CLI smoke check, compilation, `pip check` and Ruff. These mocked provider tests do not rerun the separate historical 64-case Ollama evaluation.

Updates remain weekly, with grouped Python and GitHub Actions changes. Major Python image and action upgrades require their own compatibility checks.

The resolved runtime dependency scan reported no known vulnerabilities on 28 October 2026; see `dependency-audit-2026-10-28.json`. The earlier dated scan remains as historical evidence.

Dependabot PR #11 changed root pins without updating the custom runtime lock. Its failed checks exposed the mismatch. The compatible versions were applied together with a newly resolved lock and tested as one change; the consistency gate was retained. Future root-only proposals need the same lock refresh before merging. A clean runtime-only environment must be used to resolve the lock, before installing development tools, so their dependencies do not enter the deployment closure.
