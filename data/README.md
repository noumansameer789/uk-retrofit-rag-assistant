# Evidence data

`sources.json` is the allowlisted official-source manifest. `guidance.json` contains concise,
project-authored synopses that keep the repository and tests deterministic. Each entry records
the source URL, last manual check date and SHA-256 digest of the exact text supplied to the model.

To produce a fresh local corpus from the live official pages:

```bash
PYTHONPATH=src python -m retrofit_rag.ingestion
```

The generated corpus is written to `data/generated/guidance.json` and intentionally ignored by
Git. Review source changes before using generated content; a successful fetch does not prove that
a policy statement is still applicable to a particular person or property.

Official source material remains subject to the publisher's terms, including the Open Government
Licence where applicable. The MIT licence covers this project's original code, tests and authored
documentation, not third-party source pages or separately downloaded model weights.
