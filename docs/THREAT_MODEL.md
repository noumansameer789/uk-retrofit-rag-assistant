# Threat model

This document describes the controls and residual risks for the local portfolio build. It is
not a claim that the system is suitable for regulated or unattended production use.

## Assets and trust boundaries

| Asset | Boundary | Control |
|---|---|---|
| User question | Untrusted API input | Pydantic length bounds, Unicode canonicalisation, injection and eligibility screening |
| Guidance text | Untrusted external content | HTTPS host allowlist, redirect/content-type/size checks, deterministic parsing, SHA-256 digests |
| Prompt | Application-to-model boundary | Evidence delimiter, explicit untrusted-context instruction, no secrets inserted |
| Model output | Untrusted generated content | JSON schema checks, sentence-level citations, source-ID allowlist, unsafe-claim rejection |
| Provider credential | Runtime secret | Environment only, `.env` ignored, no logging in application code |
| Container | Local execution boundary | Non-root API user, read-only filesystem, temporary `/tmp`, no-new-privileges |

## Abuse cases and mitigations

| Threat | Mitigation | Residual risk |
|---|---|---|
| Direct prompt injection | Canonicalise Unicode and reject known override, prompt-disclosure and secret-extraction patterns before retrieval | Pattern matching cannot recognise every paraphrase |
| Indirect injection in a source | Apply the same screening to retrieved context and drop contaminated chunks | Novel or obfuscated attacks can evade patterns |
| Unsupported claim or citation laundering | Require every factual sentence to carry an allowed marker and require the JSON citation list to match | A source can be cited while being misunderstood by the model |
| Malformed model output | One constrained repair attempt, followed by deterministic refusal | A small model may refuse often even for supported questions |
| Stale policy guidance | Return `checked_at`, source URL and content digest with citations | The project has no scheduled production crawler or freshness alert |
| Server-side request forgery during ingestion | Only allow HTTPS on exact GOV.UK/Ofgem hostnames; reject off-list redirects and non-default ports | Compromise of an allowlisted publisher is outside this project's boundary |
| Oversized responses | Cap source pages at 2 MB and provider responses at 1 MB | Resource limits for concurrent requests need an external gateway in production |
| Dependency compromise | Exact runtime lock, weekly Dependabot checks and `pip-audit` in CI | A zero-day may not yet be present in advisory databases |

## Deliberate non-goals

- The assistant does not make personal grant-eligibility decisions or promise funding.
- It does not crawl arbitrary user-supplied URLs.
- It does not persist prompts, answers or personal profiles.
- It does not expose Ollama directly outside the Compose network.
- It is not a substitute for current instructions from the cited authority or an accredited
  installer.

## Production work still required

A real public service would additionally need authentication/authorisation, TLS termination,
rate limiting, request and model time budgets, structured privacy-reviewed logging, monitoring,
backups, scheduled ingestion with change review, red-team evaluation, accessibility testing and
an incident-response owner.
