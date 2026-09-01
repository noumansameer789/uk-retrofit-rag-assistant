# Security policy

## Supported version

Security fixes are applied to the current `main` branch. This portfolio project has no
hosted production service and does not process user accounts or payments.

## Reporting a vulnerability

Use GitHub's **Security → Report a vulnerability** flow when it is available. Do not put
credentials, exploitable payloads or personal data in a public issue. A report should state
the affected revision, reproduction steps, impact and any suggested mitigation.

For non-sensitive defects, open a normal GitHub issue.

## Secrets

No API key is required for the default Ollama stack. If an OpenAI-compatible provider is
configured, supply its key only through the runtime environment. Never commit `.env` files,
tokens, retrieved personal data or provider responses containing sensitive information.
