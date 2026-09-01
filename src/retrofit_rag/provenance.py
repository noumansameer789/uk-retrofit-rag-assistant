"""Shared provenance and text-normalisation controls."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from urllib.parse import urlparse

TRUSTED_SOURCE_HOSTS = frozenset(
    {
        "gov.uk",
        "www.gov.uk",
        "ofgem.gov.uk",
        "www.ofgem.gov.uk",
    }
)
ZERO_WIDTH_OR_CONTROL = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
WHITESPACE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Canonicalise Unicode and spacing before safety or hashing decisions."""

    canonical = unicodedata.normalize("NFKC", text)
    canonical = ZERO_WIDTH_OR_CONTROL.sub("", canonical)
    return WHITESPACE.sub(" ", canonical).strip()


def content_sha256(text: str) -> str:
    """Return a stable digest for a normalised evidence string."""

    return hashlib.sha256(normalise_text(text).encode("utf-8")).hexdigest()


def is_trusted_source_url(url: str) -> bool:
    """Accept only HTTPS URLs on the explicit official-source allowlist."""

    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname in TRUSTED_SOURCE_HOSTS
        and parsed.username is None
        and parsed.password is None
        and parsed.port in {None, 443}
    )
