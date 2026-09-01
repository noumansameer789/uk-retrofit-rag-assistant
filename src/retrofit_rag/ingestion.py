"""Reproducible ingestion of allowlisted official UK guidance pages."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .provenance import content_sha256, is_trusted_source_url, normalise_text

MAX_RESPONSE_BYTES = 2_000_000
USER_AGENT = "uk-retrofit-rag-assistant/2.0 (+portfolio research project)"
CONTENT_TAGS = frozenset({"h1", "h2", "h3", "p", "li"})
SKIP_TAGS = frozenset({"script", "style", "noscript", "nav", "footer", "header", "svg"})


class AllowlistRedirectHandler(HTTPRedirectHandler):
    """Refuse an off-list redirect before making the redirected request."""

    def redirect_request(
        self,
        request: Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> Request | None:
        if not is_trusted_source_url(new_url):
            raise ValueError(f"Refusing redirect to non-allowlisted URL: {new_url}")
        return super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )


OFFICIAL_SOURCE_OPENER = build_opener(AllowlistRedirectHandler())


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    title: str
    url: str


@dataclass(frozen=True)
class IngestedDocument:
    title: str
    url: str
    text: str
    document_id: str
    source_id: str
    checked_at: str
    content_sha256: str
    chunk_id: str


class GuidanceHTMLParser(HTMLParser):
    """Extract readable headings, paragraphs and list items without scripts/navigation."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._capture_depth = 0
        self._buffer: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, _: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip_depth += 1
        if not self._skip_depth and tag in CONTENT_TAGS:
            self._capture_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if not self._skip_depth and tag in CONTENT_TAGS and self._capture_depth:
            self._capture_depth -= 1
            part = normalise_text(" ".join(self._buffer))
            self._buffer.clear()
            if part:
                self.parts.append(part)

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and self._capture_depth:
            self._buffer.append(data)


def extract_text(html: str) -> str:
    parser = GuidanceHTMLParser()
    parser.feed(html)
    text = normalise_text("\n".join(parser.parts))
    if len(text.split()) < 20:
        raise ValueError("Official page did not contain enough extractable guidance text")
    return text


def chunk_text(text: str, *, words_per_chunk: int = 180, overlap: int = 30) -> list[str]:
    if words_per_chunk < 20 or overlap < 0 or overlap >= words_per_chunk:
        raise ValueError("Invalid chunking parameters")
    words = normalise_text(text).split()
    if not words:
        return []
    step = words_per_chunk - overlap
    return [
        " ".join(words[start : start + words_per_chunk])
        for start in range(0, len(words), step)
        if words[start : start + words_per_chunk]
    ]


def fetch_official_html(
    url: str,
    *,
    timeout: float = 20.0,
    max_bytes: int = MAX_RESPONSE_BYTES,
) -> str:
    if not is_trusted_source_url(url):
        raise ValueError(f"Refusing non-allowlisted source URL: {url}")
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    try:
        with OFFICIAL_SOURCE_OPENER.open(request, timeout=timeout) as response:
            final_url = response.geturl()
            if not is_trusted_source_url(final_url):
                raise ValueError(f"Refusing redirect to non-allowlisted URL: {final_url}")
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise ValueError(f"Unexpected source content type: {content_type}")
            raw = response.read(max_bytes + 1)
    except HTTPError as exc:
        raise RuntimeError(f"Source returned HTTP {exc.code}: {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Source could not be reached: {url}") from exc
    if len(raw) > max_bytes:
        raise ValueError(f"Source exceeded {max_bytes} bytes: {url}")
    return raw.decode("utf-8", errors="replace")


def load_source_manifest(path: str | Path) -> list[SourceSpec]:
    payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Source manifest must be a non-empty JSON array")
    sources: list[SourceSpec] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Every source entry must be an object")
        try:
            source = SourceSpec(
                source_id=str(item["source_id"]),
                title=str(item["title"]),
                url=str(item["url"]),
            )
        except KeyError as exc:
            raise ValueError("Source entry is missing a required field") from exc
        if not source.source_id or source.source_id in seen:
            raise ValueError(f"Invalid or duplicate source_id: {source.source_id}")
        if not is_trusted_source_url(source.url):
            raise ValueError(f"Untrusted source URL: {source.url}")
        seen.add(source.source_id)
        sources.append(source)
    return sources


def ingest_sources(sources: list[SourceSpec], *, checked_at: str | None = None) -> list[dict[str, str]]:
    checked = checked_at or date.today().isoformat()
    documents: list[IngestedDocument] = []
    for source in sources:
        text = extract_text(fetch_official_html(source.url))
        for index, chunk in enumerate(chunk_text(text), start=1):
            chunk_id = f"{source.source_id}-{index:03d}"
            documents.append(
                IngestedDocument(
                    title=source.title,
                    url=source.url,
                    text=chunk,
                    document_id=chunk_id,
                    source_id=source.source_id,
                    checked_at=checked,
                    content_sha256=content_sha256(chunk),
                    chunk_id=chunk_id,
                )
            )
    return [asdict(document) for document in documents]


def main() -> None:
    root = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=str(root / "data" / "sources.json"))
    parser.add_argument("--output", default=str(root / "data" / "generated" / "guidance.json"))
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = ingest_sources(load_source_manifest(args.sources))
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(payload)} provenance-checked chunks to {output}")


if __name__ == "__main__":
    main()
