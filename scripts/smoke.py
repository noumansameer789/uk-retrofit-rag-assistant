"""End-to-end probe for a running API and model."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urlopen(request, timeout=90) as response:
        data = json.loads(response.read(1_000_001))
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response from {url}")
    return data


def wait_until_ready(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if request_json(f"{base_url}/ready").get("status") == "ready":
                return
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"API did not become ready within {timeout:g}s") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://api:8000")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    wait_until_ready(base_url, args.timeout)
    response = request_json(
        f"{base_url}/ask",
        {"question": "What document rates a property's energy efficiency?", "top_k": 3},
    )
    citations = response.get("citations")
    if response.get("status") != "answered" or not isinstance(citations, list) or not citations:
        raise RuntimeError(f"Grounded answer smoke check failed: {response}")
    if not all(isinstance(item, dict) and item.get("url") for item in citations):
        raise RuntimeError("Smoke answer did not include structured source URLs")
    print(json.dumps(response, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
