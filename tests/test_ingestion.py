import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from retrofit_rag.ingestion import (
    AllowlistRedirectHandler,
    SourceSpec,
    chunk_text,
    extract_text,
    fetch_official_html,
    ingest_sources,
    load_source_manifest,
)


class IngestionTest(unittest.TestCase):
    def test_extracts_content_without_navigation_or_script(self):
        html = (ROOT / "tests" / "fixtures" / "guidance_page.html").read_text()
        text = extract_text(html)
        self.assertIn("energy certificate", text)
        self.assertNotIn("Navigation", text)
        self.assertNotIn("ignore previous", text)
        self.assertNotIn("Footer", text)

    def test_chunking_is_deterministic_and_overlapping(self):
        words = " ".join(f"word{index}" for index in range(50))
        chunks = chunk_text(words, words_per_chunk=20, overlap=5)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0].split()[-5:], chunks[1].split()[:5])

    def test_non_allowlisted_fetch_is_rejected_before_network(self):
        with patch("retrofit_rag.ingestion.OFFICIAL_SOURCE_OPENER.open") as mocked:
            with self.assertRaises(ValueError):
                fetch_official_html("https://example.com/guidance")
        mocked.assert_not_called()

    def test_off_list_redirect_is_rejected_before_follow(self):
        handler = AllowlistRedirectHandler()
        with self.assertRaises(ValueError):
            handler.redirect_request(
                request=None,
                file_pointer=None,
                code=302,
                message="Found",
                headers={},
                new_url="https://example.com/redirect",
            )

    def test_manifest_and_ingestion_add_provenance(self):
        sources = load_source_manifest(ROOT / "data" / "sources.json")
        self.assertEqual(len(sources), 4)
        html = (ROOT / "tests" / "fixtures" / "guidance_page.html").read_text()
        with patch("retrofit_rag.ingestion.fetch_official_html", return_value=html):
            payload = ingest_sources([sources[0]], checked_at="2026-09-01")
        self.assertTrue(payload)
        self.assertEqual(payload[0]["source_id"], "govuk-epc")
        self.assertEqual(len(payload[0]["content_sha256"]), 64)

    def test_duplicate_source_id_is_rejected(self):
        payload = [
            {"source_id": "same", "title": "One", "url": "https://www.gov.uk/one"},
            {"source_id": "same", "title": "Two", "url": "https://www.gov.uk/two"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_source_manifest(path)

    def test_source_spec_is_explicit(self):
        source = SourceSpec("govuk-test", "Test", "https://www.gov.uk/test")
        self.assertEqual(source.source_id, "govuk-test")


if __name__ == "__main__":
    unittest.main()
