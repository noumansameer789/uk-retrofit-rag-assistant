import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from retrofit_rag.retrieval import Document, Retriever, load_catalogue


class RetrievalTest(unittest.TestCase):
    def test_relevant_source_ranks_first(self):
        docs = [
            Document("Heat", "https://example/heat", "heat pump boiler low carbon heating grant"),
            Document("EPC", "https://example/epc", "energy performance certificate rating"),
        ]
        self.assertEqual(Retriever(docs).search("heat pump grant")[0][0].title, "Heat")

    def test_catalogue_has_citations(self):
        docs = load_catalogue(Path(__file__).parents[1] / "data" / "guidance.json")
        self.assertTrue(all(doc.url.startswith("https://") for doc in docs))
        self.assertTrue(all(doc.content_sha256 for doc in docs))

    def test_unrelated_query_abstains(self):
        docs = [Document("Heat", "https://example/heat", "heat pump boiler grant")]
        self.assertEqual(Retriever(docs).search("football score yesterday"), [])

    def test_invalid_k_is_rejected(self):
        docs = [Document("Heat", "https://example/heat", "heat pump boiler grant")]
        with self.assertRaises(ValueError):
            Retriever(docs).search("heat", k=0)

    def test_invalid_bm25_parameters_are_rejected(self):
        docs = [Document("Heat", "https://example/heat", "heat pump boiler grant")]
        with self.assertRaises(ValueError):
            Retriever(docs, b=2)

    def test_equal_scores_have_deterministic_url_order(self):
        docs = [
            Document("Heat", "https://example/z", "heat pump"),
            Document("Heat", "https://example/a", "heat pump"),
        ]
        returned = Retriever(docs).search("heat pump")
        self.assertEqual([document.url for document, _ in returned], ["https://example/a", "https://example/z"])

    def test_catalogue_rejects_digest_mismatch(self):
        payload = json.loads(
            (Path(__file__).parents[1] / "data" / "guidance.json").read_text()
        )
        payload[0]["content_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guidance.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_catalogue(path)

    def test_catalogue_rejects_untrusted_urls(self):
        payload = [{"title": "Bad", "url": "https://example.com", "text": "text"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guidance.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_catalogue(path)


if __name__ == "__main__":
    unittest.main()
