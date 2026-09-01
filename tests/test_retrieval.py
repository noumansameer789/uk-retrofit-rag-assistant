import sys
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

    def test_unrelated_query_abstains(self):
        docs = [Document("Heat", "https://example/heat", "heat pump boiler grant")]
        self.assertEqual(Retriever(docs).search("football score yesterday"), [])

    def test_invalid_k_is_rejected(self):
        docs = [Document("Heat", "https://example/heat", "heat pump boiler grant")]
        with self.assertRaises(ValueError):
            Retriever(docs).search("heat", k=0)


if __name__ == "__main__":
    unittest.main()
