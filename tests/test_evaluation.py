import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from retrofit_rag.evaluation import evaluate_retrieval, load_cases
from retrofit_rag.retrieval import Retriever, load_catalogue


class EvaluationTest(unittest.TestCase):
    def test_included_smoke_set_meets_contract(self):
        metrics = evaluate_retrieval(
            Retriever(load_catalogue(ROOT / "data" / "guidance.json")),
            load_cases(ROOT / "data" / "evaluation.json"),
            k=3,
        )
        self.assertEqual(metrics.answerable_cases, 4)
        self.assertEqual(metrics.unanswerable_cases, 2)
        self.assertEqual(metrics.hit_rate_at_k, 1.0)
        self.assertEqual(metrics.mean_reciprocal_rank, 1.0)
        self.assertEqual(metrics.abstention_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
