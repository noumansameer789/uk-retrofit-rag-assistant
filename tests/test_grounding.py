import unittest

from retrofit_rag.safety import OutputValidationError, validate_grounded_quotes


class GroundingTest(unittest.TestCase):
    def setUp(self):
        self.evidence = {1: "Installers apply for owners. The grant is not guaranteed.",
                         2: "An energy certificate rates the property's energy efficiency."}

    def test_exact_quote_is_supported(self):
        validate_grounded_quotes("Installers apply for owners [1].", self.evidence)

    def test_changed_negation_is_rejected(self):
        with self.assertRaises(OutputValidationError):
            validate_grounded_quotes("The grant is guaranteed [1].", self.evidence)

    def test_invented_amount_is_rejected(self):
        with self.assertRaises(OutputValidationError):
            validate_grounded_quotes("Installers apply for 9000 pounds [1].", self.evidence)

    def test_correct_text_with_wrong_source_is_rejected(self):
        with self.assertRaises(OutputValidationError):
            validate_grounded_quotes("Installers apply for owners [2].", self.evidence)

    def test_uncited_tail_is_rejected(self):
        with self.assertRaises(OutputValidationError):
            validate_grounded_quotes("Installers apply for owners [1]. Funding is guaranteed.", self.evidence)

    def test_two_separately_cited_quotes(self):
        validate_grounded_quotes("Installers apply for owners [1]. An energy certificate rates the property's energy efficiency [2].", self.evidence)
