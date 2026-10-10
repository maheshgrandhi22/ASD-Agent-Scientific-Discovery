import unittest
from unittest.mock import patch

from input_validator import validate_research_question
from src.agent.discovery_agent import ScientificDiscoveryAgent


class TestResearchQuestionValidator(unittest.TestCase):
    def test_valid_question(self):
        valid, _ = validate_research_question(
            "How can AI improve drug discovery?"
        )
        self.assertTrue(valid)

    def test_empty_question(self):
        valid, message = validate_research_question("   ")
        self.assertFalse(valid)
        self.assertIn("empty", message.lower())

    def test_question_too_short(self):
        valid, _ = validate_research_question("AI?")
        self.assertFalse(valid)

    def test_question_too_long(self):
        valid, _ = validate_research_question("A" * 501)
        self.assertFalse(valid)

    def test_non_text_input(self):
        valid, message = validate_research_question(None)
        self.assertFalse(valid)
        self.assertIn("text", message.lower())

    def test_agent_rejects_invalid_question(self):
        agent = ScientificDiscoveryAgent()
        with patch("builtins.print") as mock_print:
            result = agent.run("AI?")
        self.assertFalse(result)
        self.assertTrue(mock_print.called)

    def test_agent_accepts_valid_question(self):
        agent = ScientificDiscoveryAgent()
        with patch("builtins.print"):
            result = agent.run(
                "How can AI improve drug discovery?"
            )
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
