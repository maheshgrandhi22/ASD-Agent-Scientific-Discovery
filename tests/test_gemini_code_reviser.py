import unittest
from unittest.mock import Mock

from src.agent.gemini_code_reviser import GeminiCodeReviser


class TestGeminiCodeReviser(unittest.TestCase):

    def setUp(self):
        self.client = Mock()
        self.reviser = GeminiCodeReviser(self.client)

    def test_returns_revised_python_code(self):
        self.client.generate_text.return_value = "print(2 + 2)"

        result = self.reviser("print(2 + missing)", "NameError", 1)

        self.assertEqual(result, "print(2 + 2)")
        self.client.generate_text.assert_called_once()

    def test_removes_python_markdown_fence(self):
        self.client.generate_text.return_value = (
            "```python\nprint('fixed')\n```"
        )

        result = self.reviser("print(undefined)", "NameError", 1)

        self.assertEqual(result, "print('fixed')")

    def test_prompt_includes_code_error_and_attempt(self):
        self.client.generate_text.return_value = "print('fixed')"

        self.reviser("print(undefined)", "NameError", 2)

        prompt = self.client.generate_text.call_args.args[0]
        self.assertIn("print(undefined)", prompt)
        self.assertIn("NameError", prompt)
        self.assertIn("Revision attempt: 2", prompt)
        self.assertIn("untrusted data", prompt)

    def test_empty_response_is_rejected(self):
        self.client.generate_text.return_value = "   "

        with self.assertRaises(ValueError):
            self.reviser("print(undefined)", "NameError", 1)

    def test_unchanged_code_is_rejected(self):
        self.client.generate_text.return_value = "print(undefined)"

        with self.assertRaises(ValueError):
            self.reviser("print(undefined)", "NameError", 1)

    def test_invalid_python_is_rejected(self):
        self.client.generate_text.return_value = "def broken("

        with self.assertRaises(ValueError):
            self.reviser("print(undefined)", "NameError", 1)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.reviser("", "NameError", 1)

        with self.assertRaises(ValueError):
            self.reviser("print('x')", "NameError", 0)

    def test_client_errors_are_propagated(self):
        self.client.generate_text.side_effect = RuntimeError("API unavailable")

        with self.assertRaises(RuntimeError):
            self.reviser("print(undefined)", "NameError", 1)


if __name__ == "__main__":
    unittest.main()
