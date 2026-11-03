import unittest
from unittest.mock import Mock, patch

import requests

from src.agent.gemini_client import GeminiClient


class TestGeminiClient(unittest.TestCase):

    def test_missing_api_key_is_rejected(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                GeminiClient(api_key=None)

    def test_empty_prompt_is_rejected(self):
        client = GeminiClient(api_key="test-key")

        with self.assertRaises(ValueError):
            client.generate_text(" ")

    @patch("src.agent.gemini_client.requests.post")
    def test_successful_text_generation(self, mock_post):
        response = Mock()
        response.json.return_value = {
            "steps": [
                {
                    "type": "thought",
                },
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "text",
                            "text": "Scientific discovery improved.",
                        }
                    ],
                },
            ]
        }
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        client = GeminiClient(api_key="test-key")
        result = client.generate_text("Analyze scientific discovery.")

        self.assertEqual(result, "Scientific discovery improved.")

        mock_post.assert_called_once()
        self.assertEqual(
            mock_post.call_args.kwargs["headers"]["x-goog-api-key"],
            "test-key",
        )
        self.assertEqual(
            mock_post.call_args.kwargs["json"]["model"],
            "gemini-3.5-flash-lite",
        )

    @patch("src.agent.gemini_client.requests.post")
    def test_api_error_is_propagated(self, mock_post):
        response = Mock()
        response.raise_for_status.side_effect = (
            requests.HTTPError("Gemini unavailable")
        )
        mock_post.return_value = response

        client = GeminiClient(api_key="test-key")

        with self.assertRaises(requests.HTTPError):
            client.generate_text("Analyze scientific discovery.")

    @patch("src.agent.gemini_client.requests.post")
    def test_empty_model_response_is_rejected(self, mock_post):
        response = Mock()
        response.json.return_value = {"steps": []}
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        client = GeminiClient(api_key="test-key")

        with self.assertRaises(ValueError):
            client.generate_text("Analyze scientific discovery.")


if __name__ == "__main__":
    unittest.main()
