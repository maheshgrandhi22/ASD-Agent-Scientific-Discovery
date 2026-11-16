import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from src.agent.gemini_client import GeminiClient


class TestGeminiClient(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_response(self, text="Scientific discovery improved."):
        response = Mock()
        response.json.return_value = {
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                        }
                    ],
                }
            ]
        }
        response.raise_for_status.return_value = None
        return response

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
        mock_post.return_value = self.make_response()

        client = GeminiClient(api_key="test-key")
        result = client.generate_text("Analyze scientific discovery.")

        self.assertEqual(result, "Scientific discovery improved.")
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

    def test_image_analysis_rejects_empty_prompt(self):
        client = GeminiClient(api_key="test-key")
        image_path = self.root / "chart.png"
        image_path.write_bytes(b"fake-image")

        with self.assertRaises(ValueError):
            client.analyze_image(image_path, " ")

    def test_image_analysis_rejects_missing_file(self):
        client = GeminiClient(api_key="test-key")

        with self.assertRaises(FileNotFoundError):
            client.analyze_image(
                self.root / "missing.png",
                "Describe this chart.",
            )

    def test_image_analysis_rejects_unsupported_type(self):
        client = GeminiClient(api_key="test-key")
        image_path = self.root / "document.txt"
        image_path.write_text("not an image", encoding="utf-8")

        with self.assertRaises(ValueError):
            client.analyze_image(
                image_path,
                "Describe this chart.",
            )

    def test_image_analysis_rejects_empty_file(self):
        client = GeminiClient(api_key="test-key")
        image_path = self.root / "empty.png"
        image_path.write_bytes(b"")

        with self.assertRaises(ValueError):
            client.analyze_image(
                image_path,
                "Describe this chart.",
            )

    def test_image_analysis_rejects_oversized_file(self):
        client = GeminiClient(api_key="test-key")
        image_path = self.root / "large.png"
        image_path.write_bytes(b"x")

        with patch.object(
            client,
            "MAX_INLINE_IMAGE_BYTES",
            0,
        ):
            with self.assertRaises(ValueError):
                client.analyze_image(
                    image_path,
                    "Describe this chart.",
                )

    @patch("src.agent.gemini_client.requests.post")
    def test_successful_image_analysis_uses_base64_payload(
        self,
        mock_post,
    ):
        mock_post.return_value = self.make_response(
            "The chart shows an increasing trend."
        )

        image_bytes = b"sample-png-image-data"
        image_path = self.root / "chart.png"
        image_path.write_bytes(image_bytes)

        client = GeminiClient(api_key="test-key")
        result = client.analyze_image(
            image_path,
            "Describe the chart.",
        )

        self.assertEqual(
            result,
            "The chart shows an increasing trend.",
        )

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(
            payload["model"],
            "gemini-3.5-flash-lite",
        )
        self.assertEqual(len(payload["input"]), 2)

        text_item, image_item = payload["input"]

        self.assertEqual(text_item["type"], "text")
        self.assertEqual(text_item["text"], "Describe the chart.")
        self.assertEqual(image_item["type"], "image")
        self.assertEqual(image_item["mime_type"], "image/png")
        self.assertEqual(
            base64.b64decode(image_item["data"]),
            image_bytes,
        )

        self.assertEqual(
            mock_post.call_args.kwargs["headers"]["x-goog-api-key"],
            "test-key",
        )

    @patch("src.agent.gemini_client.requests.post")
    def test_image_analysis_api_error_is_propagated(
        self,
        mock_post,
    ):
        response = Mock()
        response.raise_for_status.side_effect = (
            requests.HTTPError("Gemini image analysis unavailable")
        )
        mock_post.return_value = response

        image_path = self.root / "chart.png"
        image_path.write_bytes(b"sample-image")

        client = GeminiClient(api_key="test-key")

        with self.assertRaises(requests.HTTPError):
            client.analyze_image(
                image_path,
                "Describe this chart.",
            )


if __name__ == "__main__":
    unittest.main()
