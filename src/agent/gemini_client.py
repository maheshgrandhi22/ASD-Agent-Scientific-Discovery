import base64
import mimetypes
import os
from pathlib import Path

import requests


class GeminiClient:
    """Optional Gemini API client using Google's Interactions API."""

    ENDPOINT = (
        "https://generativelanguage.googleapis.com/v1beta/interactions"
    )

    MAX_INLINE_IMAGE_BYTES = 14 * 1024 * 1024

    SUPPORTED_IMAGE_TYPES = {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
    }

    def __init__(
        self,
        api_key=None,
        model="gemini-3.5-flash-lite",
        timeout=60,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        if not isinstance(model, str) or not model.strip():
            raise ValueError("A valid Gemini model name is required.")

        if not isinstance(timeout, int) or timeout < 1:
            raise ValueError("Timeout must be a positive integer.")

    def _extract_text(self, data):
        """Extract usable text from model_output steps."""
        text_parts = []

        for step in data.get("steps", []):
            if step.get("type") != "model_output":
                continue

            for item in step.get("content", []):
                if item.get("type") == "text":
                    value = item.get("text", "")
                    if isinstance(value, str) and value.strip():
                        text_parts.append(value.strip())

        if not text_parts:
            raise ValueError(
                "Gemini returned no usable text in its response."
            )

        return "\n".join(text_parts)

    def generate_text(self, prompt):
        """Generate text and return the model's textual response."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        response = requests.post(
            self.ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            json={
                "model": self.model,
                "input": prompt,
            },
            timeout=self.timeout,
        )

        response.raise_for_status()
        return self._extract_text(response.json())

    def analyze_image(self, image_path, prompt):
        """Analyze a local image using Gemini's multimodal input."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Image analysis prompt cannot be empty.")

        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        mime_type, _ = mimetypes.guess_type(str(image_path))

        if mime_type not in self.SUPPORTED_IMAGE_TYPES:
            raise ValueError(
                f"Unsupported image MIME type: {mime_type}"
            )

        image_bytes = image_path.read_bytes()

        if not image_bytes:
            raise ValueError("Image file is empty.")

        if len(image_bytes) > self.MAX_INLINE_IMAGE_BYTES:
            raise ValueError(
                "Image exceeds the safe inline image size limit. "
                "Use the Gemini Files API for larger images."
            )

        image_base64 = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        response = requests.post(
            self.ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            json={
                "model": self.model,
                "input": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image",
                        "data": image_base64,
                        "mime_type": mime_type,
                    },
                ],
            },
            timeout=self.timeout,
        )

        response.raise_for_status()
        return self._extract_text(response.json())
