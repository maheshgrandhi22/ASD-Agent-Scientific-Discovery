import os

import requests


class GeminiClient:
    """Optional Gemini API client using Google's Interactions API."""

    ENDPOINT = (
        "https://generativelanguage.googleapis.com/v1beta/interactions"
    )

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
        data = response.json()

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
