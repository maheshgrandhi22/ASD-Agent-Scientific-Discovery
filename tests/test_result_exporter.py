
import json
import tempfile
import unittest
from pathlib import Path

from src.agent.result_exporter import ResultExporter


class TestResultExporter(unittest.TestCase):
    def test_export_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "results.json"
            data = {
                "research_question": "How can AI improve drug discovery?",
                "analyzed_papers": [
                    {
                        "title": "AI for Drug Discovery",
                        "relevance_score": 85,
                        "key_findings": [
                            "AI can accelerate candidate screening."
                        ],
                    }
                ],
                "research_summary": {
                    "paper_count": 1,
                    "summary": "AI can accelerate candidate screening.",
                    "common_themes": [
                        {"theme": "drug", "paper_count": 1}
                    ],
                },
            }

            result = ResultExporter().export_json(data, output)

            self.assertTrue(Path(result).exists())
            self.assertTrue(output.parent.is_dir())

            with open(result, encoding="utf-8") as file:
                saved_data = json.load(file)

            self.assertEqual(saved_data, data)

    def test_export_json_rejects_non_serializable_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results.json"

            with self.assertRaises(TypeError):
                ResultExporter().export_json({"invalid": object()}, output)

    def test_export_json_rejects_invalid_destination(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            blocker = Path(temp_dir) / "not_a_directory"
            blocker.write_text("blocker", encoding="utf-8")
            output = blocker / "results.json"

            with self.assertRaises(OSError):
                ResultExporter().export_json({}, output)

    def test_failed_export_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results.json"
            original_data = {"status": "previous results"}
            output.write_text(
                json.dumps(original_data),
                encoding="utf-8",
            )

            with self.assertRaises(TypeError):
                ResultExporter().export_json(
                    {"invalid": object()},
                    output,
                )

            with output.open(encoding="utf-8") as file:
                saved_data = json.load(file)

            self.assertEqual(saved_data, original_data)


if __name__ == "__main__":
    unittest.main()
