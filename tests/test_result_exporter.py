import json
import tempfile
import unittest
from pathlib import Path

from src.agent.result_exporter import ResultExporter


class TestResultExporter(unittest.TestCase):
    def test_export_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results.json"
            data = {"question": "How can AI improve drug discovery?"}

            result = ResultExporter().export_json(data, output)

            self.assertTrue(Path(result).exists())
            with open(result, encoding="utf-8") as file:
                saved_data = json.load(file)

            self.assertEqual(saved_data, data)


if __name__ == "__main__":
    unittest.main()
