import json
from pathlib import Path


class ResultExporter:
    """Export scientific research results to JSON."""

    def export_json(self, data, output_path="research_results.json"):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

        return str(path)
