
import json
import os
import tempfile
from pathlib import Path


class ResultExporter:
    """Export scientific research results to JSON safely."""

    def export_json(self, data, output_path="research_results.json"):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temp_path = Path(file.name)
                json.dump(data, file, indent=4, ensure_ascii=False)
                file.write("\n")

            os.replace(temp_path, path)
            return str(path)

        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
