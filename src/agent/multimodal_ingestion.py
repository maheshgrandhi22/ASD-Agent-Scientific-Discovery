import hashlib
import re
from pathlib import Path

import pymupdf


class MultimodalPDFIngestor:
    """Extract text and embedded figures from scientific PDF documents."""

    def __init__(self, image_directory="data/extracted_images"):
        self.image_directory = Path(image_directory)

    def extract(self, pdf_path):
        """Extract page text, figures, and figure-caption references."""
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("Input file must be a PDF.")

        self.image_directory.mkdir(parents=True, exist_ok=True)

        pages = []
        figures = []

        with pymupdf.open(pdf_path) as document:
            for page_index, page in enumerate(document):
                page_number = page_index + 1
                page_text = page.get_text("text").strip()

                captions = re.findall(
                    r"(?im)^\s*(?:Figure|Fig\.?)\s*\d+[:. -].*$",
                    page_text,
                )

                pages.append({
                    "page_number": page_number,
                    "text": page_text,
                    "figure_captions": captions,
                })

                for image_index, image_info in enumerate(
                    page.get_images(full=True),
                    start=1,
                ):
                    xref = image_info[0]

                    try:
                        extracted = document.extract_image(xref)
                        image_bytes = extracted["image"]
                        extension = extracted.get("ext", "bin")

                        digest = hashlib.sha256(
                            image_bytes
                        ).hexdigest()[:16]

                        filename = (
                            f"{pdf_path.stem}_page_{page_number}_"
                            f"image_{image_index}_{digest}.{extension}"
                        )

                        image_path = self.image_directory / filename
                        image_path.write_bytes(image_bytes)

                        figures.append({
                            "page_number": page_number,
                            "image_index": image_index,
                            "image_path": str(image_path),
                            "width": extracted.get("width"),
                            "height": extracted.get("height"),
                            "extension": extension,
                            "sha256": hashlib.sha256(
                                image_bytes
                            ).hexdigest(),
                        })

                    except Exception as exc:
                        figures.append({
                            "page_number": page_number,
                            "image_index": image_index,
                            "error": str(exc),
                        })

        return {
            "source_pdf": str(pdf_path),
            "total_pages": len(pages),
            "total_figures": len(figures),
            "pages": pages,
            "figures": figures,
        }
