import hashlib
import re
from pathlib import Path

import pymupdf


class MultimodalPDFIngestor:
    """Extract PDF text and figures, with optional Gemini image analysis."""

    def __init__(
        self,
        image_directory="data/extracted_images",
        gemini_client=None,
    ):
        self.image_directory = Path(image_directory)
        self.gemini_client = gemini_client

    def _analyze_figure(self, image_path):
        """Return a clearly labelled, unverified AI interpretation."""
        if self.gemini_client is None:
            return {
                "status": "not_configured",
                "interpretation": None,
                "scientific_validation": "not_performed",
            }

        prompt = (
            "You are analyzing an extracted figure from a scientific paper. "
            "Describe only what is visibly supported by the image. "
            "Identify the likely chart or figure type, visible axes and "
            "units, legend or groups, observable trends, comparisons, "
            "and any readable labels. State when information is unclear "
            "or unavailable. Do not invent numerical values, infer "
            "causation, or claim statistical significance unless it is "
            "explicitly visible in the figure. Distinguish observations "
            "from possible interpretations. Mention image-quality "
            "limitations. This is an AI-generated description, not "
            "independent scientific validation."
        )

        try:
            interpretation = self.gemini_client.analyze_image(
                image_path,
                prompt,
            )

            return {
                "status": "completed",
                "interpretation": interpretation,
                "generated_by": "Gemini",
                "scientific_validation": "not_independently_verified",
            }

        except Exception as exc:
            return {
                "status": "failed",
                "interpretation": None,
                "error": f"{type(exc).__name__}: {exc}",
                "scientific_validation": "not_performed",
            }

    def extract(self, pdf_path):
        """Extract page text, figures, captions, and optional interpretations."""
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
                        ).hexdigest()

                        filename = (
                            f"{pdf_path.stem}_page_{page_number}_"
                            f"image_{image_index}_{digest[:16]}.{extension}"
                        )

                        image_path = self.image_directory / filename
                        image_path.write_bytes(image_bytes)

                        figure = {
                            "page_number": page_number,
                            "image_index": image_index,
                            "image_path": str(image_path),
                            "width": extracted.get("width"),
                            "height": extracted.get("height"),
                            "extension": extension,
                            "sha256": digest,
                        }

                        figure["image_analysis"] = self._analyze_figure(
                            image_path
                        )

                        figures.append(figure)

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
