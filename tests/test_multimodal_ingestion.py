import base64
import tempfile
import unittest
from pathlib import Path

import pymupdf

from src.agent.multimodal_ingestion import MultimodalPDFIngestor


class TestMultimodalPDFIngestor(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.pdf_path = self.root / "sample_research.pdf"
        self.image_dir = self.root / "extracted_images"

        self._create_sample_pdf()

        self.ingestor = MultimodalPDFIngestor(
            image_directory=self.image_dir
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_sample_pdf(self):
        document = pymupdf.open()

        page_one = document.new_page()
        page_one.insert_text(
            (72, 72),
            "Scientific Research Paper\nFigure 1: Sample experiment results",
        )

        page_two = document.new_page()
        page_two.insert_text(
            (72, 72),
            "Second page: Experimental methodology and findings.",
        )

        # A small valid PNG used to test embedded-image extraction.
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/"
            "mXcAAAAASUVORK5CYII="
        )

        page_two.insert_image(
            pymupdf.Rect(72, 120, 172, 220),
            stream=png_data,
        )

        document.save(self.pdf_path)
        document.close()

    def test_extracts_all_pages(self):
        result = self.ingestor.extract(self.pdf_path)

        self.assertEqual(result["total_pages"], 2)
        self.assertEqual(len(result["pages"]), 2)

    def test_preserves_page_numbers_and_text(self):
        result = self.ingestor.extract(self.pdf_path)

        self.assertEqual(
            result["pages"][0]["page_number"],
            1,
        )
        self.assertIn(
            "Scientific Research Paper",
            result["pages"][0]["text"],
        )
        self.assertEqual(
            result["pages"][1]["page_number"],
            2,
        )

    def test_extracts_figure_captions(self):
        result = self.ingestor.extract(self.pdf_path)

        captions = result["pages"][0]["figure_captions"]

        self.assertTrue(
            any("Figure 1" in caption for caption in captions)
        )

    def test_extracts_embedded_images(self):
        result = self.ingestor.extract(self.pdf_path)

        self.assertGreaterEqual(result["total_figures"], 1)

        extracted_figures = [
            figure for figure in result["figures"]
            if "image_path" in figure
        ]

        self.assertGreaterEqual(len(extracted_figures), 1)

        for figure in extracted_figures:
            self.assertTrue(Path(figure["image_path"]).exists())
            self.assertEqual(figure["page_number"], 2)

    def test_missing_pdf_raises_error(self):
        missing_path = self.root / "missing.pdf"

        with self.assertRaises(FileNotFoundError):
            self.ingestor.extract(missing_path)

    def test_invalid_extension_raises_error(self):
        invalid_path = self.root / "document.txt"
        invalid_path.write_text("Not a PDF", encoding="utf-8")

        with self.assertRaises(ValueError):
            self.ingestor.extract(invalid_path)


if __name__ == "__main__":
    unittest.main()
