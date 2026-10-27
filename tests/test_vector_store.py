
import tempfile
import unittest

from src.agent.vector_store import ScientificVectorStore


class TestScientificVectorStore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = ScientificVectorStore(
            persist_directory=self.temp_dir.name,
            collection_name="test_papers",
        )
        self.reopened_store = None

    def tearDown(self):
        # Close all ChromaDB clients before deleting temporary files.
        if self.reopened_store is not None:
            self.reopened_store.client.close()
            self.reopened_store = None

        if self.store is not None:
            self.store.client.close()

        self.temp_dir.cleanup()

    @staticmethod
    def sample_papers():
        return [
            {
                "id": "paper-1",
                "title": "Artificial Intelligence in Drug Discovery",
                "abstract": (
                    "Machine learning can help identify drug candidates "
                    "and predict molecular properties."
                ),
                "authors": ["Author A"],
                "published": "2025",
                "url": "https://example.org/paper-1",
                "source": "PubMed",
            },
            {
                "id": "paper-2",
                "title": "Climate Change and Ocean Temperatures",
                "abstract": (
                    "Ocean temperature measurements help researchers "
                    "understand global climate change."
                ),
                "authors": ["Author B"],
                "published": "2024",
                "url": "https://example.org/paper-2",
                "source": "arXiv",
            },
        ]

    def test_add_papers_stores_records(self):
        count = self.store.add_papers(self.sample_papers())

        self.assertEqual(count, 2)
        self.assertEqual(self.store.collection.count(), 2)

    def test_empty_paper_list_is_supported(self):
        self.assertEqual(self.store.add_papers([]), 0)
        self.assertEqual(self.store.collection.count(), 0)

    def test_search_returns_relevant_paper_records(self):
        self.store.add_papers(self.sample_papers())

        results = self.store.search(
            "machine learning for drug discovery"
        )

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(
            results[0]["title"],
            "Artificial Intelligence in Drug Discovery",
        )
        self.assertIn("document", results[0])
        self.assertIn("distance", results[0])
        self.assertIn("url", results[0])

    def test_upsert_updates_existing_paper(self):
        paper = self.sample_papers()[0]
        self.store.add_papers([paper])

        updated_paper = {
            **paper,
            "title": "Updated Drug Discovery Research",
            "abstract": "Updated findings about molecular prediction.",
        }

        self.store.add_papers([updated_paper])

        self.assertEqual(self.store.collection.count(), 1)

        result = self.store.collection.get(
            ids=["paper-1"],
            include=["metadatas"],
        )

        self.assertEqual(
            result["metadatas"][0]["title"],
            "Updated Drug Discovery Research",
        )

    def test_search_rejects_empty_query(self):
        with self.assertRaises(ValueError):
            self.store.search("   ")

    def test_search_rejects_invalid_result_limit(self):
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    self.store.search(
                        "drug discovery",
                        n_results=limit,
                    )

    def test_add_papers_rejects_non_list(self):
        with self.assertRaises(ValueError):
            self.store.add_papers("not a list")

    def test_add_papers_rejects_invalid_record(self):
        with self.assertRaises(ValueError):
            self.store.add_papers(["not a paper"])

    def test_add_papers_rejects_missing_content(self):
        with self.assertRaises(ValueError):
            self.store.add_papers([{"id": "empty-paper"}])

    def test_add_papers_rejects_duplicate_ids_in_batch(self):
        duplicate_papers = [
            {"id": "same-id", "title": "Paper One"},
            {"id": "same-id", "title": "Paper Two"},
        ]

        with self.assertRaises(ValueError):
            self.store.add_papers(duplicate_papers)

    def test_database_persists_after_reopening(self):
        self.store.add_papers(self.sample_papers())

        # Close the first client before opening the same database again.
        self.store.client.close()

        self.reopened_store = ScientificVectorStore(
            persist_directory=self.temp_dir.name,
            collection_name="test_papers",
        )

        self.assertEqual(self.reopened_store.collection.count(), 2)

        results = self.reopened_store.search(
            "climate and ocean temperatures"
        )

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(
            results[0]["title"],
            "Climate Change and Ocean Temperatures",
        )


if __name__ == "__main__":
    unittest.main()
