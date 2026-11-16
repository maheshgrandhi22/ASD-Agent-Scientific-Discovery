import tempfile
import unittest

from src.agent.publication_verifier import PublicationVerifier
from src.agent.vector_store import ScientificVectorStore


class FakePublicationProvider:
    """Test-only provider with an explicit DOI-to-title registry."""

    def __init__(self):
        self.records = {
            "10.1234/paper.one": "Artificial Intelligence in Drug Discovery",
            "10.1234/paper.two": "Climate Change and Ocean Temperatures",
            "10.1234/duplicate.one": "Duplicate Test Paper One",
            "10.1234/duplicate.two": "Duplicate Test Paper Two",
        }
        self.fail = False
        self.calls = []

    def verify_publication(self, paper):
        self.calls.append(dict(paper))

        if self.fail:
            raise RuntimeError("Simulated provider outage")

        doi = paper.get("doi", "").strip().lower()
        registered_title = self.records.get(doi)

        if registered_title is None:
            return {
                "peer_reviewed": False,
                "reason": "publication_not_found_in_test_registry",
            }

        return {
            "peer_reviewed": True,
            "matched_identifier": doi,
            "publication_title": registered_title,
            "publication_venue": "Journal of Test Science",
            "evidence_url": f"https://registry.example.org/record/{doi}",
        }


class TestScientificVectorStore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.provider = FakePublicationProvider()
        self.verifier = PublicationVerifier(
            provider=self.provider,
            provider_name="Unit-test publication registry",
        )
        self.store = ScientificVectorStore(
            persist_directory=self.temp_dir.name,
            collection_name="test_papers",
            publication_verifier=self.verifier,
        )
        self.reopened_store = None

    def tearDown(self):
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
                "doi": "10.1234/paper.one",
                "peer_review_status": "verified",
                "peer_reviewed_verified": True,
                "verification_source": "Unit-test publication registry",
                "verified_identifier": "10.1234/paper.one",
                "verified_publication_title": "Artificial Intelligence in Drug Discovery",
                "verified_publication_venue": "Journal of Test Science",
                "verification_evidence_url": "https://registry.example.org/record/1",
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
                "doi": "10.1234/paper.two",
                "peer_review_status": "verified",
                "peer_reviewed_verified": True,
                "verification_source": "Unit-test publication registry",
                "verified_identifier": "10.1234/paper.two",
                "verified_publication_title": "Climate Change and Ocean Temperatures",
                "verified_publication_venue": "Journal of Test Science",
                "verification_evidence_url": "https://registry.example.org/record/2",
            },
        ]

    def test_store_requires_publication_verifier(self):
        with self.assertRaisesRegex(ValueError, "publication_verifier"):
            ScientificVectorStore(
                persist_directory=self.temp_dir.name,
                collection_name="no_verifier",
            )

    def test_add_papers_stores_records(self):
        count = self.store.add_papers(self.sample_papers())

        self.assertEqual(count, 2)
        self.assertEqual(self.store.collection.count(), 2)
        self.assertEqual(len(self.provider.calls), 2)

    def test_add_papers_rejects_unverified_record(self):
        with self.assertRaisesRegex(ValueError, "verification failed"):
            self.store.add_papers([{
                "id": "unverified-1",
                "title": "Unverified paper",
                "abstract": "This record has no DOI or provider evidence.",
            }])

        self.assertEqual(self.store.collection.count(), 0)

    def test_add_papers_rejects_stale_verified_title(self):
        paper = self.sample_papers()[0]
        paper["title"] = "A different title from the verified publication"

        with self.assertRaisesRegex(ValueError, "verification failed"):
            self.store.add_papers([paper])

        self.assertEqual(self.store.collection.count(), 0)

    def test_fabricated_verification_metadata_is_not_sufficient(self):
        paper = self.sample_papers()[0]
        paper["doi"] = "10.9999/fabricated.paper"

        with self.assertRaisesRegex(ValueError, "verification failed"):
            self.store.add_papers([paper])

        self.assertEqual(self.store.collection.count(), 0)

    def test_empty_paper_list_is_supported(self):
        self.assertEqual(self.store.add_papers([]), 0)
        self.assertEqual(self.store.collection.count(), 0)

    def test_search_returns_relevant_paper_records(self):
        self.store.add_papers(self.sample_papers())

        results = self.store.search("machine learning for drug discovery")

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(
            results[0]["title"],
            "Artificial Intelligence in Drug Discovery",
        )
        self.assertIn("document", results[0])
        self.assertIn("distance", results[0])
        self.assertIn("url", results[0])

    def test_search_excludes_legacy_records_without_verification_metadata(self):
        self.store.collection.upsert(
            ids=["legacy-paper"],
            documents=["Title: Legacy paper\nAbstract: Legacy abstract."],
            metadatas=[{
                "title": "Legacy paper",
                "url": "https://example.org/legacy",
                "source": "PubMed",
                "published": "2020",
                "authors": "Legacy Author",
            }],
        )

        results = self.store.search("legacy paper")
        self.assertEqual(results, [])

    def test_search_rejects_fabricated_stored_verification_metadata(self):
        self.store.collection.upsert(
            ids=["fabricated-paper"],
            documents=["Title: Fabricated paper\nAbstract: Unverified abstract."],
            metadatas=[{
                "title": "Fabricated paper",
                "url": "https://example.org/fabricated",
                "source": "Test",
                "published": "2025",
                "authors": "Unknown Author",
                "doi": "10.9999/fabricated.paper",
                "peer_review_status": "verified",
                "peer_reviewed_verified": True,
                "verification_source": "Fake local label",
                "verified_identifier": "10.9999/fabricated.paper",
                "verified_publication_title": "Fabricated paper",
                "verified_publication_venue": "Fake Journal",
                "verification_evidence_url": "https://fake.example.org/record",
            }],
        )

        results = self.store.search("fabricated paper")
        self.assertEqual(results, [])

    def test_search_reverifies_candidates_with_provider(self):
        self.store.add_papers([self.sample_papers()[0]])
        calls_before_search = len(self.provider.calls)

        results = self.store.search("machine learning drug discovery")

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(len(self.provider.calls), calls_before_search + 1)

    def test_provider_failure_during_search_returns_no_evidence(self):
        self.store.add_papers([self.sample_papers()[0]])
        self.provider.fail = True

        results = self.store.search("machine learning drug discovery")

        self.assertEqual(results, [])

    def test_search_preserves_publication_verification_metadata(self):
        paper = self.sample_papers()[0]
        self.store.add_papers([paper])

        results = self.store.search("machine learning drug discovery")

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(
            results[0]["verified_identifier"],
            paper["verified_identifier"],
        )
        self.assertEqual(
            results[0]["verification_evidence_url"],
            "https://registry.example.org/record/10.1234/paper.one",
        )

    def test_search_restores_abstract_from_stored_document(self):
        paper = self.sample_papers()[0]
        self.store.add_papers([paper])

        results = self.store.search("machine learning drug discovery")

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["abstract"], paper["abstract"])
        self.assertEqual(results[0]["summary"], paper["abstract"])

    def test_upsert_updates_existing_paper(self):
        paper = self.sample_papers()[0]
        self.store.add_papers([paper])

        updated_paper = {
            **paper,
            "title": "Updated Drug Discovery Research",
            "abstract": "Updated findings about molecular prediction.",
        }
        self.provider.records["10.1234/paper.one"] = (
            "Updated Drug Discovery Research"
        )

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
                    self.store.search("drug discovery", n_results=limit)

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
            {
                "id": "same-id",
                "title": "Duplicate Test Paper One",
                "abstract": "First test paper.",
                "doi": "10.1234/duplicate.one",
            },
            {
                "id": "same-id",
                "title": "Duplicate Test Paper Two",
                "abstract": "Second test paper.",
                "doi": "10.1234/duplicate.two",
            },
        ]

        with self.assertRaisesRegex(ValueError, "unique"):
            self.store.add_papers(duplicate_papers)

    def test_database_persists_after_reopening(self):
        self.store.add_papers(self.sample_papers())

        self.store.client.close()

        self.reopened_store = ScientificVectorStore(
            persist_directory=self.temp_dir.name,
            collection_name="test_papers",
            publication_verifier=self.verifier,
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
