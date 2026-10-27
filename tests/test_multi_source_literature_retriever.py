
import unittest
from unittest.mock import Mock

import requests

from src.agent.multi_source_literature_retriever import (
    MultiSourceLiteratureRetriever,
)


class TestMultiSourceLiteratureRetriever(unittest.TestCase):
    def setUp(self):
        self.arxiv = Mock()
        self.pubmed = Mock()
        self.retriever = MultiSourceLiteratureRetriever(
            arxiv=self.arxiv,
            pubmed=self.pubmed,
        )

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.search(" ")

    def test_invalid_limit_is_rejected(self):
        for value in (0, -1, True, 1.5, "3"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.retriever.search("AI discovery", value)

    def test_results_from_both_sources_are_normalized(self):
        self.arxiv.search.return_value = [
            {
                "id": "http://arxiv.org/abs/1234.5678",
                "title": "AI Research",
                "summary": "An arXiv abstract.",
                "authors": ["A. Researcher"],
                "published": "2026-01-01",
            }
        ]
        self.pubmed.search.return_value = [
            {
                "id": "PMID:12345",
                "title": "Medical AI",
                "abstract": "A PubMed abstract.",
                "authors": ["B. Researcher"],
                "published": "2025",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
                "source": "PubMed",
            }
        ]

        papers = self.retriever.search("AI discovery", max_results=5)

        self.assertEqual(len(papers), 2)
        self.assertEqual(papers[0]["abstract"], "An arXiv abstract.")
        self.assertEqual(papers[0]["source"], "arXiv")
        self.assertEqual(papers[1]["abstract"], "A PubMed abstract.")
        self.assertEqual(papers[1]["source"], "PubMed")
        self.assertTrue(papers[0]["url"].startswith("http"))

    def test_one_source_failure_preserves_other_results(self):
        self.arxiv.search.side_effect = requests.ConnectionError("offline")
        self.pubmed.search.return_value = [
            {
                "id": "PMID:123",
                "title": "PubMed paper",
                "abstract": "A scientific abstract.",
            }
        ]

        papers = self.retriever.search("science")

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["source"], "PubMed")

    def test_all_sources_failing_raises_error(self):
        self.arxiv.search.side_effect = requests.ConnectionError("offline")
        self.pubmed.search.side_effect = requests.Timeout("timed out")

        with self.assertRaisesRegex(RuntimeError, "All literature sources"):
            self.retriever.search("science")

    def test_result_limit_is_applied(self):
        self.arxiv.search.return_value = [
            {"id": "arxiv-1", "title": "Paper 1"},
            {"id": "arxiv-2", "title": "Paper 2"},
        ]
        self.pubmed.search.return_value = [
            {"id": "PMID:3", "title": "Paper 3"}
        ]

        papers = self.retriever.search("science", max_results=2)

        self.assertEqual(len(papers), 2)


if __name__ == "__main__":
    unittest.main()
