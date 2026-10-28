
import unittest
from unittest.mock import Mock

from src.agent.agent_roles import (
    CodeExecutor,
    HypothesisGenerator,
    LiteratureReviewer,
)


class TestLiteratureReviewer(unittest.TestCase):
    def setUp(self):
        self.retriever = Mock()
        self.analyzer = Mock()

        self.reviewer = LiteratureReviewer(
            self.retriever,
            self.analyzer,
        )

        self.valid_paper = {
            "id": "PMID:123",
            "title": "AI in discovery",
            "abstract": "AI may improve screening.",
            "source": "PubMed",
            "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
        }

        self.analyzed_paper = {
            "id": "PMID:123",
            "title": "AI in discovery",
            "abstract": "AI may improve screening.",
            "source": "PubMed",
            "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
            "relevance_score": 80,
            "key_findings": ["Screening may improve."],
        }

    def test_review_returns_ranked_evidence(self):
        self.retriever.search.return_value = [self.valid_paper]
        self.analyzer.analyze.return_value = [self.analyzed_paper]

        result = self.reviewer.review(
            "How can AI improve discovery?"
        )

        self.assertEqual(result["role"], "Literature Reviewer")
        self.assertEqual(result["paper_count"], 1)
        self.assertEqual(
            result["evidence"][0]["relevance_score"],
            80,
        )
        self.assertEqual(
            result["evidence"][0]["paper_id"],
            "PMID:123",
        )

        self.retriever.search.assert_called_once_with(
            "How can AI improve discovery?",
            max_results=5,
        )

    def test_empty_research_question_is_rejected(self):
        with self.assertRaises(ValueError):
            self.reviewer.review(" ")

    def test_non_string_research_question_is_rejected(self):
        with self.assertRaises(ValueError):
            self.reviewer.review(None)

    def test_zero_max_results_is_rejected(self):
        with self.assertRaises(ValueError):
            self.reviewer.review(
                "AI in discovery",
                max_results=0,
            )

    def test_negative_max_results_is_rejected(self):
        with self.assertRaises(ValueError):
            self.reviewer.review(
                "AI in discovery",
                max_results=-1,
            )

    def test_boolean_max_results_is_rejected(self):
        with self.assertRaises(ValueError):
            self.reviewer.review(
                "AI in discovery",
                max_results=True,
            )

    def test_non_integer_max_results_is_rejected(self):
        with self.assertRaises(ValueError):
            self.reviewer.review(
                "AI in discovery",
                max_results="5",
            )

    def test_missing_retriever_and_papers_is_rejected(self):
        reviewer = LiteratureReviewer(
            retriever=None,
            analyzer=self.analyzer,
        )

        with self.assertRaises(ValueError):
            reviewer.review("AI in discovery")

    def test_non_list_papers_are_rejected(self):
        with self.assertRaises(ValueError):
            self.reviewer.review(
                "AI in discovery",
                papers="not a list",
            )

    def test_missing_analyzer_is_rejected(self):
        reviewer = LiteratureReviewer(
            retriever=self.retriever,
            analyzer=None,
        )

        with self.assertRaises(ValueError):
            reviewer.review(
                "AI in discovery",
                papers=[self.valid_paper],
            )

    def test_papers_without_usable_content_are_filtered(self):
        papers = [
            {},
            {"title": "   ", "abstract": ""},
            None,
            "invalid paper record",
            {"title": "", "abstract": "  ", "document": ""},
        ]

        self.analyzer.analyze.return_value = []

        result = self.reviewer.review(
            "  AI in discovery  ",
            papers=papers,
        )

        self.analyzer.analyze.assert_called_once_with(
            "AI in discovery",
            [],
        )
        self.assertEqual(result["research_question"], "AI in discovery")
        self.assertEqual(result["paper_count"], 0)
        self.assertEqual(result["evidence"], [])

    def test_papers_with_usable_content_are_retained(self):
        papers = [
            self.valid_paper,
            {"title": "A useful paper"},
            {"abstract": "Useful scientific evidence."},
            {"document": "Full scientific paper text."},
            {},
            None,
        ]

        self.analyzer.analyze.return_value = []

        self.reviewer.review(
            "AI in discovery",
            papers=papers,
        )

        self.analyzer.analyze.assert_called_once_with(
            "AI in discovery",
            papers[:4],
        )

    def test_non_dictionary_papers_are_ignored(self):
        papers = [
            None,
            "invalid record",
            123,
            self.valid_paper,
        ]

        self.analyzer.analyze.return_value = [self.analyzed_paper]

        result = self.reviewer.review(
            "AI in discovery",
            papers=papers,
        )

        self.analyzer.analyze.assert_called_once_with(
            "AI in discovery",
            [self.valid_paper],
        )
        self.assertEqual(result["paper_count"], 1)

    def test_non_dictionary_analyzer_results_are_ignored(self):
        self.analyzer.analyze.return_value = [
            None,
            "invalid result",
            self.analyzed_paper,
        ]

        result = self.reviewer.review(
            "AI in discovery",
            papers=[self.valid_paper],
        )

        self.assertEqual(result["paper_count"], 1)
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(
            result["evidence"][0]["paper_id"],
            "PMID:123",
        )

    def test_non_list_key_findings_become_empty_list(self):
        paper = {
            **self.analyzed_paper,
            "key_findings": "not a list",
        }
        self.analyzer.analyze.return_value = [paper]

        result = self.reviewer.review(
            "AI in discovery",
            papers=[self.valid_paper],
        )

        self.assertEqual(
            result["evidence"][0]["key_findings"],
            [],
        )


class TestHypothesisGenerator(unittest.TestCase):
    def test_hypotheses_include_source_evidence(self):
        generator = HypothesisGenerator()
        review = {
            "evidence": [
                {
                    "paper_id": "PMID:123",
                    "title": "AI in discovery",
                    "source": "PubMed",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
                    "key_findings": ["AI may improve screening."],
                }
            ]
        }

        result = generator.generate("AI in discovery", review)

        self.assertEqual(result["role"], "Hypothesis Generator")
        self.assertEqual(len(result["hypotheses"]), 1)
        self.assertEqual(
            result["hypotheses"][0]["supporting_paper_id"],
            "PMID:123",
        )
        self.assertEqual(
            result["hypotheses"][0]["validation_status"],
            "not experimentally validated",
        )

    def test_empty_evidence_produces_no_hypotheses(self):
        result = HypothesisGenerator().generate(
            "AI in discovery",
            {"evidence": []},
        )

        self.assertEqual(result["hypotheses"], [])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            HypothesisGenerator().generate(
                " ",
                {"evidence": []},
            )


class TestCodeExecutor(unittest.TestCase):
    def test_execution_is_blocked_without_sandbox(self):
        result = CodeExecutor().execute("print('hello')")

        self.assertEqual(result["status"], "blocked")
        self.assertIn("secure sandbox", result["error"])

    def test_empty_code_is_rejected(self):
        with self.assertRaises(ValueError):
            CodeExecutor().execute(" ")

    def test_timeout_limit_is_validated(self):
        with self.assertRaises(ValueError):
            CodeExecutor().execute(
                "print('hello')",
                timeout_seconds=100,
            )


if __name__ == "__main__":
    unittest.main()
