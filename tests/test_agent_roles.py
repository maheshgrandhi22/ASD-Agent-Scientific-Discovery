
import unittest
from unittest.mock import Mock

from src.agent.agent_roles import (
    CodeExecutor,
    HypothesisGenerator,
    LiteratureReviewer,
)


class TestLiteratureReviewer(unittest.TestCase):
    def test_review_returns_ranked_evidence(self):
        retriever = Mock()
        analyzer = Mock()

        retriever.search.return_value = [
            {
                "id": "PMID:123",
                "title": "AI in discovery",
                "abstract": "AI may improve screening.",
                "source": "PubMed",
                "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
            }
        ]

        analyzer.analyze.return_value = [
            {
                "id": "PMID:123",
                "title": "AI in discovery",
                "abstract": "AI may improve screening.",
                "source": "PubMed",
                "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
                "relevance_score": 80,
                "key_findings": ["Screening may improve."],
            }
        ]

        reviewer = LiteratureReviewer(retriever, analyzer)
        result = reviewer.review("How can AI improve discovery?")

        self.assertEqual(result["role"], "Literature Reviewer")
        self.assertEqual(result["paper_count"], 1)
        self.assertEqual(result["evidence"][0]["relevance_score"], 80)


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
            HypothesisGenerator().generate(" ", {"evidence": []})


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
            CodeExecutor().execute("print('hello')", timeout_seconds=100)


if __name__ == "__main__":
    unittest.main()
