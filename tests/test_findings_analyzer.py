import unittest

from src.agent.findings_analyzer import FindingsAnalyzer


class TestFindingsAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = FindingsAnalyzer()
        self.question = "How can AI improve drug discovery?"

        self.papers = [
            {
                "title": "Climate Change Research",
                "abstract": "This paper studies global temperatures.",
                "url": "https://example.com/climate",
            },
            {
                "title": "AI for Drug Discovery",
                "abstract": (
                    "AI models help identify drug candidates. "
                    "Machine learning can support medicine development."
                ),
                "url": "https://example.com/drug",
            },
        ]

    def test_papers_are_ranked_by_relevance(self):
        results = self.analyzer.analyze(
            self.question, self.papers
        )

        self.assertEqual(
            results[0]["title"], "AI for Drug Discovery"
        )
        self.assertGreaterEqual(
            results[0]["relevance_score"],
            results[1]["relevance_score"],
        )

    def test_relevance_score_is_between_zero_and_100(self):
        results = self.analyzer.analyze(
            self.question, self.papers
        )

        for paper in results:
            self.assertGreaterEqual(paper["relevance_score"], 0)
            self.assertLessEqual(paper["relevance_score"], 100)

    def test_key_findings_are_extracted(self):
        results = self.analyzer.analyze(
            self.question, self.papers
        )

        findings = results[0]["key_findings"]

        self.assertTrue(findings)
        self.assertIn("AI models help identify drug candidates.", findings)

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            self.analyzer.analyze("  ", self.papers)

    def test_papers_must_be_a_list(self):
        with self.assertRaises(ValueError):
            self.analyzer.analyze(self.question, "not a list")

    def test_empty_paper_list_is_supported(self):
        self.assertEqual(
            self.analyzer.analyze(self.question, []), []
        )


if __name__ == "__main__":
    unittest.main()
