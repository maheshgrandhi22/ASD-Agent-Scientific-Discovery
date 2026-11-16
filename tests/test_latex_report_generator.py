import unittest

from src.agent.latex_report_generator import LatexReportGenerator


class TestLatexReportGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = LatexReportGenerator()

    def test_generates_basic_report(self):
        report = self.generator.generate({
            "research_question": "Can AI improve drug discovery?",
            "research_summary": {
                "summary": "A sample research summary."
            },
        })

        self.assertIn(r"\documentclass", report)
        self.assertIn(r"\section{Research Question}", report)
        self.assertIn("Can AI improve drug discovery?", report)
        self.assertIn(r"\end{document}", report)

    def test_escapes_latex_special_characters(self):
        report = self.generator.generate({
            "research_question": "AI & biology: 50% improvement_test",
        })

        self.assertIn(r"AI \& biology: 50\% improvement\_test", report)

    def test_includes_literature_metadata_and_references(self):
        report = self.generator.generate({
            "analyzed_papers": [{
                "title": "A scientific paper",
                "doi": "10.1000/example",
                "published": "2025",
            }],
        })

        self.assertIn("A scientific paper", report)
        self.assertIn("10.1000/example", report)
        self.assertIn(r"\section{References}", report)

    def test_hypotheses_are_explicitly_labelled_as_candidates(self):
        report = self.generator.generate({
            "candidate_hypotheses": [
                {"hypothesis": "AI may improve screening."}
            ],
        })

        self.assertIn("Candidate Hypotheses", report)
        self.assertIn("candidate hypotheses", report)
        self.assertIn("AI may improve screening.", report)
        self.assertIn("does not establish", report)

    def test_missing_sections_are_handled(self):
        report = self.generator.generate({})

        self.assertIn("No literature records were provided", report)
        self.assertIn("No candidate hypotheses were provided", report)
        self.assertIn("No holdout evaluation data was provided", report)

    def test_includes_optional_evaluation_data(self):
        report = self.generator.generate({
            "statistical_critic": {"p_value": 0.04},
            "holdout_evaluation": {"status": "completed"},
        })

        self.assertIn("Statistical Critic", report)
        self.assertIn("0.04", report)
        self.assertIn("Holdout Evaluation", report)
        self.assertIn("completed", report)

    def test_includes_measured_execution_metrics(self):
        report = self.generator.generate({
            "execution_metrics": {
                "workflow_duration_seconds": 12.345,
                "duration_unit": "seconds",
                "token_usage_status": "not_exposed_by_provider",
                "api_cost_status": "not_available",
            }
        })

        self.assertIn("Execution Metrics", report)
        self.assertIn("12.345", report)
        self.assertIn("not exposed by provider", report)
        self.assertIn("not available", report)

    def test_rejects_non_dictionary_input(self):
        with self.assertRaises(TypeError):
            self.generator.generate([])


if __name__ == "__main__":
    unittest.main()
