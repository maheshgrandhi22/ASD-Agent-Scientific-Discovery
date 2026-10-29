import io
import json
import unittest
from contextlib import redirect_stdout

from src.agent.experiment_builder import ScientificExperimentBuilder


class TestScientificExperimentBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = ScientificExperimentBuilder()
        self.question = "How can AI improve drug discovery?"
        self.hypotheses = [{
            "hypothesis": "AI models may improve candidate prioritization"
        }]
        self.evidence = [{
            "title": "AI drug discovery study",
            "abstract": (
                "This abstract contains enough substantive text "
                "to pass the evidence quality threshold."
            ),
            "source": "PubMed",
        }]

    def test_builds_reproducible_audit_from_context(self):
        first = self.builder.build(
            self.question, self.hypotheses, self.evidence
        )
        second = self.builder.build(
            self.question, self.hypotheses, self.evidence
        )
        self.assertEqual(first["code"], second["code"])
        self.assertEqual(first["evidence_records_considered"], 1)
        self.assertEqual(first["evidence_records_rejected"], 0)
        self.assertEqual(first["evidence_coverage_percent"], 100.0)
        self.assertEqual(
            first["comparison_status"], "dataset_not_provided"
        )
        self.assertIsNone(first["mae"])
        self.assertEqual(
            first["validation_status"], "not experimentally validated"
        )

    def test_comparison_calculates_mae_and_rmse(self):
        dataset = [
            {"observed": 10, "baseline_prediction": 8,
             "candidate_prediction": 9},
            {"observed": 20, "baseline_prediction": 17,
             "candidate_prediction": 19},
        ]
        result = self.builder.build(
            self.question, self.hypotheses, self.evidence, dataset
        )
        self.assertEqual(result["valid_dataset_rows"], 2)
        self.assertEqual(result["mae"]["baseline"], 2.5)
        self.assertEqual(result["mae"]["candidate"], 1.0)
        self.assertAlmostEqual(result["rmse"]["baseline"], (6.5) ** 0.5)
        self.assertEqual(result["candidate_mae_improvement_percent"], 60.0)

    def test_generated_code_reports_comparison_results(self):
        dataset = [
            {"observed": 10, "baseline_prediction": 8,
             "candidate_prediction": 9},
        ]
        result = self.builder.build(
            self.question, self.hypotheses, self.evidence, dataset
        )
        output = io.StringIO()
        with redirect_stdout(output):
            exec(result["code"], {})
        report = json.loads(output.getvalue())
        self.assertEqual(report["mae"]["baseline"], 2.0)
        self.assertEqual(report["mae"]["candidate"], 1.0)
        self.assertEqual(report["valid_dataset_rows"], 1)
        self.assertEqual(report["scientific_validation"], "not_performed")

    def test_invalid_dataset_rows_are_ignored(self):
        dataset = [
            {"observed": 1, "baseline_prediction": 1,
             "candidate_prediction": 1},
            {"observed": float("nan"), "baseline_prediction": 1,
             "candidate_prediction": 1},
            {"observed": True, "baseline_prediction": 1,
             "candidate_prediction": 1},
            None,
        ]
        result = self.builder.build(
            self.question, self.hypotheses, self.evidence, dataset
        )
        self.assertEqual(result["valid_dataset_rows"], 1)
        self.assertEqual(result["comparison_status"], "completed_on_supplied_data")

    def test_invalid_dataset_type_is_rejected(self):
        with self.assertRaises(ValueError):
            self.builder.build(self.question, dataset="not a list")

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            self.builder.build(" ")

    def test_invalid_hypothesis_text_is_rejected(self):
        with self.assertRaises(ValueError):
            self.builder.build(
                self.question, hypotheses=[{"hypothesis": "  "}],
                evidence=self.evidence,
            )

    def test_malformed_hypothesis_record_is_rejected(self):
        with self.assertRaises(ValueError):
            self.builder.build(
                self.question,
                hypotheses=[None, {"hypothesis": 123}],
                evidence=self.evidence,
            )

    def test_missing_hypothesis_is_allowed_but_not_claimed(self):
        result = self.builder.build(
            self.question, hypotheses=[], evidence=self.evidence
        )
        self.assertEqual(result["candidate_hypothesis"], "")
        self.assertEqual(
            result["validation_status"], "not experimentally validated"
        )

    def test_title_only_evidence_is_rejected(self):
        result = self.builder.build(
            self.question, self.hypotheses,
            evidence=[{"title": "A paper title only"}],
        )
        self.assertEqual(result["evidence_records_considered"], 0)
        self.assertEqual(result["evidence_records_rejected"], 1)
        self.assertEqual(result["evidence_coverage_percent"], 0.0)

    def test_invalid_evidence_records_are_ignored(self):
        result = self.builder.build(
            self.question, self.hypotheses,
            evidence=[
                None, {}, {"title": "Title without an abstract"},
                {"title": "Useful study", "abstract":
                 "This is enough substantive abstract content "
                 "to meet the evidence threshold."},
            ],
        )
        self.assertEqual(result["evidence_records_received"], 4)
        self.assertEqual(result["evidence_records_considered"], 1)
        self.assertEqual(result["evidence_records_rejected"], 3)

    def test_evidence_list_type_is_validated(self):
        with self.assertRaises(ValueError):
            self.builder.build(self.question, evidence="not a list")

    def test_hypothesis_list_type_is_validated(self):
        with self.assertRaises(ValueError):
            self.builder.build(self.question, hypotheses="not a list")


    def test_hypothesis_evidence_reference_is_matched(self):
        hypotheses = [{
            "hypothesis": "AI may improve candidate prioritization",
            "supporting_paper_id": "PMID-123",
        }]
        evidence = [{
            "title": "AI drug discovery study",
            "abstract": (
                "This abstract contains enough substantive text "
                "to pass the evidence quality threshold."
            ),
            "source": "PubMed",
            "paper_id": "PMID-123",
            "url": "https://example.org/paper/123",
        }]

        result = self.builder.build(
            self.question, hypotheses, evidence
        )

        self.assertEqual(
            result["hypothesis_evidence_traceability"][0]["status"],
            "matched",
        )
        self.assertEqual(
            result["hypothesis_evidence_traceability_counts"]["matched"],
            1,
        )
        self.assertEqual(
            result["evidence_records"][0]["paper_id"], "PMID-123"
        )
        self.assertEqual(
            result["evidence_records"][0]["url"],
            "https://example.org/paper/123",
        )

        output = io.StringIO()
        with redirect_stdout(output):
            exec(result["code"], {})
        report = json.loads(output.getvalue())

        self.assertEqual(
            report["hypothesis_evidence_traceability"][0]["status"],
            "matched",
        )
        self.assertEqual(
            report["evidence_records"][0]["url"],
            "https://example.org/paper/123",
        )

    def test_hypothesis_evidence_reference_is_unmatched(self):
        hypotheses = [{
            "hypothesis": "AI may improve candidate prioritization",
            "supporting_paper_id": "PMID-NOT-FOUND",
        }]
        evidence = [{
            "title": "AI drug discovery study",
            "abstract": (
                "This abstract contains enough substantive text "
                "to pass the evidence quality threshold."
            ),
            "paper_id": "PMID-123",
        }]

        result = self.builder.build(
            self.question, hypotheses, evidence
        )

        self.assertEqual(
            result["hypothesis_evidence_traceability"][0]["status"],
            "unmatched_reference",
        )
        self.assertEqual(
            result["hypothesis_evidence_traceability_counts"][
                "unmatched_reference"
            ],
            1,
        )

    def test_hypothesis_evidence_reference_is_missing(self):
        hypotheses = [{
            "hypothesis": "AI may improve candidate prioritization",
        }]
        evidence = [{
            "title": "AI drug discovery study",
            "abstract": (
                "This abstract contains enough substantive text "
                "to pass the evidence quality threshold."
            ),
            "paper_id": "PMID-123",
        }]

        result = self.builder.build(
            self.question, hypotheses, evidence
        )

        self.assertIsNone(
            result["hypothesis_evidence_traceability"][0][
                "supporting_paper_id"
            ]
        )
        self.assertEqual(
            result["hypothesis_evidence_traceability"][0]["status"],
            "missing_reference",
        )
        self.assertEqual(
            result["hypothesis_evidence_traceability_counts"][
                "missing_reference"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
