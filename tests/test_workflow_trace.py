import unittest

from src.agent.workflow_trace import build_workflow_trace


class TestWorkflowTrace(unittest.TestCase):
    def test_insufficient_verified_evidence_marks_later_stages_skipped(self):
        results = {
            "research_plan": {"objective": "Test"},
            "publication_verification": {
                "status": "insufficient_verified_evidence",
                "retrieved_count": 3,
                "verified_count": 0,
            },
            "literature_review": {"paper_count": 3, "evidence": []},
            "evidence_retrieval": {
                "status": "skipped_insufficient_verified_evidence"
            },
            "hypothesis_generation": {
                "status": "skipped_insufficient_verified_evidence",
                "hypotheses": [],
            },
            "experiment_specification": {
                "status": "skipped_insufficient_verified_evidence"
            },
            "code_execution": {
                "status": "skipped_insufficient_verified_evidence"
            },
            "statistical_critic": {
                "status": "skipped_insufficient_verified_evidence"
            },
            "holdout_evaluation": {"status": "not_provided"},
            "pdf_ingestion": {"status": "not_requested"},
        }

        trace = build_workflow_trace(results)
        stages = {item["stage"]: item for item in trace}

        self.assertEqual(stages["Research planning"]["status"], "completed")
        self.assertEqual(stages["Literature retrieval"]["status"], "completed")
        self.assertEqual(stages["Publication verification"]["status"], "blocked")
        self.assertEqual(stages["Evidence retrieval"]["status"], "skipped")
        self.assertEqual(stages["Candidate hypothesis generation"]["status"], "skipped")
        self.assertEqual(stages["Experiment specification"]["status"], "skipped")
        self.assertEqual(stages["Experiment execution"]["status"], "skipped")
        self.assertEqual(stages["Statistical evaluation"]["status"], "skipped")
        self.assertEqual(stages["Holdout evaluation"]["status"], "skipped")
        self.assertEqual(stages["PDF ingestion"]["status"], "skipped")

    def test_successful_execution_is_recorded_as_completed(self):
        results = {
            "research_plan": {},
            "publication_verification": {
                "status": "completed",
                "retrieved_count": 3,
                "verified_count": 2,
            },
            "evidence_retrieval": {"status": "success"},
            "hypothesis_generation": {
                "status": "assisted",
                "hypotheses": [{"text": "Candidate"}],
            },
            "experiment_specification": {"status": "completed", "code": "print(1)"},
            "code_execution": {"status": "success"},
            "statistical_critic": {"status": "completed"},
            "holdout_evaluation": {"status": "completed"},
            "pdf_ingestion": {"status": "completed"},
        }

        trace = build_workflow_trace(results)
        stages = {item["stage"]: item for item in trace}

        self.assertEqual(stages["Publication verification"]["status"], "completed")
        self.assertEqual(stages["Experiment execution"]["status"], "completed")
        self.assertEqual(stages["Statistical evaluation"]["status"], "completed")
        self.assertEqual(stages["Holdout evaluation"]["status"], "completed")

    def test_missing_optional_sections_are_not_claimed_as_completed(self):
        trace = build_workflow_trace({})
        stages = {item["stage"]: item for item in trace}

        self.assertEqual(stages["Research planning"]["status"], "unavailable")
        self.assertEqual(stages["Experiment execution"]["status"], "unavailable")
        self.assertEqual(stages["PDF ingestion"]["status"], "unavailable")

    def test_non_dictionary_input_is_rejected(self):
        with self.assertRaises(TypeError):
            build_workflow_trace([])


if __name__ == "__main__":
    unittest.main()
