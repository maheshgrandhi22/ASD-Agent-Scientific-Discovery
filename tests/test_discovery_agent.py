import io
import unittest
import requests

from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from src.agent.discovery_agent import ScientificDiscoveryAgent
from src.agent.publication_verifier import PublicationVerifier
from src.agent.crossref_peer_review_provider import CrossrefPeerReviewProvider


class TestScientificDiscoveryAgent(unittest.TestCase):

    def setUp(self):
        self.question = "How can AI improve drug discovery?"

        self.plan = {
            "research_question": self.question,
            "objective": "Investigate AI in drug discovery.",
            "subquestions": ["What is already known?"],
            "methods": ["Review scientific literature."],
            "next_steps": ["Collect relevant papers."],
        }

        self.paper = {
            "title": "AI for Drug Discovery",
            "authors": ["Researcher One"],
            "published": "2026-01-01T00:00:00Z",
            "url": "https://arxiv.org/abs/1234.5678",
            "abstract": "AI methods for scientific research.",
            "doi": "10.1234/ai.drug.discovery",
        }
        self.verified_paper = {
            **self.paper,
            "peer_review_status": "verified",
            "peer_reviewed_verified": True,
            "verification_source": "Unit-test publication registry",
            "verified_identifier": "10.1234/ai.drug.discovery",
            "verified_publication_title": "AI for Drug Discovery",
            "verified_publication_venue": "Journal of Test Science",
            "verification_evidence_url": "https://registry.example.org/test-record",
        }

    def create_agent(self, **kwargs):
        class TestPublicationProvider:
            def verify_publication(self, paper):
                return {
                    "peer_reviewed": True,
                    "matched_identifier": paper.get("doi", ""),
                    "publication_title": paper.get("title", ""),
                    "publication_venue": "Journal of Test Science",
                    "evidence_url": "https://registry.example.org/test-record",
                }

        verifier = PublicationVerifier(
            provider=TestPublicationProvider(),
            provider_name="Unit-test publication registry",
        )
        return ScientificDiscoveryAgent(
            publication_verifier=verifier,
            **kwargs,
        )

    def test_crossref_provider_is_configured_by_default(self):
        agent = ScientificDiscoveryAgent()
        self.assertIsInstance(
            agent.publication_verifier.provider,
            CrossrefPeerReviewProvider,
        )
        self.assertEqual(
            agent.publication_verifier.provider_name,
            CrossrefPeerReviewProvider.PROVIDER_NAME,
        )

    @patch("src.agent.discovery_agent.ResultExporter.export_json")
    @patch("src.agent.discovery_agent.MultiSourceLiteratureRetriever.search")
    @patch("src.agent.discovery_agent.ResearchPlanner.create_plan")
    def test_complete_workflow(
        self, mock_plan, mock_search, mock_export
    ):
        mock_plan.return_value = self.plan
        mock_search.return_value = [self.paper]
        mock_export.return_value = "research_results.json"

        mock_store = Mock()
        mock_store.add_papers.return_value = 1
        mock_store.search.return_value = [self.verified_paper]

        agent = self.create_agent(vector_store=mock_store)
        dataset = [
            {
                "observed": 100,
                "baseline_prediction": 90,
                "candidate_prediction": 95,
                "data_split": "train",
            },
            {
                "observed": 10,
                "baseline_prediction": 8,
                "candidate_prediction": 9,
                "data_split": "holdout",
            },
            {
                "observed": 20,
                "baseline_prediction": 17,
                "candidate_prediction": 19,
                "data_split": "holdout",
            },
        ]
        output = io.StringIO()

        with redirect_stdout(output):
            result = agent.run(self.question, dataset=dataset)

        self.assertTrue(result)
        mock_plan.assert_called_once_with(self.question)
        mock_search.assert_called_once_with(
            self.question, max_results=3
        )

        mock_export.assert_called_once()
        exported_data = mock_export.call_args.args[0]

        self.assertEqual(
            exported_data["research_question"], self.question
        )
        self.assertEqual(exported_data["research_plan"], self.plan)
        self.assertEqual(len(exported_data["analyzed_papers"]), 1)
        self.assertEqual(
            exported_data["analyzed_papers"][0]["title"],
            "AI for Drug Discovery",
        )
        self.assertIn("research_summary", exported_data)
        holdout = exported_data["holdout_evaluation"]
        self.assertEqual(holdout["status"], "completed_small_sample")
        self.assertEqual(holdout["train_sample_size"], 1)
        self.assertEqual(holdout["holdout_sample_size"], 2)
        self.assertAlmostEqual(
            holdout["holdout_metrics"]["mae"]["baseline"], 2.5
        )
        self.assertAlmostEqual(
            holdout["holdout_metrics"]["mae"]["candidate"], 1.0
        )
        self.assertEqual(
            exported_data["evidence_retrieval"]["method"],
            "vector_store",
        )
        self.assertFalse(
            exported_data["evidence_retrieval"]["fallback_used"]
        )

        text = output.getvalue()
        self.assertIn("Scientific Discovery Research Plan", text)
        self.assertIn("Scientific Literature Search", text)
        self.assertIn("AI for Drug Discovery", text)
        self.assertIn("Scientific Research Summary", text)
        self.assertIn("Papers analyzed: 1", text)
        self.assertIn("Common research themes:", text)
        self.assertIn(
            "Research results exported to: research_results.json",
            text,
        )
        self.assertIn("Research workflow completed.", text)

    @patch("src.agent.discovery_agent.ResultExporter.export_json")
    @patch("src.agent.discovery_agent.MultiSourceLiteratureRetriever.search")
    @patch("src.agent.discovery_agent.ResearchPlanner.create_plan")
    def test_unconfigured_verifier_fails_closed_and_skips_execution(
        self, mock_plan, mock_search, mock_export
    ):
        mock_plan.return_value = self.plan
        mock_search.return_value = [{
            **self.paper,
            "peer_reviewed_verified": True,
            "peer_review_status": "verified",
            "verification_source": "Untrusted upstream flag",
        }]
        mock_export.return_value = "research_results.json"
        sandbox = Mock()

        agent = ScientificDiscoveryAgent(
            sandbox=sandbox,
            publication_verifier=PublicationVerifier(),
        )
        output = io.StringIO()
        with redirect_stdout(output):
            result = agent.run(self.question)

        self.assertTrue(result)
        exported = mock_export.call_args.args[0]
        self.assertEqual(
            exported["publication_verification"]["status"],
            "insufficient_verified_evidence",
        )
        self.assertEqual(exported["publication_verification"]["verified_count"], 0)
        self.assertEqual(exported["hypothesis_generation"]["hypotheses"], [])
        self.assertEqual(
            exported["code_execution"]["status"],
            "skipped_insufficient_verified_evidence",
        )
        sandbox.execute.assert_not_called()
        self.assertIn(
            "No papers met the configured Crossref peer-review evidence policy",
            output.getvalue(),
        )

    @patch("src.agent.discovery_agent.ResultExporter.export_json")
    @patch("src.agent.discovery_agent.MultiSourceLiteratureRetriever.search")
    @patch("src.agent.discovery_agent.ResearchPlanner.create_plan")
    def test_only_verified_papers_enter_vector_store_and_report(
        self, mock_plan, mock_search, mock_export
    ):
        mock_plan.return_value = self.plan
        unverified = {
            "title": "Unverified article",
            "abstract": "This record has no publication identifier.",
            "source": "arXiv",
            "url": "https://arxiv.org/abs/9999.00001",
        }
        mock_search.return_value = [self.paper, unverified]
        mock_export.return_value = "research_results.json"

        mock_store = Mock()
        mock_store.add_papers.return_value = 1
        mock_store.search.return_value = [self.verified_paper]

        agent = self.create_agent(vector_store=mock_store)
        with redirect_stdout(io.StringIO()):
            agent.run(self.question)

        stored = mock_store.add_papers.call_args.args[0]
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["title"], self.paper["title"])

        exported = mock_export.call_args.args[0]
        verification = exported["publication_verification"]
        self.assertEqual(verification["retrieved_count"], 2)
        self.assertEqual(verification["verified_count"], 1)
        self.assertEqual(verification["rejected_count"], 1)
        self.assertEqual(
            verification["rejected_papers"][0]["reason"],
            "missing_doi_or_pmid",
        )

    @patch("src.agent.discovery_agent.LiteratureRetriever.search")
    def test_invalid_question_stops_workflow(self, mock_search):
        agent = self.create_agent()
        output = io.StringIO()

        with redirect_stdout(output):
            result = agent.run("short")

        self.assertFalse(result)
        mock_search.assert_not_called()
        self.assertIn("Invalid research question", output.getvalue())

    @patch("src.agent.discovery_agent.ResultExporter.export_json")
    @patch("src.agent.discovery_agent.MultiSourceLiteratureRetriever.search")
    @patch("src.agent.discovery_agent.ResearchPlanner.create_plan")
    def test_export_failure_does_not_crash_agent(
        self, mock_plan, mock_search, mock_export
    ):
        mock_plan.return_value = self.plan
        mock_search.return_value = [self.paper]
        mock_export.side_effect = OSError("Disk is unavailable")

        agent = self.create_agent()
        output = io.StringIO()

        with redirect_stdout(output):
            result = agent.run(self.question)

        self.assertTrue(result)
        mock_export.assert_called_once()
        self.assertIn("Research result export failed", output.getvalue())
        self.assertIn("Research workflow completed.", output.getvalue())

    @patch("src.agent.discovery_agent.ResultExporter.export_json")
    @patch("src.agent.discovery_agent.MultiSourceLiteratureRetriever.search")
    @patch("src.agent.discovery_agent.ResearchPlanner.create_plan")
    def test_no_papers_exports_insufficient_evidence_report(
        self, mock_plan, mock_search, mock_export
    ):
        mock_plan.return_value = self.plan
        mock_search.return_value = []
        mock_export.return_value = "research_results.json"

        agent = self.create_agent()
        dataset = [
            {
                "observed": 100,
                "baseline_prediction": 90,
                "candidate_prediction": 95,
                "data_split": "train",
            },
            {
                "observed": 10,
                "baseline_prediction": 8,
                "candidate_prediction": 9,
                "data_split": "holdout",
            },
            {
                "observed": 20,
                "baseline_prediction": 17,
                "candidate_prediction": 19,
                "data_split": "holdout",
            },
        ]
        output = io.StringIO()

        with redirect_stdout(output):
            result = agent.run(self.question, dataset=dataset)

        self.assertTrue(result)
        self.assertIn("No papers were found", output.getvalue())
        mock_export.assert_called_once()

        exported = mock_export.call_args.args[0]
        self.assertEqual(
            exported["publication_verification"]["status"],
            "no_papers_retrieved",
        )
        self.assertEqual(
            exported["publication_verification"]["retrieved_count"], 0
        )
        self.assertEqual(exported["analyzed_papers"], [])
        self.assertEqual(
            exported["holdout_evaluation"]["status"],
            "completed_small_sample",
        )
        self.assertEqual(
            exported["holdout_evaluation"]["holdout_sample_size"], 2
        )
        self.assertEqual(exported["hypothesis_generation"]["hypotheses"], [])
        self.assertEqual(
            exported["hypothesis_generation"]["status"],
            "skipped_insufficient_verified_evidence",
        )
        self.assertEqual(
            exported["code_execution"]["status"],
            "skipped_insufficient_verified_evidence",
        )
        critic = exported["statistical_critic"]
        self.assertEqual(critic["sample_size"], 3)
        self.assertAlmostEqual(critic["baseline_mae"], 5.0)
        self.assertAlmostEqual(critic["candidate_mae"], 7 / 3)
        self.assertEqual(
            critic["evaluation_scope"],
            "independent_paired_dataset",
        )
        self.assertEqual(
            critic["scientific_validation"],
            "not_established",
        )
        self.assertIn("does not validate literature evidence", critic["notice"])
        self.assertNotEqual(
            critic["status"],
            "skipped_insufficient_verified_evidence",
        )
        self.assertIn(
            "Insufficient-evidence report exported",
            output.getvalue(),
        )
        self.assertIn("Research workflow completed.", output.getvalue())

    @patch("src.agent.discovery_agent.MultiSourceLiteratureRetriever.search")
    @patch("src.agent.discovery_agent.ResearchPlanner.create_plan")
    def test_literature_failure_does_not_crash_agent(
        self, mock_plan, mock_search
    ):
        mock_plan.return_value = self.plan
        mock_search.side_effect = requests.RequestException(
            "Service temporarily unavailable"
        )

        agent = self.create_agent()
        output = io.StringIO()

        with redirect_stdout(output):
            result = agent.run(self.question)

        self.assertTrue(result)
        self.assertIn("Literature search unavailable", output.getvalue())
        self.assertIn("Research workflow completed.", output.getvalue())


    @patch("src.agent.discovery_agent.ResultExporter.export_json")
    @patch("src.agent.discovery_agent.MultiSourceLiteratureRetriever.search")
    @patch("src.agent.discovery_agent.ResearchPlanner.create_plan")
    def test_workflow_uses_injected_sandbox(
        self, mock_plan, mock_search, mock_export
    ):
        mock_plan.return_value = self.plan
        mock_search.return_value = [self.paper]
        mock_export.return_value = "research_results.json"

        sandbox = Mock()
        sandbox.execute.return_value = {
            "status": "success",
            "output": "experiment completed",
            "error": "",
            "return_code": 0,
        }

        agent = self.create_agent(sandbox=sandbox)
        output = io.StringIO()

        with redirect_stdout(output):
            result = agent.run(self.question)

        self.assertTrue(result)
        exported_data = mock_export.call_args.args[0]
        experiment = exported_data["experiment_specification"]
        expected_code = experiment["code"]
        self.assertIn(self.question, expected_code)
        self.assertEqual(
            experiment["experiment_type"], "baseline_context_check"
        )
        sandbox.execute.assert_called_once_with(
            expected_code,
            timeout_seconds=10,
        )
        self.assertEqual(
            exported_data["code_execution"]["status"], "success"
        )
        self.assertIn("Execution status: success", output.getvalue())

    @patch("src.agent.discovery_agent.ResultExporter.export_json")
    @patch("src.agent.discovery_agent.MultiSourceLiteratureRetriever.search")
    @patch("src.agent.discovery_agent.ResearchPlanner.create_plan")
    def test_workflow_revises_failed_experiment(
        self, mock_plan, mock_search, mock_export
    ):
        mock_plan.return_value = self.plan
        mock_search.return_value = [self.paper]
        mock_export.return_value = "research_results.json"

        sandbox = Mock()
        sandbox.execute.side_effect = [
            {
                "status": "error",
                "output": "",
                "error": "SyntaxError",
            },
            {
                "status": "success",
                "output": "experiment completed",
                "error": "",
                "return_code": 0,
            },
        ]

        reviser = Mock(return_value="print('revised experiment')")
        agent = self.create_agent(
            sandbox=sandbox,
            code_reviser=reviser,
        )
        output = io.StringIO()

        with redirect_stdout(output):
            result = agent.run(self.question)

        self.assertTrue(result)
        self.assertEqual(sandbox.execute.call_count, 2)
        exported_data = mock_export.call_args.args[0]
        initial_code = exported_data["experiment_specification"]["code"]
        reviser.assert_called_once_with(
            initial_code,
            "SyntaxError",
            1,
        )
        execution = exported_data["code_execution"]
        self.assertEqual(execution["status"], "success")
        self.assertEqual(len(execution["attempts"]), 2)
        self.assertEqual(
            execution["final_code"], "print('revised experiment')"
        )
        self.assertIn("Execution attempts: 2", output.getvalue())


    @patch("src.agent.discovery_agent.ResultExporter.export_json")
    @patch("src.agent.discovery_agent.MultiSourceLiteratureRetriever.search")
    @patch("src.agent.discovery_agent.ResearchPlanner.create_plan")
    def test_vector_store_failure_is_recorded_and_falls_back(
        self, mock_plan, mock_search, mock_export
    ):
        mock_plan.return_value = self.plan
        mock_search.return_value = [self.paper]
        mock_export.return_value = "research_results.json"

        mock_store = Mock()
        mock_store.add_papers.return_value = 1
        mock_store.search.side_effect = RuntimeError(
            "vector search unavailable"
        )

        agent = self.create_agent(vector_store=mock_store)
        output = io.StringIO()

        with redirect_stdout(output):
            result = agent.run(self.question)

        self.assertTrue(result)
        exported_data = mock_export.call_args.args[0]
        retrieval = exported_data["evidence_retrieval"]

        self.assertEqual(
            retrieval["status"],
            "fallback_vector_store_unavailable",
        )
        self.assertEqual(retrieval["method"], "direct_literature")
        self.assertTrue(retrieval["fallback_used"])
        self.assertIn(
            "vector search unavailable",
            retrieval["fallback_reason"],
        )
        self.assertEqual(retrieval["papers_retrieved"], 0)
        self.assertEqual(exported_data["retrieved_evidence"], [])
        self.assertEqual(
            exported_data["literature_review"]["paper_count"], 1
        )
        self.assertIn("Vector store unavailable", output.getvalue())


    @patch("src.agent.discovery_agent.ResultExporter.export_json")
    @patch("src.agent.discovery_agent.MultiSourceLiteratureRetriever.search")
    @patch("src.agent.discovery_agent.ResearchPlanner.create_plan")
    def test_pdf_ingestion_is_included_in_export(
        self, mock_plan, mock_search, mock_export
    ):
        mock_plan.return_value = self.plan
        mock_search.return_value = [self.paper]
        mock_export.return_value = "research_results.json"

        mock_store = Mock()
        mock_store.add_papers.return_value = 1
        mock_store.search.return_value = [self.verified_paper]

        ingestor = Mock()
        ingestor.extract.side_effect = [
            {
                "source_pdf": "paper_one.pdf",
                "total_pages": 3,
                "total_figures": 1,
                "pages": [],
                "figures": [],
            },
            {
                "source_pdf": "paper_two.pdf",
                "total_pages": 5,
                "total_figures": 2,
                "pages": [],
                "figures": [],
            },
        ]

        agent = self.create_agent(
            vector_store=mock_store,
            pdf_ingestor=ingestor,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            result = agent.run(
                self.question,
                pdf_paths=["paper_one.pdf", "paper_two.pdf"],
            )

        self.assertTrue(result)
        self.assertEqual(ingestor.extract.call_count, 2)

        exported_data = mock_export.call_args.args[0]
        pdf_data = exported_data["pdf_ingestion"]

        self.assertEqual(pdf_data["status"], "completed")
        self.assertEqual(len(pdf_data["documents"]), 2)
        self.assertEqual(
            pdf_data["documents"][0]["source_pdf"],
            "paper_one.pdf",
        )
        self.assertEqual(
            pdf_data["documents"][1]["total_figures"],
            2,
        )
        self.assertFalse(pdf_data["used_as_hypothesis_evidence"])

    @patch("src.agent.discovery_agent.ResultExporter.export_json")
    @patch("src.agent.discovery_agent.MultiSourceLiteratureRetriever.search")
    @patch("src.agent.discovery_agent.ResearchPlanner.create_plan")
    def test_pdf_ingestion_failure_is_recorded_without_crashing(
        self, mock_plan, mock_search, mock_export
    ):
        mock_plan.return_value = self.plan
        mock_search.return_value = [self.paper]
        mock_export.return_value = "research_results.json"

        mock_store = Mock()
        mock_store.add_papers.return_value = 1
        mock_store.search.return_value = [self.verified_paper]

        ingestor = Mock()
        ingestor.extract.side_effect = FileNotFoundError(
            "PDF file not found"
        )

        agent = self.create_agent(
            vector_store=mock_store,
            pdf_ingestor=ingestor,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            result = agent.run(
                self.question,
                pdf_paths=["missing_paper.pdf"],
            )

        self.assertTrue(result)
        mock_export.assert_called_once()

        exported_data = mock_export.call_args.args[0]
        pdf_data = exported_data["pdf_ingestion"]

        self.assertEqual(pdf_data["status"], "failed")
        self.assertEqual(len(pdf_data["documents"]), 1)
        self.assertEqual(
            pdf_data["documents"][0]["status"],
            "failed",
        )
        self.assertIn(
            "FileNotFoundError",
            pdf_data["documents"][0]["error"],
        )
        self.assertIn(
            "PDF ingestion failed",
            output.getvalue(),
        )

class TestGeminiDependencyInjection(unittest.TestCase):
    def test_gemini_client_is_injected_into_hypothesis_generator(self):
        class FakeGemini:
            def generate_text(self, prompt):
                return "Refined candidate hypothesis."

        client = FakeGemini()
        agent = ScientificDiscoveryAgent(gemini_client=client)

        self.assertIs(
            agent.hypothesis_generator.gemini_client,
            client,
        )


if __name__ == "__main__":
    unittest.main()
