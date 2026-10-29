import io
import unittest
import requests

from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from src.agent.discovery_agent import ScientificDiscoveryAgent


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
        }

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
        mock_store.search.return_value = [self.paper]

        agent = ScientificDiscoveryAgent(vector_store=mock_store)
        output = io.StringIO()

        with redirect_stdout(output):
            result = agent.run(self.question)

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

    @patch("src.agent.discovery_agent.LiteratureRetriever.search")
    def test_invalid_question_stops_workflow(self, mock_search):
        agent = ScientificDiscoveryAgent()
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

        agent = ScientificDiscoveryAgent()
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
    def test_no_papers_skips_export(
        self, mock_plan, mock_search, mock_export
    ):
        mock_plan.return_value = self.plan
        mock_search.return_value = []

        agent = ScientificDiscoveryAgent()
        output = io.StringIO()

        with redirect_stdout(output):
            result = agent.run(self.question)

        self.assertTrue(result)
        self.assertIn("No papers were found", output.getvalue())
        mock_export.assert_not_called()
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

        agent = ScientificDiscoveryAgent()
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

        agent = ScientificDiscoveryAgent(sandbox=sandbox)
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
        agent = ScientificDiscoveryAgent(
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

        agent = ScientificDiscoveryAgent(vector_store=mock_store)
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

if __name__ == "__main__":
    unittest.main()
