import io
import unittest
import requests

from contextlib import redirect_stdout
from unittest.mock import patch

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

    @patch("src.agent.discovery_agent.LiteratureRetriever.search")
    @patch("src.agent.discovery_agent.ResearchPlanner.create_plan")
    def test_complete_workflow(self, mock_plan, mock_search):
        mock_plan.return_value = self.plan
        mock_search.return_value = [self.paper]

        agent = ScientificDiscoveryAgent()
        output = io.StringIO()

        with redirect_stdout(output):
            result = agent.run(self.question)

        self.assertTrue(result)
        mock_plan.assert_called_once_with(self.question)
        mock_search.assert_called_once_with(
            self.question, max_results=3
        )
        self.assertIn("Scientific Discovery Research Plan", output.getvalue())
        self.assertIn("Scientific Literature Search", output.getvalue())
        self.assertIn("AI for Drug Discovery", output.getvalue())
        self.assertIn("Scientific Research Summary", output.getvalue())
        self.assertIn("Papers analyzed: 1", output.getvalue())
        self.assertIn("Common research themes:", output.getvalue())
        self.assertIn("Research workflow completed.", output.getvalue())

    @patch("src.agent.discovery_agent.LiteratureRetriever.search")
    def test_invalid_question_stops_workflow(self, mock_search):
        agent = ScientificDiscoveryAgent()
        output = io.StringIO()

        with redirect_stdout(output):
            result = agent.run("short")

        self.assertFalse(result)
        mock_search.assert_not_called()
        self.assertIn("Invalid research question", output.getvalue())

    @patch("src.agent.discovery_agent.LiteratureRetriever.search")
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


if __name__ == "__main__":
    unittest.main()

