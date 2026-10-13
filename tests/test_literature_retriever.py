import unittest
from unittest.mock import Mock, patch

import requests

from src.agent.literature_retriever import LiteratureRetriever


SAMPLE_XML = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/1234.5678</id>
    <published>2026-01-15T10:00:00Z</published>
    <title>AI for Drug Discovery</title>
    <summary>Using AI to discover new medicines.</summary>
    <author><name>Alex Researcher</name></author>
  </entry>
</feed>
"""


class TestLiteratureRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = LiteratureRetriever()

    def test_search_returns_paper_metadata(self):
        response = Mock()
        response.content = SAMPLE_XML.encode("utf-8")

        with patch(
            "src.agent.literature_retriever.requests.get",
            return_value=response,
        ) as mock_get:
            papers = self.retriever.search("drug discovery", 1)

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["title"], "AI for Drug Discovery")
        self.assertEqual(papers[0]["authors"], ["Alex Researcher"])
        self.assertIn("AI to discover", papers[0]["abstract"])
        self.assertEqual(
            papers[0]["url"],
            "https://arxiv.org/abs/1234.5678",
        )
        mock_get.assert_called_once()

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.search("   ")

    def test_invalid_result_count_is_rejected(self):
        for count in (0, 21, True, 1.5):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    self.retriever.search("science", count)

    def test_http_error_is_propagated(self):
        response = Mock()
        response.raise_for_status.side_effect = (
            requests.HTTPError("Server error")
        )

        with patch(
            "src.agent.literature_retriever.requests.get",
            return_value=response,
        ):
            with self.assertRaises(requests.HTTPError):
                self.retriever.search("science", 1)


if __name__ == "__main__":
    unittest.main()
