
import unittest
from unittest.mock import Mock, patch
import requests
import xml.etree.ElementTree as ET

from src.agent.literature_retriever import LiteratureRetriever


class TestLiteratureRetriever(unittest.TestCase):

    def setUp(self):
        self.retriever = LiteratureRetriever()

    def test_empty_query_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "non-empty string"
        ):
            self.retriever.search("")

    def test_whitespace_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.search("   ")

    def test_non_string_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.search(None)

    def test_invalid_max_results_is_rejected(self):
        for value in (0, -1, 1.5, True, "5"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.retriever.search("science", value)

    @patch("src.agent.literature_retriever.requests.get")
    def test_search_returns_parsed_paper(self, mock_get):
        xml = b"""
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/1234.5678</id>
                <title>AI for Science</title>
                <summary>Research summary.</summary>
                <author><name>Jane Smith</name></author>
                <published>2026-01-01T00:00:00Z</published>
            </entry>
        </feed>
        """

        response = Mock()
        response.content = xml
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        papers = self.retriever.search("science", 1)

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["title"], "AI for Science")
        self.assertEqual(papers[0]["authors"], ["Jane Smith"])
        self.assertEqual(papers[0]["summary"], "Research summary.")
        mock_get.assert_called_once()

    @patch("src.agent.literature_retriever.requests.get")
    def test_search_removes_duplicate_papers(self, mock_get):
        xml = b"""
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/1234</id>
                <title>Paper A</title>
                <summary>Summary A</summary>
            </entry>
            <entry>
                <id>http://arxiv.org/abs/1234</id>
                <title>Paper A duplicate</title>
                <summary>Duplicate summary</summary>
            </entry>
        </feed>
        """

        response = Mock()
        response.content = xml
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        papers = self.retriever.search("science", 2)

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["title"], "Paper A")

    @patch("src.agent.literature_retriever.requests.get")
    def test_malformed_xml_response_raises_clear_error(self, mock_get):
        response = Mock()
        response.content = b"<feed><entry>"
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        with self.assertRaisesRegex(
            ValueError, "Invalid XML response received from arXiv"
        ):
            self.retriever.search("science", 1)

    @patch("src.agent.literature_retriever.requests.get")
    def test_request_timeout_raises_clear_error(self, mock_get):
        mock_get.side_effect = requests.Timeout("Connection timed out")

        with self.assertRaisesRegex(
            requests.Timeout,
            "arXiv request timed out after 20 seconds",
        ):
            self.retriever.search("drug discovery", 1)

        mock_get.assert_called_once()
        self.assertEqual(
            mock_get.call_args.kwargs["timeout"], 20
        )

    @patch("src.agent.literature_retriever.requests.get")
    def test_connection_error_raises_clear_error(self, mock_get):
        mock_get.side_effect = requests.ConnectionError(
            "Network unavailable"
        )

        with self.assertRaisesRegex(
            requests.ConnectionError,
            "Unable to connect to the arXiv API",
        ):
            self.retriever.search("drug discovery", 1)

        mock_get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
