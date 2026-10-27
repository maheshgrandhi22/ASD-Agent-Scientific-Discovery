
import unittest
from unittest.mock import Mock, patch

import requests

from src.agent.pubmed_retriever import PubMedRetriever


class TestPubMedRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = PubMedRetriever()

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.search(" ")

    def test_invalid_max_results_is_rejected(self):
        for value in (0, -1, True, 1.5, "3"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.retriever.search("scientific discovery", value)

    @patch("src.agent.pubmed_retriever.requests.get")
    def test_no_search_results_returns_empty_list(self, mock_get):
        response = Mock()
        response.json.return_value = {
            "esearchresult": {"idlist": []}
        }
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        result = self.retriever.search("rare topic")

        self.assertEqual(result, [])
        self.assertEqual(mock_get.call_count, 1)

    @patch("src.agent.pubmed_retriever.requests.get")
    def test_search_request_failure_is_propagated(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("offline")

        with self.assertRaises(requests.ConnectionError):
            self.retriever.search("scientific discovery")

    @patch("src.agent.pubmed_retriever.requests.get")
    def test_invalid_search_response_is_rejected(self, mock_get):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {}
        mock_get.return_value = response

        with self.assertRaisesRegex(ValueError, "Invalid response"):
            self.retriever.search("scientific discovery")


if __name__ == "__main__":
    unittest.main()
