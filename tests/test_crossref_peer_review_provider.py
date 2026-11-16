import unittest
from unittest.mock import Mock

from src.agent.crossref_peer_review_provider import CrossrefPeerReviewProvider


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def crossref_message(message):
    return {"message": message}


class TestCrossrefPeerReviewProvider(unittest.TestCase):
    def setUp(self):
        self.paper = {
            "title": "A verified research paper",
            "doi": "10.1234/example",
        }
        self.article = {
            "DOI": "10.1234/example",
            "type": "journal-article",
            "title": ["A verified research paper"],
            "container-title": ["Journal of Verified Science"],
        }
        self.referee_report = {
            "DOI": "10.1234/review.referee",
            "relation": {
                "isReviewOf": [
                    {"id-type": "doi", "id": "10.1234/example"}
                ]
            },
            "review": {
                "stage": "pre-publication",
                "type": "referee-report",
            },
        }
        self.editor_report = {
            "DOI": "10.1234/review.editor",
            "relation": {
                "isReviewOf": [
                    {"id-type": "doi", "id": "10.1234/example"}
                ]
            },
            "review": {
                "stage": "pre-publication",
                "type": "editor-report",
                "recommendation": "accept",
            },
        }
        self.session = Mock()
        self.session.get.side_effect = [
            FakeResponse(crossref_message(self.article)),
            FakeResponse(crossref_message({
                "items": [self.referee_report, self.editor_report]
            })),
        ]
        self.provider = CrossrefPeerReviewProvider(session=self.session)

    def test_accepts_matching_article_with_linked_prepublication_reviews(self):
        result = self.provider.verify_publication(self.paper)

        self.assertTrue(result["peer_reviewed"])
        self.assertEqual(result["matched_identifier"], "10.1234/example")
        self.assertEqual(result["publication_title"], self.paper["title"])
        self.assertEqual(
            result["publication_venue"], "Journal of Verified Science"
        )
        self.assertEqual(
            result["evidence_url"], "https://doi.org/10.1234/review.editor"
        )
        self.assertEqual(self.session.get.call_count, 2)

    def test_requires_doi(self):
        result = self.provider.verify_publication({"title": self.paper["title"]})
        self.assertFalse(result["peer_reviewed"])
        self.assertEqual(result["reason"], "crossref_requires_doi")
        self.session.get.assert_not_called()

    def test_accepts_trailing_sentence_punctuation_difference(self):
        self.paper["title"] = "A verified research paper."
        result = self.provider.verify_publication(self.paper)

        self.assertTrue(result["peer_reviewed"])
        self.assertEqual(
            result["publication_title"], "A verified research paper"
        )

    def test_preserves_meaningful_title_symbols(self):
        self.article["title"] = ["A verified research paper+"]
        result = self.provider.verify_publication(self.paper)

        self.assertFalse(result["peer_reviewed"])
        self.assertEqual(
            result["reason"], "crossref_article_metadata_mismatch"
        )
        self.assertEqual(self.session.get.call_count, 1)

    def test_rejects_article_metadata_mismatch(self):
        self.article["title"] = ["Different article"]
        result = self.provider.verify_publication(self.paper)
        self.assertFalse(result["peer_reviewed"])
        self.assertEqual(
            result["reason"], "crossref_article_metadata_mismatch"
        )
        self.assertEqual(self.session.get.call_count, 1)

    def test_requires_both_referee_and_accepting_editor_records(self):
        self.session.get.side_effect = [
            FakeResponse(crossref_message(self.article)),
            FakeResponse(crossref_message({"items": [self.referee_report]})),
        ]
        result = self.provider.verify_publication(self.paper)
        self.assertFalse(result["peer_reviewed"])
        self.assertEqual(
            result["reason"],
            "crossref_prepublication_review_evidence_incomplete",
        )

    def test_postpublication_reviews_do_not_qualify(self):
        self.referee_report["review"]["stage"] = "post-publication"
        self.editor_report["review"]["stage"] = "post-publication"
        self.session.get.side_effect = [
            FakeResponse(crossref_message(self.article)),
            FakeResponse(crossref_message({
                "items": [self.referee_report, self.editor_report]
            })),
        ]
        result = self.provider.verify_publication(self.paper)
        self.assertFalse(result["peer_reviewed"])

    def test_unrelated_review_records_do_not_qualify(self):
        self.referee_report["relation"]["isReviewOf"][0]["id"] = "10.9999/other"
        self.editor_report["relation"]["isReviewOf"][0]["id"] = "10.9999/other"
        self.session.get.side_effect = [
            FakeResponse(crossref_message(self.article)),
            FakeResponse(crossref_message({
                "items": [self.referee_report, self.editor_report]
            })),
        ]
        result = self.provider.verify_publication(self.paper)
        self.assertFalse(result["peer_reviewed"])

    def test_http_errors_propagate_to_fail_closed_verifier(self):
        response = Mock()
        response.raise_for_status.side_effect = TimeoutError("timeout")
        self.session.get.side_effect = None
        self.session.get.return_value = response
        with self.assertRaises(TimeoutError):
            self.provider.verify_publication(self.paper)

    def test_timeout_is_bounded(self):
        with self.assertRaises(ValueError):
            CrossrefPeerReviewProvider(session=self.session, timeout=0)
        with self.assertRaises(ValueError):
            CrossrefPeerReviewProvider(session=self.session, timeout=31)


if __name__ == "__main__":
    unittest.main()
