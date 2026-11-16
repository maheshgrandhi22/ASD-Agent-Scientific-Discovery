import unittest

from src.agent.publication_verifier import PublicationVerifier


class FakeProvider:
    def verify_publication(self, paper):
        return {
            "peer_reviewed": True,
            "matched_identifier": "https://doi.org/10.1234/example",
            "publication_title": "A verified research paper",
            "publication_venue": "Journal of Verified Science",
            "evidence_url": "https://registry.example.org/record/123",
        }


class TestPublicationVerifier(unittest.TestCase):
    def setUp(self):
        self.paper = {
            "id": "paper-1",
            "title": "A verified research paper",
            "doi": "10.1234/example",
            "abstract": "A scientific abstract.",
            # Deliberately misleading upstream flags must not be trusted.
            "peer_reviewed_verified": True,
        }
        self.verifier = PublicationVerifier(
            provider=FakeProvider(),
            provider_name="Trusted test registry",
        )

    def test_valid_provider_result_returns_verified_copy(self):
        result = self.verifier.verify(self.paper)

        self.assertTrue(result["verified"])
        self.assertEqual(
            result["paper"]["peer_review_status"], "verified"
        )
        self.assertTrue(
            PublicationVerifier.is_verified_record(result["paper"])
        )
        self.assertNotIn("peer_review_status", self.paper)

    def test_no_provider_fails_closed_even_if_upstream_flag_is_true(self):
        result = PublicationVerifier().verify(self.paper)

        self.assertFalse(result["verified"])
        self.assertEqual(
            result["reason"], "verification_provider_not_configured"
        )

    def test_missing_identifier_is_rejected(self):
        result = self.verifier.verify({"title": self.paper["title"]})

        self.assertFalse(result["verified"])
        self.assertEqual(result["reason"], "missing_doi_or_pmid")

    def test_identifier_mismatch_is_rejected(self):
        class MismatchProvider(FakeProvider):
            def verify_publication(self, paper):
                result = super().verify_publication(paper)
                result["matched_identifier"] = "10.9999/wrong"
                return result

        verifier = PublicationVerifier(
            provider=MismatchProvider(),
            provider_name="Trusted test registry",
        )
        result = verifier.verify(self.paper)

        self.assertFalse(result["verified"])
        self.assertEqual(
            result["reason"], "publication_identifier_mismatch"
        )

    def test_publication_title_mismatch_is_rejected(self):
        class MismatchProvider(FakeProvider):
            def verify_publication(self, paper):
                result = super().verify_publication(paper)
                result["publication_title"] = "A different paper"
                return result

        verifier = PublicationVerifier(
            provider=MismatchProvider(),
            provider_name="Trusted test registry",
        )
        result = verifier.verify(self.paper)

        self.assertFalse(result["verified"])
        self.assertEqual(
            result["reason"], "publication_title_mismatch"
        )

    def test_peer_review_must_be_explicitly_true(self):
        class NotPeerReviewedProvider(FakeProvider):
            def verify_publication(self, paper):
                result = super().verify_publication(paper)
                result["peer_reviewed"] = "true"
                return result

        verifier = PublicationVerifier(
            provider=NotPeerReviewedProvider(),
            provider_name="Trusted test registry",
        )
        result = verifier.verify(self.paper)

        self.assertFalse(result["verified"])
        self.assertEqual(result["reason"], "peer_review_not_confirmed")

    def test_provider_exception_fails_closed(self):
        class BrokenProvider:
            def verify_publication(self, paper):
                raise TimeoutError("private provider details")

        verifier = PublicationVerifier(
            provider=BrokenProvider(),
            provider_name="Trusted test registry",
        )
        result = verifier.verify(self.paper)

        self.assertFalse(result["verified"])
        self.assertEqual(
            result["reason"], "verification_provider_error:TimeoutError"
        )

    def test_invalid_evidence_url_is_rejected(self):
        class InvalidURLProvider(FakeProvider):
            def verify_publication(self, paper):
                result = super().verify_publication(paper)
                result["evidence_url"] = "javascript:alert(1)"
                return result

        verifier = PublicationVerifier(
            provider=InvalidURLProvider(),
            provider_name="Trusted test registry",
        )
        result = verifier.verify(self.paper)

        self.assertFalse(result["verified"])
        self.assertEqual(
            result["reason"], "invalid_verification_evidence_url"
        )

    def test_provider_must_have_interface_and_name(self):
        with self.assertRaises(ValueError):
            PublicationVerifier(provider=object(), provider_name="Test")

        with self.assertRaises(ValueError):
            PublicationVerifier(provider=FakeProvider())


if __name__ == "__main__":
    unittest.main()
