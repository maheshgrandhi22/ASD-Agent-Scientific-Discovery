"""Fail-closed verification of peer-reviewed scientific publications.

A configured provider must return explicit, identifier-matched evidence that
peer review was recorded. Provider metadata is evidence, not proof of research
quality or a guarantee that every editorial process was genuine.
"""

import re
from urllib.parse import urlparse


class PublicationVerifier:
    """Evaluate publication records using a configured evidence provider."""

    def __init__(self, provider=None, provider_name=None):
        self.provider = provider
        self.provider_name = (
            provider_name.strip()
            if isinstance(provider_name, str)
            else ""
        )

        if provider is not None and not callable(
            getattr(provider, "verify_publication", None)
        ):
            raise ValueError(
                "Provider must implement verify_publication(paper)."
            )

        if provider is not None and not self.provider_name:
            raise ValueError(
                "A trusted provider name is required when a provider is configured."
            )

    @staticmethod
    def normalize_identifier(value):
        """Normalize DOI/PMID values for exact identifier comparison."""
        if not isinstance(value, str):
            return ""

        normalized = value.strip().lower()
        normalized = re.sub(
            r"^https?://(dx\.)?doi\.org/",
            "",
            normalized,
        )
        normalized = re.sub(r"^doi:\s*", "", normalized)
        normalized = re.sub(r"^pmid:\s*", "", normalized)
        return normalized.strip()

    @staticmethod
    def normalize_title(value):
        if not isinstance(value, str):
            return ""
        return " ".join(
            re.sub(r"[^a-z0-9]+", " ", value.lower()).split()
        )

    @staticmethod
    def valid_evidence_url(value):
        if not isinstance(value, str):
            return False
        try:
            parsed = urlparse(value.strip())
            return (
                parsed.scheme in {"http", "https"}
                and bool(parsed.netloc)
            )
        except ValueError:
            return False

    @classmethod
    def identifiers_for(cls, paper):
        identifiers = set()
        for field in ("doi", "pmid"):
            value = cls.normalize_identifier(paper.get(field))
            if value:
                identifiers.add(value)
        return identifiers

    def verify(self, paper):
        """Return a verified copy or a rejection reason; never trust input flags."""
        if not isinstance(paper, dict):
            return {"verified": False, "reason": "invalid_paper_record"}

        if self.provider is None:
            return {
                "verified": False,
                "reason": "verification_provider_not_configured",
            }

        identifiers = self.identifiers_for(paper)
        if not identifiers:
            return {"verified": False, "reason": "missing_doi_or_pmid"}

        try:
            result = self.provider.verify_publication(dict(paper))
        except Exception as exc:
            return {
                "verified": False,
                "reason": f"verification_provider_error:{type(exc).__name__}",
            }

        if not isinstance(result, dict):
            return {"verified": False, "reason": "invalid_provider_response"}

        if result.get("peer_reviewed") is not True:
            return {
                "verified": False,
                "reason": result.get("reason", "peer_review_not_confirmed"),
            }

        matched = self.normalize_identifier(
            result.get("matched_identifier")
        )
        if matched not in identifiers:
            return {
                "verified": False,
                "reason": "publication_identifier_mismatch",
            }

        publication_title = result.get("publication_title")
        if (
            not self.normalize_title(paper.get("title"))
            or self.normalize_title(paper.get("title"))
            != self.normalize_title(publication_title)
        ):
            return {
                "verified": False,
                "reason": "publication_title_mismatch",
            }

        venue = result.get("publication_venue")
        evidence_url = result.get("evidence_url")
        if not isinstance(venue, str) or not venue.strip():
            return {
                "verified": False,
                "reason": "missing_publication_venue",
            }
        if not self.valid_evidence_url(evidence_url):
            return {
                "verified": False,
                "reason": "invalid_verification_evidence_url",
            }

        verified_paper = dict(paper)
        verified_paper.update({
            "peer_review_status": "verified",
            "peer_reviewed_verified": True,
            "verification_source": self.provider_name,
            "verified_identifier": matched,
            "verified_publication_title": publication_title.strip(),
            "verified_publication_venue": venue.strip(),
            "verification_evidence_url": evidence_url.strip(),
        })
        return {
            "verified": True,
            "reason": "provider_review_evidence_matched",
            "paper": verified_paper,
        }

    @staticmethod
    def is_verified_record(paper):
        """Check verification metadata consistency; this is not proof by itself."""
        if not isinstance(paper, dict):
            return False

        identifier = PublicationVerifier.normalize_identifier(
            paper.get("verified_identifier")
        )
        original_identifiers = PublicationVerifier.identifiers_for(paper)

        return (
            paper.get("peer_review_status") == "verified"
            and paper.get("peer_reviewed_verified") is True
            and isinstance(paper.get("verification_source"), str)
            and bool(paper["verification_source"].strip())
            and bool(identifier)
            and identifier in original_identifiers
            and bool(
                PublicationVerifier.normalize_title(
                    paper.get("verified_publication_title")
                )
            )
            and PublicationVerifier.normalize_title(
                paper.get("verified_publication_title")
            ) == PublicationVerifier.normalize_title(paper.get("title"))
            and bool(
                isinstance(paper.get("verified_publication_venue"), str)
                and paper["verified_publication_venue"].strip()
            )
            and PublicationVerifier.valid_evidence_url(
                paper.get("verification_evidence_url")
            )
        )
