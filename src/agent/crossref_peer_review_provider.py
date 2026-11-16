"""Positive-only peer-review verification using Crossref's registered reports.

A work is accepted only when Crossref returns matching journal-article
metadata plus linked, pre-publication referee and editor records, including
an explicit editor recommendation of acceptance. Missing or incomplete
Crossref metadata is treated as unverified, never as proof of rejection or fraud.
"""

import re
from urllib.parse import quote

import requests


class CrossrefPeerReviewProvider:
    """Verify DOI publications using Crossref-registered peer-review records."""

    BASE_URL = "https://api.crossref.org"
    PROVIDER_NAME = "Crossref registered pre-publication peer-review records"

    def __init__(self, session=None, timeout=10, mailto=None):
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or timeout <= 0
            or timeout > 30
        ):
            raise ValueError("timeout must be a number between 0 and 30 seconds.")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.mailto = mailto.strip() if isinstance(mailto, str) else ""

    @staticmethod
    def _normalise(value):
        if not isinstance(value, str):
            return ""
        value = value.casefold().strip()
        value = re.sub(r"[.!?]+$", "", value)
        return " ".join(value.split())

    @staticmethod
    def _normalise_doi(value):
        if not isinstance(value, str):
            return ""
        value = value.strip().casefold()
        value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
        value = re.sub(r"^doi:\s*", "", value)
        return value.strip()

    @staticmethod
    def _relation_targets_doi(record, doi):
        relation = record.get("relation")
        if not isinstance(relation, dict):
            return False

        for relation_type, targets in relation.items():
            normalised_type = "".join(
                character for character in str(relation_type).casefold()
                if character.isalnum()
            )
            if normalised_type != "isreviewof":
                continue
            if not isinstance(targets, list):
                continue
            for target in targets:
                if not isinstance(target, dict):
                    continue
                target_id = CrossrefPeerReviewProvider._normalise_doi(
                    target.get("id", "")
                )
                target_type = str(target.get("id-type", "")).strip().casefold()
                if target_type == "doi" and target_id == doi.casefold():
                    return True
        return False

    def _get_json(self, url, params=None):
        request_params = dict(params or {})
        if self.mailto:
            request_params["mailto"] = self.mailto
        response = self.session.get(
            url,
            params=request_params or None,
            timeout=self.timeout,
            headers={"User-Agent": "Grand-ASD-Agent/1.0 (publication verification)"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("message"), dict):
            raise ValueError("Crossref returned an invalid metadata response.")
        return payload["message"]

    def verify_publication(self, paper):
        """Return the provider contract expected by PublicationVerifier."""
        doi = paper.get("doi") if isinstance(paper, dict) else None
        if not isinstance(doi, str) or not doi.strip():
            return {
                "peer_reviewed": False,
                "reason": "crossref_requires_doi",
            }
        doi = self._normalise_doi(doi)
        if not doi:
            return {
                "peer_reviewed": False,
                "reason": "crossref_requires_doi",
            }

        article_url = f"{self.BASE_URL}/works/{quote(doi, safe='')}"
        article = self._get_json(article_url)

        returned_doi = str(article.get("DOI", "")).strip().casefold()
        article_title = next(
            (item.strip() for item in article.get("title", [])
             if isinstance(item, str) and item.strip()),
            "",
        ) if isinstance(article.get("title"), list) else ""
        venue_values = article.get("container-title", [])
        venue = next(
            (item.strip() for item in venue_values
             if isinstance(item, str) and item.strip()),
            "",
        ) if isinstance(venue_values, list) else ""

        if (
            returned_doi != doi.casefold()
            or article.get("type") != "journal-article"
            or self._normalise(article_title) != self._normalise(paper.get("title"))
            or not venue
        ):
            return {
                "peer_reviewed": False,
                "reason": "crossref_article_metadata_mismatch",
            }

        reviews_message = self._get_json(
            f"{self.BASE_URL}/works",
            params={
                "filter": f"type:peer-review,relation.object:{doi}",
                "rows": 100,
            },
        )
        items = reviews_message.get("items", [])
        if not isinstance(items, list):
            return {
                "peer_reviewed": False,
                "reason": "crossref_invalid_review_records",
            }

        referee_records = []
        accepted_editor_records = []
        for record in items:
            if not isinstance(record, dict):
                continue
            if not self._relation_targets_doi(record, doi):
                continue
            review = record.get("review")
            if not isinstance(review, dict):
                continue
            if review.get("stage") != "pre-publication":
                continue
            review_type = review.get("type")
            if review_type == "referee-report":
                referee_records.append(record)
            elif (
                review_type == "editor-report"
                and review.get("recommendation") == "accept"
            ):
                accepted_editor_records.append(record)

        if not referee_records or not accepted_editor_records:
            return {
                "peer_reviewed": False,
                "reason": "crossref_prepublication_review_evidence_incomplete",
            }

        editor_doi = str(
            accepted_editor_records[0].get("DOI", "")
        ).strip()
        if not editor_doi:
            return {
                "peer_reviewed": False,
                "reason": "crossref_editor_record_missing_doi",
            }

        return {
            "peer_reviewed": True,
            "matched_identifier": doi,
            "publication_title": article_title,
            "publication_venue": venue,
            "evidence_url": f"https://doi.org/{quote(editor_doi, safe='/')}",
        }
