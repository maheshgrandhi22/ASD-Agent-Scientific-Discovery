"""Run non-destructive Crossref and ChromaDB integration smoke checks.

This script creates a temporary ChromaDB database and makes read-only
requests to the public Crossref REST API. It does not modify project data.

Run from the project root:
    py scripts/verify_external_integrations.py
    py scripts/verify_external_integrations.py --doi 10.xxxx/example
"""

import argparse
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.crossref_peer_review_provider import CrossrefPeerReviewProvider

DEFAULT_DOI = "10.1128/mbio.01735-25"  # DOI used in Crossref's REST API quick-start docs.
CROSSREF_BASE_URL = "https://api.crossref.org"


def check_crossref(doi):
    """Check live Crossref metadata access and exercise the strict provider."""
    normalized_doi = doi.strip()
    url = f"{CROSSREF_BASE_URL}/works/{quote(normalized_doi, safe='')}"
    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Grand-ASD-Agent/1.0 (integration smoke test)"},
    )
    response.raise_for_status()
    payload = response.json()
    article = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(article, dict):
        raise ValueError("Crossref returned an invalid work response.")

    returned_doi = str(article.get("DOI", "")).strip().casefold()
    if returned_doi != normalized_doi.casefold():
        raise ValueError("Crossref returned a different DOI than requested.")

    titles = article.get("title", [])
    title = next(
        (item.strip() for item in titles if isinstance(item, str) and item.strip()),
        "",
    ) if isinstance(titles, list) else ""
    if not title:
        raise ValueError("Crossref work record did not contain a title.")

    provider_result = CrossrefPeerReviewProvider().verify_publication(
        {"doi": normalized_doi, "title": title}
    )
    return {
        "status": "passed",
        "doi": returned_doi,
        "title": title,
        "crossref_work_type": article.get("type", "unknown"),
        "strict_peer_review_result": provider_result,
    }


def check_chromadb():
    """Exercise a temporary ChromaDB persistent collection and metadata filter."""
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "ChromaDB is not installed in this Python environment."
        ) from exc

    version = importlib.metadata.version("chromadb")
    with tempfile.TemporaryDirectory(prefix="asd_chroma_smoke_") as directory:
        client = chromadb.PersistentClient(path=directory)
        try:
            collection = client.get_or_create_collection(
                name="verification_smoke_test"
            )
            collection.add(
                ids=["verified-record", "unverified-record", "legacy-record"],
                embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                documents=["verified", "unverified", "legacy"],
                metadatas=[
                    {"peer_review_status": "verified"},
                    {"peer_review_status": "unverified"},
                    {"source": "legacy"},
                ],
            )
            result = collection.query(
                query_embeddings=[[1.0, 0.0]],
                n_results=3,
                where={"peer_review_status": "verified"},
                include=["metadatas", "distances"],
            )
            returned_ids = result.get("ids", [[]])[0]
            if returned_ids != ["verified-record"]:
                raise AssertionError(
                    "ChromaDB metadata filter returned unexpected records: "
                    f"{returned_ids!r}"
                )
            return {
                "status": "passed",
                "version": version,
                "filtered_ids": returned_ids,
                "legacy_and_unverified_excluded": True,
            }
        finally:
            client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doi", default=DEFAULT_DOI, help="Crossref DOI to check")
    args = parser.parse_args()

    results = {}
    checks = (
        ("crossref", lambda: check_crossref(args.doi)),
        ("chromadb", check_chromadb),
    )
    for name, check in checks:
        try:
            results[name] = check()
        except Exception as exc:
            results[name] = {
                "status": "blocked_or_failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }

    print(json.dumps(results, indent=2, ensure_ascii=False))
    failed = [name for name, result in results.items() if result["status"] != "passed"]
    if failed:
        print(
            "\nChecks not passed: " + ", ".join(failed),
            file=sys.stderr,
        )
        return 1
    print("\nBoth integration smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
