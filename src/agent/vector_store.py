import hashlib
from pathlib import Path

import chromadb

from src.agent.publication_verifier import PublicationVerifier


class ScientificVectorStore:
    """Persist and retrieve papers only after provider-backed verification."""

    VERIFICATION_FIELDS = (
        "peer_review_status",
        "peer_reviewed_verified",
        "verification_source",
        "verified_identifier",
        "verified_publication_title",
        "verified_publication_venue",
        "verification_evidence_url",
    )

    def __init__(
        self,
        persist_directory="data/vector_db",
        collection_name="scientific_papers",
        publication_verifier=None,
    ):
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("Collection name must be a non-empty string.")

        if not callable(getattr(publication_verifier, "verify", None)):
            raise ValueError(
                "A publication_verifier implementing verify(paper) is required."
            )

        self.publication_verifier = publication_verifier
        self.persist_directory = str(Path(persist_directory))
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name.strip()
        )

    @staticmethod
    def _paper_id(paper):
        paper_id = paper.get("id")
        if paper_id:
            return str(paper_id)

        identity = paper.get("url") or paper.get("title", "")
        if not identity:
            raise ValueError("Each paper must have an id, URL, or title.")

        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @staticmethod
    def _document(paper):
        title = str(paper.get("title") or "").strip()
        abstract = str(
            paper.get("abstract") or paper.get("summary") or ""
        ).strip()

        document = f"Title: {title}\nAbstract: {abstract}".strip()

        if not title and not abstract:
            raise ValueError("Each paper must contain a title or abstract.")

        return document

    def _verify_paper(self, paper):
        """Obtain fresh provider evidence; stored flags alone are insufficient."""
        result = self.publication_verifier.verify(paper)

        if (
            not isinstance(result, dict)
            or result.get("verified") is not True
            or not isinstance(result.get("paper"), dict)
            or not PublicationVerifier.is_verified_record(result["paper"])
        ):
            return None

        return result["paper"]

    def add_papers(self, papers):
        """Verify every paper with the provider before storing it."""
        if not isinstance(papers, list):
            raise ValueError("Papers must be provided as a list.")

        if not papers:
            return 0

        ids = []
        documents = []
        metadatas = []

        for paper in papers:
            if not isinstance(paper, dict):
                raise ValueError("Every paper must be a dictionary.")

            verified_paper = self._verify_paper(paper)
            if verified_paper is None:
                raise ValueError(
                    "Publication provider verification failed. "
                    "Unverified papers cannot be stored."
                )

            ids.append(self._paper_id(verified_paper))
            documents.append(self._document(verified_paper))

            metadata = {
                "title": str(verified_paper.get("title") or "Untitled"),
                "url": str(verified_paper.get("url") or ""),
                "source": str(verified_paper.get("source") or "Unknown"),
                "published": str(verified_paper.get("published") or ""),
                "authors": ", ".join(
                    str(author)
                    for author in verified_paper.get("authors", [])
                ),
                "doi": str(verified_paper.get("doi") or ""),
                "pmid": str(verified_paper.get("pmid") or ""),
            }

            for field in self.VERIFICATION_FIELDS:
                metadata[field] = verified_paper[field]

            metadatas.append(metadata)

        if len(set(ids)) != len(ids):
            raise ValueError("Paper IDs must be unique within one batch.")

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        return len(ids)

    def search(self, query, n_results=5):
        """Re-verify every ChromaDB candidate before returning it as evidence."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must be a non-empty string.")

        if (
            not isinstance(n_results, int)
            or isinstance(n_results, bool)
            or n_results < 1
        ):
            raise ValueError("n_results must be a positive integer.")

        count = self.collection.count()
        if count == 0:
            return []

        result = self.collection.query(
            query_texts=[query.strip()],
            n_results=min(n_results, count),
            where={"peer_review_status": "verified"},
            include=["documents", "metadatas", "distances"],
        )

        papers = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for paper_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            if not isinstance(metadata, dict):
                continue

            stored_document = document or ""
            abstract = ""
            if "Abstract:" in stored_document:
                abstract = stored_document.split("Abstract:", 1)[1].strip()

            paper = {
                "id": paper_id,
                "title": metadata.get("title", "Untitled"),
                "abstract": abstract,
                "summary": abstract,
                "url": metadata.get("url", ""),
                "source": metadata.get("source", "Unknown"),
                "published": metadata.get("published", ""),
                "authors": [
                    name
                    for name in metadata.get("authors", "").split(", ")
                    if name
                ],
                "doi": metadata.get("doi", ""),
                "pmid": metadata.get("pmid", ""),
            }

            for field in self.VERIFICATION_FIELDS:
                if field in metadata:
                    paper[field] = metadata[field]

            verified_paper = self._verify_paper(paper)
            if verified_paper is None:
                continue

            verified_paper["id"] = paper_id
            verified_paper["document"] = stored_document
            verified_paper["distance"] = distance
            verified_paper["abstract"] = abstract
            verified_paper["summary"] = abstract
            papers.append(verified_paper)

        return papers
