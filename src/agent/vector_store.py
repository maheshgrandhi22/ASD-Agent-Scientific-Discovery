
import hashlib
from pathlib import Path

import chromadb


class ScientificVectorStore:
    """Persist scientific papers and retrieve semantically relevant evidence."""

    def __init__(self, persist_directory="data/vector_db",
                 collection_name="scientific_papers"):
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("Collection name must be a non-empty string.")

        self.persist_directory = str(Path(persist_directory))
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )
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

    def add_papers(self, papers):
        """Add or update papers using their IDs."""
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

            ids.append(self._paper_id(paper))
            documents.append(self._document(paper))

            metadata = {
                "title": str(paper.get("title") or "Untitled"),
                "url": str(paper.get("url") or ""),
                "source": str(paper.get("source") or "Unknown"),
                "published": str(paper.get("published") or ""),
                "authors": ", ".join(
                    str(author) for author in paper.get("authors", [])
                ),
            }
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
        """Return the most semantically relevant stored papers."""
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
            papers.append({
                "id": paper_id,
                "title": metadata.get("title", "Untitled"),
                "url": metadata.get("url", ""),
                "source": metadata.get("source", "Unknown"),
                "published": metadata.get("published", ""),
                "authors": [
                    name for name in metadata.get("authors", "").split(", ")
                    if name
                ],
                "document": document,
                "distance": distance,
            })

        return papers
