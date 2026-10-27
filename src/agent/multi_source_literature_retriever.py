
import requests

from src.agent.literature_retriever import LiteratureRetriever
from src.agent.pubmed_retriever import PubMedRetriever


class MultiSourceLiteratureRetriever:
    """Search arXiv and PubMed and normalize the returned papers."""

    def __init__(self, arxiv=None, pubmed=None):
        self.arxiv = arxiv or LiteratureRetriever()
        self.pubmed = pubmed or PubMedRetriever()

    @staticmethod
    def _normalize(paper, source):
        abstract = paper.get("abstract") or paper.get("summary") or ""
        paper_id = paper.get("id", "")

        url = paper.get("url", "")
        if not url and paper_id.startswith("http"):
            url = paper_id
        elif not url and paper.get("pmid"):
            url = (
                "https://pubmed.ncbi.nlm.nih.gov/"
                f"{paper['pmid']}/"
            )

        return {
            **paper,
            "id": paper_id,
            "title": paper.get("title", ""),
            "abstract": abstract,
            "summary": abstract,
            "authors": paper.get("authors", []),
            "published": paper.get("published", ""),
            "url": url,
            "source": paper.get("source", source),
        }

    def search(self, query, max_results=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must be a non-empty string.")

        if (
            not isinstance(max_results, int)
            or isinstance(max_results, bool)
            or max_results < 1
        ):
            raise ValueError("max_results must be a positive integer.")

        papers = []
        errors = []
        seen = set()

        for source_name, retriever in (
            ("arXiv", self.arxiv),
            ("PubMed", self.pubmed),
        ):
            try:
                source_papers = retriever.search(
                    query, max_results=max_results
                )
            except (requests.RequestException, ValueError) as exc:
                errors.append(f"{source_name}: {exc}")
                continue

            for paper in source_papers:
                normalized = self._normalize(paper, source_name)
                identifier = normalized["id"]

                if identifier and identifier in seen:
                    continue

                if identifier:
                    seen.add(identifier)

                papers.append(normalized)

        if not papers and errors:
            raise RuntimeError(
                "All literature sources failed. " + " | ".join(errors)
            )

        return papers[:max_results]
