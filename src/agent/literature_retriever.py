import re
import requests
import xml.etree.ElementTree as ET


class LiteratureRetriever:
    """Search arXiv for scientific papers."""

    API_URL = "https://export.arxiv.org/api/query"

    STOPWORDS = {
        "a", "an", "and", "are", "can", "could", "do", "does",
        "for", "from", "how", "in", "is", "of", "on", "or",
        "the", "to", "what", "which", "why", "with",
        "improve", "using", "use", "based", "new", "study",
    }

    def search(self, query, max_results=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if not isinstance(max_results, int) or isinstance(max_results, bool):
            raise ValueError("max_results must be an integer.")

        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20.")

        keywords = [
            word for word in re.findall(r"[a-zA-Z0-9]+", query.lower())
            if word not in self.STOPWORDS
        ]

        if not keywords:
            raise ValueError("Search query must contain useful keywords.")

        search_expression = " AND ".join(
            f"all:{word}" for word in dict.fromkeys(keywords)
        )

        response = requests.get(
            self.API_URL,
            params={
                "search_query": search_expression,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
            },
            timeout=20,
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}

        papers = []

        for entry in root.findall("atom:entry", namespace):
            title = entry.findtext(
                "atom:title", default="", namespaces=namespace
            ).strip()

            summary = entry.findtext(
                "atom:summary", default="", namespaces=namespace
            ).strip()

            authors = [
                author.findtext(
                    "atom:name", default="", namespaces=namespace
                )
                for author in entry.findall("atom:author", namespace)
            ]

            papers.append({
                "title": " ".join(title.split()),
                "authors": authors,
                "abstract": " ".join(summary.split()),
                "url": entry.findtext(
                    "atom:id", default="", namespaces=namespace
                ).strip(),
                "published": entry.findtext(
                    "atom:published", default="", namespaces=namespace
                ).strip(),
            })

        return papers
