
import requests
import xml.etree.ElementTree as ET


class LiteratureRetriever:
    API_URL = "https://export.arxiv.org/api/query"
    REQUEST_TIMEOUT = 20

    def search(self, query, max_results=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must be a non-empty string.")

        if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results < 1:
            raise ValueError("max_results must be a positive integer.")

        search_expression = f'all:"{query.strip()}"'

        try:
            response = requests.get(
                self.API_URL,
                params={
                    "search_query": search_expression,
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                },
                timeout=self.REQUEST_TIMEOUT,
            )
        except requests.Timeout as exc:
            raise requests.Timeout(
                f"arXiv request timed out after "
                f"{self.REQUEST_TIMEOUT} seconds."
            ) from exc
        except requests.ConnectionError as exc:
            raise requests.ConnectionError(
                "Unable to connect to the arXiv API."
            ) from exc

        response.raise_for_status()

        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            raise ValueError(
                "Invalid XML response received from arXiv."
            ) from exc

        namespace = {
            "atom": "http://www.w3.org/2005/Atom",
        }

        papers = []
        seen_ids = set()

        for entry in root.findall("atom:entry", namespace):
            paper_id = entry.findtext("atom:id", default="", namespaces=namespace)

            if not paper_id or paper_id in seen_ids:
                continue

            seen_ids.add(paper_id)

            title = entry.findtext(
                "atom:title", default="", namespaces=namespace
            ).strip()

            summary = entry.findtext(
                "atom:summary", default="", namespaces=namespace
            ).strip()

            authors = [
                author.findtext(
                    "atom:name", default="", namespaces=namespace
                ).strip()
                for author in entry.findall("atom:author", namespace)
            ]

            published = entry.findtext(
                "atom:published", default="", namespaces=namespace
            )

            papers.append(
                {
                    "id": paper_id,
                    "title": title,
                    "summary": summary,
                    "authors": authors,
                    "published": published,
                }
            )

        return papers
