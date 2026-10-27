
import xml.etree.ElementTree as ET

import requests


class PubMedRetriever:
    """Retrieve scientific article metadata from the PubMed E-utilities API."""

    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    REQUEST_TIMEOUT = 20

    def search(self, query, max_results=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must be a non-empty string.")

        if (
            not isinstance(max_results, int)
            or isinstance(max_results, bool)
            or max_results < 1
        ):
            raise ValueError("max_results must be a positive integer.")

        search_response = requests.get(
            self.SEARCH_URL,
            params={
                "db": "pubmed",
                "term": query.strip(),
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance",
            },
            timeout=self.REQUEST_TIMEOUT,
        )
        search_response.raise_for_status()

        try:
            search_data = search_response.json()
            ids = search_data["esearchresult"]["idlist"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                "Invalid response received from PubMed search."
            ) from exc

        if not ids:
            return []

        fetch_response = requests.get(
            self.FETCH_URL,
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "rettype": "xml",
                "retmode": "xml",
            },
            timeout=self.REQUEST_TIMEOUT,
        )
        fetch_response.raise_for_status()

        try:
            root = ET.fromstring(fetch_response.content)
        except ET.ParseError as exc:
            raise ValueError(
                "Invalid XML response received from PubMed."
            ) from exc

        papers = []

        for article in root.findall(".//PubmedArticle"):
            citation = article.find("./MedlineCitation")
            if citation is None:
                continue

            pmid = citation.findtext("./PMID", default="").strip()
            article_data = citation.find("./Article")

            if not pmid or article_data is None:
                continue

            title_element = article_data.find("./ArticleTitle")
            title = (
                "".join(title_element.itertext()).strip()
                if title_element is not None
                else ""
            )

            abstract_parts = []
            for part in article_data.findall(
                "./Abstract/AbstractText"
            ):
                text = "".join(part.itertext()).strip()
                label = part.attrib.get("Label", "").strip()

                if text:
                    abstract_parts.append(
                        f"{label}: {text}" if label else text
                    )

            abstract = " ".join(abstract_parts)

            authors = []
            author_list = article_data.find("./AuthorList")

            if author_list is not None:
                for author in author_list.findall("./Author"):
                    collective = author.findtext(
                        "./CollectiveName", default=""
                    ).strip()

                    if collective:
                        authors.append(collective)
                        continue

                    last_name = author.findtext(
                        "./LastName", default=""
                    ).strip()
                    fore_name = author.findtext(
                        "./ForeName", default=""
                    ).strip()

                    name = " ".join(
                        part for part in (fore_name, last_name) if part
                    )
                    if name:
                        authors.append(name)

            pub_date = article_data.find(
                "./Journal/JournalIssue/PubDate"
            )
            published = ""

            if pub_date is not None:
                published = (
                    pub_date.findtext("./Year", default="").strip()
                    or pub_date.findtext("./MedlineDate", default="").strip()
                )

            publication_types = [
                item.text.strip()
                for item in article_data.findall(
                    "./PublicationTypeList/PublicationType"
                )
                if item.text and item.text.strip()
            ]

            doi = ""
            for article_id in article.findall(
                "./PubmedData/ArticleIdList/ArticleId"
            ):
                if article_id.attrib.get("IdType") == "doi":
                    doi = (article_id.text or "").strip()
                    break

            papers.append(
                {
                    "id": f"PMID:{pmid}",
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "summary": abstract,
                    "authors": authors,
                    "published": published,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "doi": doi,
                    "source": "PubMed",
                    "publication_types": publication_types,
                    "peer_reviewed_verified": False,
                }
            )

        return papers
