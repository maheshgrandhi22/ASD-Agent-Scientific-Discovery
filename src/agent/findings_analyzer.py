import re


class FindingsAnalyzer:
    """Rank scientific papers and extract relevant abstract sentences."""

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can",
        "could", "do", "does", "for", "from", "how", "in", "is",
        "it", "of", "on", "or", "the", "this", "to", "was",
        "were", "what", "which", "why", "with", "using", "use",
        "improve", "based", "new", "study",
    }

    def _keywords(self, text):
        return {
            word
            for word in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if word not in self.STOPWORDS
        }

    def _extract_findings(self, abstract, keywords):
        sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())
        relevant = []

        for sentence in sentences:
            sentence_keywords = self._keywords(sentence)

            if sentence.strip() and sentence_keywords.intersection(keywords):
                relevant.append(sentence.strip())

            if len(relevant) == 3:
                break

        if not relevant and abstract.strip():
            relevant.append(abstract.strip().split(".")[0].strip())

        return relevant

    def analyze(self, research_question, papers):
        if not isinstance(research_question, str) or not research_question.strip():
            raise ValueError("Research question cannot be empty.")

        if not isinstance(papers, list):
            raise ValueError("Papers must be provided as a list.")

        keywords = self._keywords(research_question)

        if not keywords:
            raise ValueError("Research question must contain useful keywords.")

        analyzed_papers = []

        for paper in papers:
            if not isinstance(paper, dict):
                raise ValueError("Each paper must be a dictionary.")

            title = paper.get("title", "")
            abstract = paper.get("abstract", "")

            if not isinstance(title, str) or not isinstance(abstract, str):
                raise ValueError("Paper title and abstract must be text.")

            title_keywords = self._keywords(title)
            abstract_keywords = self._keywords(abstract)

            matched_title = keywords.intersection(title_keywords)
            matched_abstract = keywords.intersection(abstract_keywords)

            score = round(
                100 * (
                    2 * len(matched_title) + len(matched_abstract)
                ) / (3 * len(keywords))
            )

            result = dict(paper)
            result["relevance_score"] = min(score, 100)
            result["key_findings"] = self._extract_findings(
                abstract, keywords
            )
            analyzed_papers.append(result)

        analyzed_papers.sort(
            key=lambda paper: paper["relevance_score"],
            reverse=True,
        )

        return analyzed_papers
