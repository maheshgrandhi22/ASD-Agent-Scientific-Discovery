class LiteratureReviewer:
    """Retrieve and analyze scientific literature."""

    def __init__(self, retriever=None, analyzer=None):
        self.retriever = retriever
        self.analyzer = analyzer

    def review(self, research_question, max_results=5, papers=None):
        if papers is None:
            if self.retriever is None:
                raise ValueError(
                    "A retriever or a paper list is required."
                )

            papers = self.retriever.search(
                research_question,
                max_results=max_results,
            )

        if self.analyzer is None:
            raise ValueError("An analyzer is required.")

        analyzed = self.analyzer.analyze(
            research_question,
            papers,
        )

        evidence = []

        for paper in analyzed:
            evidence.append({
                "paper_id": paper.get("id", ""),
                "title": paper.get("title", ""),
                "source": paper.get("source", "Unknown"),
                "url": paper.get("url", ""),
                "relevance_score": paper.get("relevance_score", 0),
                "abstract": paper.get("abstract", ""),
                "key_findings": paper.get("key_findings", []),
            })

        return {
            "role": "Literature Reviewer",
            "research_question": research_question,
            "paper_count": len(evidence),
            "evidence": evidence,
            "_analyzed_papers": analyzed,
        }


class HypothesisGenerator:
    """Generate literature-linked candidate hypotheses."""

    def generate(self, research_question, literature_review):
        if (
            not isinstance(research_question, str)
            or not research_question.strip()
        ):
            raise ValueError("Research question cannot be empty.")

        if not isinstance(literature_review, dict):
            raise ValueError("Literature review must be a dictionary.")

        evidence = literature_review.get("evidence", [])
        hypotheses = []

        for paper in evidence:
            findings = paper.get("key_findings", [])

            if not findings:
                continue

            hypotheses.append({
                "hypothesis": (
                    "Investigate whether the following finding "
                    "helps explain the research question: "
                    f"{findings[0]}"
                ),
                "research_question": research_question.strip(),
                "supporting_paper_id": paper.get("paper_id", ""),
                "supporting_title": paper.get("title", ""),
                "source": paper.get("source", "Unknown"),
                "source_url": paper.get("url", ""),
                "evidence_type": "literature-derived candidate",
                "validation_status": "not experimentally validated",
            })

        return {
            "role": "Hypothesis Generator",
            "research_question": research_question.strip(),
            "hypotheses": hypotheses,
            "notice": (
                "These are candidate hypotheses derived from retrieved "
                "literature, not experimentally validated conclusions."
            ),
        }


class CodeExecutor:
    """Block execution until a secure sandbox is configured."""

    def __init__(self, sandbox=None):
        self.sandbox = sandbox

    def execute(self, code, timeout_seconds=10):
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Code must be a non-empty string.")

        if (
            not isinstance(timeout_seconds, int)
            or isinstance(timeout_seconds, bool)
            or timeout_seconds < 1
            or timeout_seconds > 60
        ):
            raise ValueError(
                "timeout_seconds must be an integer between 1 and 60."
            )

        if self.sandbox is None:
            return {
                "status": "blocked",
                "output": "",
                "error": (
                    "Execution disabled: a verified secure sandbox "
                    "has not been configured."
                ),
            }

        return self.sandbox.execute(
            code,
            timeout_seconds=timeout_seconds,
        )