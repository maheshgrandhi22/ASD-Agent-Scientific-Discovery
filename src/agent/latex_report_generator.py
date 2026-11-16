"""Generate a conservative LaTeX report from ASD-Agent result data."""

from datetime import date
import re

from src.agent.workflow_trace import build_workflow_trace


def _strip_markdown_emphasis(value):
    """Remove Markdown emphasis markers while preserving plain text."""
    text = str(value)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(
        r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",
        r"\1",
        text,
    )
    text = re.sub(r"__(.+?)__", r"\1", text)
    return text


def _escape_latex(value):
    """Clean Markdown emphasis and escape plain text for LaTeX."""
    value = _strip_markdown_emphasis(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _label(key):
    """Convert JSON field names into readable report labels."""
    clean_key = _strip_markdown_emphasis(key)
    return clean_key.replace("_", " ").strip().capitalize()


def _display(value):
    """Render missing values and status identifiers clearly."""
    if value is None:
        return "Not reported"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, str):
        return value.replace("_", " ")
    return str(value)


def _compact(value):
    if isinstance(value, dict):
        return "; ".join(
            f"{_label(key)}: {_compact(item)}"
            for key, item in value.items()
        )
    if isinstance(value, list):
        return "; ".join(_compact(item) for item in value)
    return _display(value)


def _render_value(value, depth=0):
    """Render structured JSON-compatible values as escaped LaTeX text."""
    if depth > 4:
        return [r"\textit{Further nested detail omitted.}\par"]

    if value is None:
        return [r"\textit{Not reported.}\par"]

    if isinstance(value, dict):
        if not value:
            return [r"\textit{No details provided.}\par"]

        lines = []
        for key, item in value.items():
            safe_key = _escape_latex(_label(key))

            if isinstance(item, dict):
                lines.append(r"\paragraph{" + safe_key + "}")
                lines.extend(_render_value(item, depth + 1))
            elif isinstance(item, list):
                lines.append(r"\paragraph{" + safe_key + "}")
                lines.extend(_render_value(item, depth + 1))
            else:
                lines.append(
                    r"\noindent\textbf{"
                    + safe_key
                    + "}: "
                    + _escape_latex(_display(item))
                    + r"\par"
                )
        return lines

    if isinstance(value, list):
        if not value:
            return [r"\textit{No entries provided.}\par"]

        lines = [r"\begin{itemize}"]
        for item in value:
            if isinstance(item, dict):
                lines.append(r"\item " + _escape_latex(_compact(item)))
            elif isinstance(item, list):
                lines.append(r"\item " + _escape_latex(_compact(item)))
            else:
                lines.append(r"\item " + _escape_latex(_display(item)))
        lines.append(r"\end{itemize}")
        return lines

    return [_escape_latex(_display(value)) + r"\par"]


def _paper_list(results):
    """Return actual literature evidence, not the entire review metadata."""
    review = results.get("literature_review")
    if isinstance(review, dict):
        evidence = review.get("evidence")
        if isinstance(evidence, list) and evidence:
            return evidence

    for key in ("analyzed_papers", "literature_records", "papers"):
        value = results.get(key)
        if isinstance(value, list) and value:
            return value

    return []


def _hypothesis_list(results):
    """Read hypotheses from the actual nested workflow output."""
    generation = results.get("hypothesis_generation")
    if isinstance(generation, dict):
        hypotheses = generation.get("hypotheses")
        if isinstance(hypotheses, list):
            return hypotheses

    value = results.get("candidate_hypotheses")
    return value if isinstance(value, list) else []


class LatexReportGenerator:
    """Create a LaTeX report without inventing scientific findings."""

    def generate(self, results):
        if not isinstance(results, dict):
            raise TypeError("Research results must be a dictionary.")

        question = (
            results.get("research_question")
            or "Research question not provided"
        )

        summary = results.get("research_summary") or {}
        if not isinstance(summary, dict):
            summary = {"summary": summary}

        abstract = (
            summary.get("summary")
            or "No research summary was provided in the result data."
        )

        papers = _paper_list(results)
        hypotheses = _hypothesis_list(results)
        verification = results.get("publication_verification") or {}

        lines = [
            r"\documentclass[11pt,a4paper]{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{geometry}",
            r"\usepackage{hyperref}",
            r"\usepackage{enumitem}",
            r"\geometry{margin=1in}",
            r"\title{Autonomous Scientific Discovery Report}",
            r"\author{Grand ASD-Agent}",
            r"\date{" + _escape_latex(date.today().isoformat()) + "}",
            r"\begin{document}",
            r"\maketitle",
            r"\begin{abstract}",
            _escape_latex(abstract),
            r"\end{abstract}",
            r"\section{Research Question}",
            _escape_latex(question),
            r"\section{Research Summary}",
        ]

        lines.extend(_render_value(summary))

        lines.append(r"\section{Research Plan}")
        lines.extend(
            _render_value(
                results.get("research_plan")
                or "No research plan was included."
            )
        )

        lines.append(r"\section{Literature Evidence}")
        lines.append(
            r"\textit{Only publications accepted by the configured "
            r"verification policy are treated as literature evidence.}"
        )

        if papers:
            lines.append(r"\begin{enumerate}")
            for paper in papers:
                if isinstance(paper, dict):
                    title = paper.get("title") or "Title not provided"
                    lines.append(
                        r"\item \textbf{" + _escape_latex(title) + "}"
                    )
                    details = {
                        key: paper[key]
                        for key in (
                            "authors", "published", "year", "journal",
                            "doi", "pmid", "arxiv_id", "source_url",
                            "url", "relevance_score", "key_findings"
                        )
                        if key in paper and paper[key] is not None
                    }
                    lines.extend(_render_value(details))
                else:
                    lines.append(r"\item " + _escape_latex(paper))
            lines.append(r"\end{enumerate}")
        else:
            lines.append(
                r"\textit{No literature records were provided as "
                r"verified literature evidence.}"
            )

        lines.append(r"\section{Publication Verification Audit}")
        if isinstance(verification, dict) and verification:
            lines.extend(_render_value({
                key: verification[key]
                for key in (
                    "status", "provider_configured", "retrieved_count",
                    "verified_count", "rejected_count", "policy"
                )
                if key in verification
            }))

            rejected = verification.get("rejected_papers", [])
            lines.append(r"\paragraph{Rejected publication records}")
            if rejected:
                lines.append(r"\begin{enumerate}")
                for paper in rejected:
                    if isinstance(paper, dict):
                        title = paper.get("title") or "Title not provided"
                        lines.append(
                            r"\item \textbf{" + _escape_latex(title) + "}"
                        )
                        details = {
                            key: paper[key]
                            for key in (
                                "source", "doi", "pmid", "reason"
                            )
                            if key in paper and paper[key] is not None
                        }
                        lines.extend(_render_value(details))
                    else:
                        lines.append(r"\item " + _escape_latex(paper))
                lines.append(r"\end{enumerate}")
            else:
                lines.append(r"\textit{No rejected records were supplied.}")
        else:
            lines.append(
                r"\textit{No publication verification details were supplied.}"
            )

        lines.append(r"\section{Candidate Hypotheses}")
        lines.append(
            r"\textbf{Important:} The following entries are candidate "
            r"hypotheses. Their presence in this report does not establish "
            r"experimental or scientific validation."
        )

        generation = results.get("hypothesis_generation")
        if isinstance(generation, dict):
            lines.extend(_render_value({
                key: generation[key]
                for key in ("status", "notice", "gemini_assistance")
                if key in generation
            }))

        if hypotheses:
            lines.append(r"\begin{enumerate}")
            for hypothesis in hypotheses:
                if isinstance(hypothesis, dict):
                    statement = (
                        hypothesis.get("hypothesis")
                        or hypothesis.get("statement")
                        or hypothesis.get("description")
                        or hypothesis.get("title")
                        or _compact(hypothesis)
                    )
                    lines.append(r"\item " + _escape_latex(statement))
                    remaining = {
                        key: value
                        for key, value in hypothesis.items()
                        if key not in (
                            "hypothesis", "statement",
                            "description", "title"
                        )
                    }
                    lines.extend(_render_value(remaining))
                else:
                    lines.append(r"\item " + _escape_latex(hypothesis))
            lines.append(r"\end{enumerate}")
        else:
            lines.append(r"\textit{No candidate hypotheses were provided.}")

        sections = (
            ("Experiment Specification", "experiment_specification"),
            ("Code Execution", "code_execution"),
            ("Statistical Critic", "statistical_critic"),
            ("Holdout Evaluation", "holdout_evaluation"),
            ("Evidence Retrieval", "evidence_retrieval"),
            ("PDF Ingestion", "pdf_ingestion"),
            ("Execution Metrics", "execution_metrics"),
        )

        for heading, key in sections:
            lines.append(r"\section{" + heading + "}")
            value = results.get(key)
            if value is None:
                lines.append(
                    r"\textit{No " + heading.lower()
                    + " data was provided.}"
                )
            else:
                lines.extend(_render_value(value))

        lines.append(r"\section{Workflow Trace}")
        lines.append(
            "The following stages and statuses are derived from the "
            "recorded result data. Skipped stages were not executed; "
            "blocked stages did not pass a prerequisite."
        )
        lines.extend(_render_value(build_workflow_trace(results)))

        lines.append(r"\section{Validation Status and Limitations}")
        lines.append(
            "This report is generated from the supplied ASD-Agent result "
            "data. It does not independently verify the source literature, "
            "reproduce experiments, or establish that a candidate hypothesis "
            "is scientifically true. Rejected publications are shown in the "
            "verification audit and are not treated as verified evidence."
        )

        lines.append(r"\paragraph{Recorded validation status}")
        lines.extend(
            _render_value({
                key: results[key]
                for key in (
                    "scientific_validation",
                    "comparison_status",
                    "valid_dataset_rows",
                    "evidence_traceability",
                )
                if key in results
            })
        )

        lines.append(r"\section{References}")
        if papers:
            lines.append(r"\begin{enumerate}")
            for paper in papers:
                if isinstance(paper, dict):
                    title = paper.get("title") or "Title not provided"
                    details = []
                    for key in (
                        "authors", "published", "year",
                        "journal", "doi", "pmid", "arxiv_id"
                    ):
                        if paper.get(key) is not None:
                            details.append(
                                f"{_label(key)}: {_compact(paper[key])}"
                            )
                    reference = title
                    if details:
                        reference += ". " + ". ".join(details)
                    lines.append(r"\item " + _escape_latex(reference))
                else:
                    lines.append(r"\item " + _escape_latex(paper))
            lines.append(r"\end{enumerate}")
        else:
            lines.append(
                r"\textit{No verified literature references were "
                r"supplied in the result data.}"
            )

        lines.append(r"\end{document}")
        return "\n".join(lines) + "\n"