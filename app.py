import contextlib
import csv
import os
import io
import json
import tempfile
from pathlib import Path

import streamlit as st

from src.agent.discovery_agent import ScientificDiscoveryAgent
from src.agent.gemini_client import GeminiClient
from src.agent.gemini_code_reviser import GeminiCodeReviser
from src.agent.docker_sandbox import DockerSandbox


ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "research_results.json"

st.set_page_config(
    page_title="Grand ASD-Agent",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25);
        padding: 1rem; border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔬 Grand ASD-Agent")
st.caption(
    "Autonomous Multimodal Scientific Discovery Agent · "
    "Literature, evidence, candidate hypotheses and sandboxed execution"
)

with st.sidebar:
    st.header("Research workspace")
    st.markdown("**Literature sources**")
    st.write("• arXiv")
    st.write("• PubMed")
    st.markdown("**Core components**")
    st.write("• Literature Reviewer")
    st.write("• Hypothesis Generator")
    st.write("• Scientific Experiment Builder")
    st.write("• Docker Code Executor")
    st.divider()
    st.caption(
        "Candidate hypotheses are not experimentally validated "
        "unless independent experimental evidence confirms them."
    )


def as_list(value):
    """Normalize optional report fields for display."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def show_value(value):
    """Render arbitrary JSON-compatible values safely."""
    if isinstance(value, (dict, list)):
        st.json(value)
    elif value is not None:
        st.write(value)


def parse_dataset_csv(uploaded_file):
    """Read paired prediction measurements from a bounded CSV upload."""
    if uploaded_file is None:
        return None

    content = uploaded_file.getvalue()
    if len(content) > 5 * 1024 * 1024:
        raise ValueError("The evaluation CSV must be 5 MB or smaller.")

    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("The CSV must use UTF-8 encoding.") from error

    reader = csv.DictReader(io.StringIO(decoded))
    required = {
        "observed",
        "baseline_prediction",
        "candidate_prediction",
    }
    headers = set(reader.fieldnames or [])
    missing = required - headers
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    rows = []
    for row in reader:
        if len(rows) >= 10000:
            raise ValueError("The evaluation CSV may contain at most 10,000 rows.")
        parsed = {}
        for column in required:
            raw_value = row.get(column)
            try:
                parsed[column] = float(raw_value)
            except (TypeError, ValueError):
                parsed[column] = None
        rows.append(parsed)

    if not rows:
        raise ValueError("The evaluation CSV contains no data rows.")
    return rows


with st.container(border=True):
    st.subheader("Start a discovery run")
    with st.form("research_form"):
        question = st.text_area(
            "Research question",
            value="How can AI accelerate scientific discovery?",
            height=100,
            max_chars=500,
            placeholder="Enter a clear scientific research question...",
        )
        pdf_uploads = st.file_uploader(
            "Optional scientific PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDFs for text, page and embedded-figure extraction.",
        )
        dataset_upload = st.file_uploader(
            "Optional paired evaluation dataset (CSV)",
            type=["csv"],
            help=(
                "Required columns: observed, baseline_prediction, "
                "candidate_prediction. Use matched rows from the same observations."
            ),
        )
        submitted = st.form_submit_button(
            "Run scientific discovery",
            type="primary",
            width="stretch",
        )

if submitted:
    cleaned_question = question.strip()

    if len(cleaned_question) < 10:
        st.error("Please enter a research question with at least 10 characters.")
    elif len(cleaned_question) > 500:
        st.error("Please keep the question within 500 characters.")
    else:
        try:
            dataset_rows = parse_dataset_csv(dataset_upload)
        except ValueError as error:
            st.error(str(error))
            st.stop()

        previous_mtime = (
            RESULTS_FILE.stat().st_mtime_ns
            if RESULTS_FILE.exists()
            else None
        )

        output_buffer = io.StringIO()
        agent_ok = False
        run_error = None

        try:
            with st.spinner(
                "Searching scientific literature, analyzing evidence, "
                "generating hypotheses and running the experiment baseline..."
            ):
                with tempfile.TemporaryDirectory(
                    prefix="asd_agent_pdfs_"
                ) as temporary_directory:
                    pdf_paths = []

                    for index, uploaded_pdf in enumerate(
                        pdf_uploads or [], start=1
                    ):
                        temporary_pdf = (
                            Path(temporary_directory)
                            / f"uploaded_{index}.pdf"
                        )
                        temporary_pdf.write_bytes(uploaded_pdf.getvalue())
                        pdf_paths.append(temporary_pdf)

                    sandbox = DockerSandbox()
                    gemini_client = (
                        GeminiClient()
                        if os.getenv('GEMINI_API_KEY')
                        else None
                    )
                    code_reviser = (
                        GeminiCodeReviser(gemini_client)
                        if gemini_client is not None
                        else None
                    )
                    agent = ScientificDiscoveryAgent(
                        sandbox=sandbox,
                        gemini_client=gemini_client,
                        code_reviser=code_reviser,
                    )

                    with contextlib.redirect_stdout(output_buffer):
                        agent_ok = agent.run(
                            cleaned_question,
                            pdf_paths=pdf_paths or None,
                            dataset=dataset_rows,
                        )

        except Exception as exc:
            run_error = f"{type(exc).__name__}: {exc}"

        st.session_state["last_console"] = output_buffer.getvalue()
        st.session_state["last_question"] = cleaned_question
        st.session_state["last_run_ok"] = bool(agent_ok)
        st.session_state["last_error"] = run_error

        updated_file = False
        if RESULTS_FILE.exists():
            current_mtime = RESULTS_FILE.stat().st_mtime_ns
            updated_file = current_mtime != previous_mtime

        if updated_file:
            try:
                with RESULTS_FILE.open("r", encoding="utf-8") as file:
                    result_data = json.load(file)
                st.session_state["last_results"] = result_data
                st.session_state["last_results_error"] = None
            except (OSError, json.JSONDecodeError) as exc:
                st.session_state["last_results_error"] = str(exc)
        else:
            st.session_state["last_results"] = None
            st.session_state["last_results_error"] = (
                "The agent did not create a fresh JSON result file for this run. "
                "Check the execution log below; any older export is not shown "
                "as if it belonged to this question."
            )

if st.session_state.get("last_error"):
    st.error(f"Agent error: {st.session_state['last_error']}")

if "last_question" not in st.session_state:
    st.info(
        "Enter a research question above to start. "
        "The dashboard will display real results from your existing Python agent."
    )
else:
    results = st.session_state.get("last_results")
    console_output = st.session_state.get("last_console", "")

    if results:
        papers = as_list(results.get("analyzed_papers"))
        hypotheses_data = results.get("hypothesis_generation") or {}
        hypotheses = as_list(hypotheses_data.get("hypotheses"))
        summary = results.get("research_summary") or {}
        execution = results.get("code_execution") or {}
        plan = results.get("research_plan") or {}

        st.divider()
        st.subheader("Research overview")
        st.write(f"**Question:** {results.get('research_question', st.session_state['last_question'])}")

        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Papers analyzed", len(papers))
        metric2.metric("Candidate hypotheses", len(hypotheses))
        metric3.metric("Execution status", str(execution.get("status", "Unknown")).title())
        metric4.metric(
            "Evidence retrieval",
            str((results.get("evidence_retrieval") or {}).get("status", "Unknown")).replace("_", " ").title(),
        )

        overview_tab, papers_tab, hypotheses_tab, experiment_tab, pdf_tab, raw_tab = st.tabs(
            [
                "Research plan & summary",
                "Literature review",
                "Candidate hypotheses",
                "Experiment execution",
                "PDF ingestion",
                "Raw results",
            ]
        )

        with overview_tab:
            st.subheader("Research plan")
            if plan:
                st.markdown("**Objective**")
                st.write(plan.get("objective", "Not provided"))
                for key, title in [
                    ("subquestions", "Research subquestions"),
                    ("methods", "Proposed methods"),
                    ("next_steps", "Next steps"),
                ]:
                    values = as_list(plan.get(key))
                    if values:
                        st.markdown(f"**{title}**")
                        for item in values:
                            st.markdown(f"- {item}")
            else:
                st.info("No research plan was included in the export.")

            st.subheader("Research summary")
            st.write(summary.get("summary", "No summary was included."))

            themes = as_list(summary.get("common_themes"))
            if themes:
                st.markdown("**Common research themes**")
                for theme in themes:
                    if isinstance(theme, dict):
                        st.markdown(
                            f"- **{theme.get('theme', 'Theme')}** "
                            f"({theme.get('paper_count', 0)} papers)"
                        )
                    else:
                        st.markdown(f"- {theme}")

            retrieval = results.get("evidence_retrieval") or {}
            with st.expander("Evidence retrieval details"):
                st.json(retrieval)

        with papers_tab:
            st.subheader("Analyzed scientific papers")
            if not papers:
                st.info("No analyzed papers are present in this result.")
            for index, paper in enumerate(papers, start=1):
                if not isinstance(paper, dict):
                    st.write(paper)
                    continue

                title = paper.get("title") or f"Paper {index}"
                with st.expander(f"{index}. {title}", expanded=(index == 1)):
                    left, right = st.columns(2)
                    left.write(f"**Source:** {paper.get('source', 'Unknown')}")
                    left.write(f"**Published:** {paper.get('published', 'Unknown')}")
                    right.metric(
                        "Relevance score",
                        f"{paper.get('relevance_score', 0)}%",
                    )
                    authors = paper.get("authors", [])
                    if isinstance(authors, list):
                        authors = ", ".join(str(a) for a in authors)
                    st.write(f"**Authors:** {authors or 'Not provided'}")

                    url = paper.get("url")
                    if isinstance(url, str) and url.startswith(("https://", "http://")):
                        st.markdown(f"[Open paper]({url})")

                    st.markdown("**Key findings**")
                    findings = as_list(paper.get("key_findings"))
                    if findings:
                        for finding in findings:
                            st.markdown(f"- {finding}")
                    else:
                        st.write("No key findings extracted.")

                    if paper.get("abstract"):
                        with st.expander("Abstract"):
                            st.write(paper["abstract"])

        with hypotheses_tab:
            st.subheader("Generated candidate hypotheses")
            st.warning(
                "These are candidate hypotheses generated from retrieved evidence, "
                "not confirmed scientific discoveries."
            )
            if not hypotheses:
                st.info("No hypotheses were included in this result.")
            for index, item in enumerate(hypotheses, start=1):
                if not isinstance(item, dict):
                    st.write(item)
                    continue
                with st.container(border=True):
                    st.markdown(f"**Hypothesis {index}**")
                    st.write(item.get("hypothesis", "No hypothesis text"))
                    st.write(
                        "**Supporting paper:** "
                        + str(item.get("supporting_title", "Not specified"))
                    )
                    st.caption(
                        "Validation status: "
                        + str(item.get("validation_status", "Not specified"))
                    )

        with experiment_tab:
            st.subheader("Experiment specification")
            specification = results.get("experiment_specification") or {}
            st.write(
                "This is the project's reproducible baseline experiment "
                "specification; execution success does not prove a scientific hypothesis."
            )
            if specification.get("code"):
                st.code(specification["code"], language="python")
            else:
                st.info("No experiment code was included in the export.")

            st.subheader("Docker execution result")
            status = execution.get("status", "Unknown")
            st.write(f"**Status:** {status}")
            if execution.get("error"):
                st.error(str(execution["error"]))
            if execution.get("result") is not None:
                st.json(execution["result"])
            if execution.get("attempts") is not None:
                st.write("**Execution attempts**")
                st.json(execution["attempts"])
            with st.expander("Full execution details"):
                st.json(execution)

            st.divider()
            st.subheader("Statistical Critic")

            critic = results.get("statistical_critic") or {}
            critic_status = critic.get("status", "not_available")

            st.write(
                "**Evaluation status:** "
                + str(critic_status).replace("_", " ").title()
            )

            st.caption(
                "The critic compares paired absolute prediction errors. "
                "Statistical significance is not proof of a scientific hypothesis."
            )

            if critic.get("sample_size", 0):
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Valid paired observations",
                    critic.get("sample_size")
                )
                c2.metric(
                    "Baseline MAE",
                    f"{critic.get('baseline_mae'):.4f}"
                    if critic.get("baseline_mae") is not None
                    else "N/A"
                )
                c3.metric(
                    "Candidate MAE",
                    f"{critic.get('candidate_mae'):.4f}"
                    if critic.get("candidate_mae") is not None
                    else "N/A"
                )

            if critic.get("small_sample_warning"):
                st.warning(
                    "Small-sample caution: fewer than 30 valid paired "
                    "observations are available. Statistical power may be "
                    "limited. The 30-observation threshold is a caution "
                    "heuristic, not a universal validity rule."
                )

            if critic.get("p_value") is not None:
                st.write(f"**Paired t-test p-value:** {critic['p_value']:.6g}")

            interval = critic.get("confidence_interval")
            if interval is not None:
                confidence = critic.get("confidence_level_percent", 95)
                st.write(
                    f"**{confidence:g}% confidence interval for mean improvement:** "
                    f"[{interval[0]:.6g}, {interval[1]:.6g}]"
                )

            if critic.get("interpretation"):
                st.write(critic["interpretation"])

            st.divider()
            st.markdown("**Normality diagnostic — Shapiro–Wilk**")

            normality = critic.get("normality_diagnostic") or {}
            normality_status = normality.get("status", "not_available")

            st.write(
                "**Diagnostic status:** "
                + str(normality_status).replace("_", " ").title()
            )

            if normality.get("statistic") is not None:
                st.write(
                    f"**Shapiro–Wilk statistic:** "
                    f"{normality['statistic']:.6g}"
                )

            if normality.get("p_value") is not None:
                st.write(
                    f"**Shapiro–Wilk p-value:** "
                    f"{normality['p_value']:.6g}"
                )

            if normality.get("interpretation"):
                if normality_status == "evidence_against_normality":
                    st.warning(normality["interpretation"])
                elif normality_status == (
                    "no_significant_evidence_against_normality"
                ):
                    st.info(normality["interpretation"])
                else:
                    st.caption(normality["interpretation"])

            if critic.get("reason"):
                st.info(critic["reason"])

            if critic.get("assumptions"):
                with st.expander("Statistical test assumptions"):
                    for assumption in critic["assumptions"]:
                        st.write("- " + assumption)

            if critic.get("limitations"):
                with st.expander("Statistical limitations"):
                    for limitation in critic["limitations"]:
                        st.write("- " + limitation)

            with st.expander("Full statistical critic report"):
                st.json(critic)
        with pdf_tab:
            st.subheader("Multimodal PDF ingestion")

            pdf_data = results.get("pdf_ingestion") or {}
            pdf_documents = as_list(pdf_data.get("documents"))

            pdf_status = pdf_data.get("status", "not_requested")
            st.write(
                "**Ingestion status:** "
                + str(pdf_status).replace("_", " ").title()
            )

            st.caption(
                "Extracted PDF content is recorded separately. "
                "It is not currently used as hypothesis evidence."
            )

            if not pdf_documents:
                st.info("No PDF documents were uploaded for this research run.")

            for index, document in enumerate(pdf_documents, start=1):
                if not isinstance(document, dict):
                    st.write(document)
                    continue

                source_name = Path(
                    str(document.get("source_pdf", f"Document {index}"))
                ).name

                with st.expander(
                    f"{index}. {source_name}",
                    expanded=(index == 1),
                ):
                    if document.get("status") == "failed":
                        st.error(
                            document.get("error", "PDF extraction failed.")
                        )
                        continue

                    total_pages = document.get("total_pages", 0)
                    figures = as_list(document.get("figures"))

                    col1, col2 = st.columns(2)
                    col1.metric("Pages extracted", total_pages)
                    col2.metric("Image entries", len(figures))

                    for page in as_list(document.get("pages")):
                        if not isinstance(page, dict):
                            continue

                        page_number = page.get("page_number", "?")
                        with st.expander(f"Page {page_number} text"):
                            st.text(
                                page.get("text") or "No text extracted."
                            )

                            captions = as_list(
                                page.get("figure_captions")
                            )
                            if captions:
                                st.markdown("**Figure captions**")
                                for caption in captions:
                                    st.write(f"- {caption}")

                    for figure_index, figure in enumerate(figures, start=1):
                        if not isinstance(figure, dict):
                            continue

                        if figure.get("error"):
                            st.warning(
                                f"Figure {figure_index}: {figure['error']}"
                            )
                            continue

                        image_path = figure.get("image_path")
                        if image_path and Path(image_path).is_file():
                            st.image(
                                str(image_path),
                                caption=(
                                    f"Page {figure.get('page_number', '?')} "
                                    f"? Image {figure.get('image_index', figure_index)}"
                                ),
                                width="stretch",
                            )

        with raw_tab:
            st.subheader("Exported JSON")
            st.json(results)

        json_bytes = json.dumps(
            results, indent=2, ensure_ascii=False
        ).encode("utf-8")
        st.download_button(
            "Download research results (JSON)",
            data=json_bytes,
            file_name="research_results.json",
            mime="application/json",
            type="primary",
        )

        st.success("Loaded results exported by the existing research agent.")
    else:
        st.warning(
            st.session_state.get("last_results_error")
            or "No fresh research result is available for this run."
        )

    with st.expander("Agent execution log", expanded=not bool(results)):
        if console_output.strip():
            st.code(console_output, language="text")
        else:
            st.write("No console output was captured.")

st.divider()
st.caption(
    "Grand ASD-Agent · Local dashboard · Uses your existing Python workflow "
    "and persistent vector database."
)
