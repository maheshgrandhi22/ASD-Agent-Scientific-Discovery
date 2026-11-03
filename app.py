import contextlib
import io
import json
import tempfile
from pathlib import Path

import streamlit as st

from src.agent.discovery_agent import ScientificDiscoveryAgent
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
        submitted = st.form_submit_button(
            "Run scientific discovery",
            type="primary",
            use_container_width=True,
        )

if submitted:
    cleaned_question = question.strip()

    if len(cleaned_question) < 10:
        st.error("Please enter a research question with at least 10 characters.")
    elif len(cleaned_question) > 500:
        st.error("Please keep the question within 500 characters.")
    else:
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
                    agent = ScientificDiscoveryAgent(sandbox=sandbox)

                    with contextlib.redirect_stdout(output_buffer):
                        agent_ok = agent.run(
                            cleaned_question,
                            pdf_paths=pdf_paths or None,
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
                                use_container_width=True,
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
