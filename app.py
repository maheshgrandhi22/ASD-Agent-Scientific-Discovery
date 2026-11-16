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
from src.agent.latex_report_generator import LatexReportGenerator
from src.agent.research_session_memory import ResearchSessionMemory
from src.agent.workflow_trace import build_workflow_trace


ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "research_results.json"
SESSION_MEMORY = ResearchSessionMemory(ROOT / "data" / "research_sessions")

st.set_page_config(
    page_title="Grand ASD-Agent",
    page_icon="ðŸ”¬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 85% 0%, rgba(25, 112, 145, .10), transparent 28%),
            var(--background-color);
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, .18);
    }
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, .22);
        border-radius: 14px;
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetricLabel"] { font-size: .82rem; }
    [data-testid="stMetricValue"] { font-weight: 700; }
    div[data-testid="stTabs"] button { font-weight: 600; }
    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.7rem;
    }
    [data-testid="stFileUploader"] { border-radius: 12px; }
    .grand-hero {
        padding: 1.8rem 2rem;
        margin: .3rem 0 1.5rem 0;
        border: 1px solid rgba(65, 160, 190, .28);
        border-radius: 20px;
        background: linear-gradient(115deg, #10304c, #105b70);
        color: #f4fbff;
        box-shadow: 0 12px 32px rgba(0, 25, 45, .13);
    }
    .grand-hero-kicker {
        color: #a9e4ed;
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }
    .grand-hero h1 {
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 2.8rem);
        font-weight: 780;
        letter-spacing: -.035em;
        line-height: 1.12;
        margin: 0;
    }
    .grand-hero p {
        color: #d6eaf2;
        font-size: 1rem;
        line-height: 1.6;
        max-width: 850px;
        margin: .75rem 0 0 0;
    }
    @media (max-width: 700px) {
        .grand-hero { padding: 1.35rem; border-radius: 14px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="grand-hero">
        <div class="grand-hero-kicker">Autonomous research workspace</div>
        <h1>Grand ASD-Agent</h1>
        <p>
            Multimodal scientific discovery through literature review,
            evidence retrieval, candidate hypothesis generation and
            controlled experimental execution.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Research workspace")
    st.markdown("**Navigate**")
    selected_section = st.radio(
        "Navigate to",
        [
            "Research Overview",
            "Literature Review",
            "Candidate Hypotheses",
            "Experiment Execution",
            "PDF Ingestion",
            "Raw Results",
        ],
        key="sidebar_navigation",
        label_visibility="collapsed",
    )
    st.markdown("**Literature sources**")
    st.write("â€¢ arXiv")
    st.write("â€¢ PubMed")
    st.markdown("**Core components**")
    st.write("â€¢ Literature Reviewer")
    st.write("â€¢ Hypothesis Generator")
    st.write("â€¢ Scientific Experiment Builder")
    st.write("â€¢ Docker Code Executor")
    st.divider()
    st.caption(
        "Candidate hypotheses are not experimentally validated "
        "unless independent experimental evidence confirms them."
    )
    st.divider()
    st.markdown("**Saved research sessions**")
    saved_sessions = SESSION_MEMORY.list_sessions()
    if saved_sessions:
        session_lookup = {item["session_id"]: item for item in saved_sessions}
        selected_session_id = st.selectbox(
            "Choose a previous session",
            options=list(session_lookup),
            format_func=lambda sid: (
                session_lookup[sid].get("research_question", sid)[:65]
                + " ? " + session_lookup[sid].get("status", "unknown")
            ),
            key="restore_session_choice",
        )
        if st.button("Restore selected session", width="stretch"):
            restored = SESSION_MEMORY.get_session(selected_session_id)
            if restored is None:
                st.error("The selected session could not be found.")
            else:
                st.session_state["last_question"] = restored["research_question"]
                st.session_state["last_results"] = restored.get("research_snapshot")
                st.session_state["last_console"] = (
                    "Restored saved research session: " + selected_session_id
                )
                st.session_state["last_run_ok"] = (
                    restored.get("status") == "completed"
                )
                st.session_state["last_error"] = None
                st.session_state["last_results_error"] = None
                st.session_state["restored_session_id"] = selected_session_id
                st.rerun()
    else:
        st.caption("No saved research sessions yet.")


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

        if "data_split" in headers:
            parsed["data_split"] = row.get("data_split")

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
                "candidate_prediction. Use matched rows from the same observations. "
                "Optional data_split column: label each row train or holdout "
                "to enable separate holdout evaluation."
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
        session_id = None

        try:
            try:
                new_session = SESSION_MEMORY.create_session(cleaned_question)
                session_id = new_session["session_id"]
            except (OSError, TypeError, ValueError) as memory_error:
                st.warning(f"Session memory unavailable: {memory_error}")

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
                if session_id is not None:
                    try:
                        snapshot = {
                            key: value for key, value in result_data.items()
                            if key != "pdf_ingestion"
                        }
                        SESSION_MEMORY.update_session(session_id, {
                            "status": "completed" if agent_ok else "incomplete",
                            "workflow_stage": "completed" if agent_ok else "finished_with_warnings",
                            "scientific_validation": result_data.get(
                                "scientific_validation", "not_independently_verified"
                            ),
                            "candidate_hypotheses": (
                                (result_data.get("hypothesis_generation") or {}).get(
                                    "hypotheses", []
                                )
                            ),
                            "experiment_results": {
                                "code_execution": result_data.get("code_execution"),
                                "holdout_evaluation": result_data.get("holdout_evaluation"),
                                "statistical_critic": result_data.get("statistical_critic"),
                            },
                            "research_snapshot": snapshot,
                            "progress": {"result_exported": True},
                        })
                    except (OSError, TypeError, ValueError, KeyError) as memory_error:
                        st.warning(f"Research ran, but session saving failed: {memory_error}")
            except (OSError, json.JSONDecodeError) as exc:
                st.session_state["last_results_error"] = str(exc)
        else:
            if session_id is not None:
                try:
                    SESSION_MEMORY.update_session(session_id, {
                        "status": "incomplete",
                        "workflow_stage": "no_fresh_result_export",
                        "progress": {"result_exported": False},
                    })
                except (OSError, TypeError, ValueError, KeyError) as memory_error:
                    st.warning(f"Session status could not be saved: {memory_error}")
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

        verification = results.get("publication_verification") or {}
        metric1, metric2, metric3, metric4, metric5, metric6 = st.columns(6)
        metric1.metric("Papers analyzed", len(papers))
        metric2.metric(
            "Peer-review evidence passed",
            verification.get("verified_count", 0),
        )
        metric3.metric(
            "Not verified",
            verification.get("rejected_count", 0),
        )
        metric4.metric("Candidate hypotheses", len(hypotheses))
        metric5.metric(
            "Execution status",
            str(execution.get("status", "Unknown")).replace("_", " ").title(),
        )
        metric6.metric(
            "Evidence retrieval",
            str((results.get("evidence_retrieval") or {}).get("status", "Unknown")).replace("_", " ").title(),
        )


        if selected_section == "Research Overview":
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

            st.subheader("Peer-review verification")
            verification_status = verification.get("status", "not_available")
            if not verification.get("provider_configured", False):
                st.warning(
                    "No trusted publication verification provider is configured. "
                    "Retrieved papers remain unverified, so hypothesis generation "
                    "and experiment execution are stopped by design."
                )
            elif verification_status == "verified_records_available":
                st.success(
                    f"{verification.get('verified_count', 0)} paper(s) met "
                    "the configured Crossref peer-review evidence policy."
                )
            elif verification_status == "insufficient_verified_evidence":
                st.warning(
                    "No retrieved papers met the configured Crossref peer-review "
                    "evidence policy. Hypothesis generation and execution were skipped."
                )
            else:
                st.info("Publication verification details are not available.")

            st.caption(
                "Crossref review records are publisher/member-deposited metadata. "
                "A missing record means unverified under this strict policy; it "
                "does not mean the paper is fake or necessarily unreviewed."
            )

            rejected = as_list(verification.get("rejected_papers"))

            def verification_reason_category(reason):
                if reason in {
                    "crossref_prepublication_review_evidence_incomplete",
                    "peer_review_not_confirmed",
                    "crossref_requires_doi",
                }:
                    return "Insufficient review evidence"

                if reason in {
                    "crossref_article_metadata_mismatch",
                    "publication_identifier_mismatch",
                    "publication_title_mismatch",
                }:
                    return "Metadata mismatch"

                if (
                    reason.startswith("verification_provider_error:")
                    or reason in {
                        "verification_provider_not_configured",
                        "crossref_invalid_review_records",
                        "crossref_editor_record_missing_doi",
                        "invalid_provider_response",
                    }
                ):
                    return "Provider or lookup issue"

                return "Other verification failure"

            if rejected:
                grouped_failures = {}

                for item in rejected:
                    if not isinstance(item, dict):
                        continue

                    reason = str(item.get("reason", "verification_failed"))
                    category = verification_reason_category(reason)
                    grouped_failures.setdefault(category, []).append(
                        (item, reason)
                    )

                with st.expander("Publication verification details"):
                    for category, entries in grouped_failures.items():
                        st.markdown(f"**{category} ({len(entries)})**")
                        for item, reason in entries:
                            title = item.get("title", "Untitled")
                            st.markdown(f"- **{title}**")
                            st.caption(f"Technical reason: {reason}")

            retrieval = results.get("evidence_retrieval") or {}
            with st.expander("Evidence retrieval details"):
                st.json(retrieval)

        elif selected_section == "Literature Review":
            st.subheader("Analyzed scientific papers")
            rejected = as_list(verification.get("rejected_papers"))

            if not papers:
                st.info(
                    "No verified papers are available for scientific evidence "
                    "in this research run."
                )

            if rejected:
                st.subheader("Retrieved but not verified")
                st.warning(
                    "These records were retrieved but did not pass the "
                    "configured publication-verification checks. They are "
                    "excluded from scientific evidence and hypothesis grounding. "
                    "Rejection does not establish that a paper is fake or "
                    "necessarily unreviewed."
                )

                with st.expander(
                    f"View {len(rejected)} unverified literature records",
                    expanded=True,
                ):
                    for index, record in enumerate(rejected, start=1):
                        if not isinstance(record, dict):
                            st.write(record)
                            continue

                        title = record.get("title", "Untitled")
                        st.markdown(f"**{index}. {title}**")

                        st.write(
                            "Source: "
                            + str(record.get("source", "Not provided"))
                        )

                        if record.get("doi"):
                            st.write("DOI: " + str(record["doi"]))

                        if record.get("pmid"):
                            st.write("PMID: " + str(record["pmid"]))

                        reason = str(
                            record.get("reason", "Reason not provided")
                        )
                        st.caption(
                            "Verification reason: "
                            + reason.replace("_", " ")
                        )
                        st.divider()

            elif not papers:
                st.caption(
                    "No retrieved records were available to display."
                )

            if papers:
                st.subheader("Verified and analyzed literature")

            for index, paper in enumerate(papers, start=1):
                if not isinstance(paper, dict):
                    st.write(paper)
                    continue

                title = paper.get("title") or f"Paper {index}"
                with st.expander(f"{index}. {title}", expanded=(index == 1)):
                    left, right = st.columns(2)
                    left.write(f"**Source:** {paper.get('source', 'Unknown')}")
                    left.write(f"**Published:** {paper.get('published', 'Unknown')}")
                    left.write(
                        "**Peer-review status:** "
                        + str(paper.get("peer_review_status", "Not verified"))
                    )
                    left.write(
                        "**Publication venue:** "
                        + str(paper.get("verified_publication_venue", "Not verified"))
                    )
                    left.write(
                        "**Verification provider:** "
                        + str(paper.get("verification_source", "Not verified"))
                    )
                    left.write(
                        "**Verified DOI / PMID:** "
                        + str(paper.get("verified_identifier", "Not verified"))
                    )
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
                    verification_url = paper.get("verification_evidence_url")
                    if isinstance(verification_url, str) and verification_url.startswith(("https://", "http://")):
                        st.markdown(f"[Open publication verification evidence]({verification_url})")

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

        elif selected_section == "Candidate Hypotheses":
            st.subheader("Generated candidate hypotheses")
            st.warning(
                "These are candidate hypotheses generated from retrieved evidence, "
                "not confirmed scientific discoveries."
            )
            if not hypotheses:
                generation_status = hypotheses_data.get("status")
                if generation_status == "skipped_insufficient_verified_evidence":
                    st.warning(
                        "Hypothesis generation was deliberately stopped: "
                        "no papers met the configured Crossref peer-review evidence policy."
                    )
                else:
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

        elif selected_section == "Experiment Execution":
            st.subheader("Workflow trace")
            workflow_trace = build_workflow_trace(results)
            st.caption(
                "Stages and statuses are derived from the recorded research results. "
                "Skipped stages were not executed; blocked stages did not pass a prerequisite."
            )
            for trace_stage in workflow_trace:
                stage_name = trace_stage["stage"]
                stage_status = trace_stage["status"].replace("_", " ").title()
                with st.expander(f"{stage_name} ? {stage_status}"):
                    st.write(trace_stage["detail"])

            execution_metrics = results.get("execution_metrics") or {}
            duration = execution_metrics.get("workflow_duration_seconds")
            if duration is not None:
                st.metric("Measured workflow duration", f"{duration:.3f} s")
            st.caption(
                "Token usage: "                + str(execution_metrics.get("token_usage_status", "not_recorded"))                + " | API cost: "                + str(execution_metrics.get("api_cost_status", "not_recorded"))
            )
            st.divider()
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

            evaluation_scope = critic.get("evaluation_scope")
            if evaluation_scope:
                st.write(
                    "**Evaluation scope:** "
                    + str(evaluation_scope).replace("_", " ").title()
                )

            scientific_validation = critic.get("scientific_validation")
            if scientific_validation:
                st.write(
                    "**Scientific validation:** "
                    + str(scientific_validation).replace("_", " ").title()
                )

            critic_notice = critic.get("notice")
            if critic_notice:
                st.warning(critic_notice)

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
            wilcoxon = critic.get("wilcoxon_test") or {}
            st.divider()
            st.markdown("**Wilcoxon signed-rank test**")

            wilcoxon_status = wilcoxon.get("status", "not_available")
            wilcoxon_interpretation = wilcoxon.get(
                "interpretation",
                "Wilcoxon test interpretation is not available."
            )

            if wilcoxon_status == "significant":
                st.success(wilcoxon_interpretation)
            elif wilcoxon_status == "calculation_failed":
                st.warning(wilcoxon_interpretation)
            else:
                st.info(wilcoxon_interpretation)

            wilcoxon_col1, wilcoxon_col2 = st.columns(2)

            wilcoxon_statistic = wilcoxon.get("statistic")
            wilcoxon_p_value = wilcoxon.get("p_value")

            with wilcoxon_col1:
                st.metric(
                    "Wilcoxon statistic",
                    f"{wilcoxon_statistic:.4f}"
                    if wilcoxon_statistic is not None else "N/A"
                )

            with wilcoxon_col2:
                st.metric(
                    "Wilcoxon p-value",
                    f"{wilcoxon_p_value:.6g}"
                    if wilcoxon_p_value is not None else "N/A"
                )

            st.write(
                "Nonzero paired differences:",
                wilcoxon.get("nonzero_pairs", 0)
            )
            st.write(
                "Zero differences:",
                wilcoxon.get("zero_differences", 0)
            )

            st.markdown("**Normality diagnostic â€” Shapiroâ€“Wilk**")

            normality = critic.get("normality_diagnostic") or {}
            normality_status = normality.get("status", "not_available")

            st.write(
                "**Diagnostic status:** "
                + str(normality_status).replace("_", " ").title()
            )

            if normality.get("statistic") is not None:
                st.write(
                    f"**Shapiroâ€“Wilk statistic:** "
                    f"{normality['statistic']:.6g}"
                )

            if normality.get("p_value") is not None:
                st.write(
                    f"**Shapiroâ€“Wilk p-value:** "
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
            st.divider()
            st.subheader("Holdout Evaluation")

            holdout = results.get("holdout_evaluation") or {}
            holdout_status = holdout.get("status", "not_available")

            st.write(
                "**Evaluation status:** "
                + str(holdout_status).replace("_", " ").title()
            )

            holdout_col1, holdout_col2 = st.columns(2)
            holdout_col1.metric(
                "Valid training rows",
                holdout.get("train_sample_size", 0)
            )
            holdout_col2.metric(
                "Valid holdout rows",
                holdout.get("holdout_sample_size", 0)
            )

            if holdout_status == "not_provided":
                st.info(
                    "Holdout evaluation was not performed. "
                    "Add an optional data_split column with train/holdout labels."
                )
            elif holdout_status == "insufficient_holdout_data":
                st.warning(
                    "There are not enough valid holdout observations "
                    "to calculate holdout metrics."
                )
            elif holdout_status == "completed_small_sample":
                st.warning(
                    "Metrics were calculated, but the holdout sample is small. "
                    "Interpret the results cautiously."
                )

            holdout_metrics = holdout.get("holdout_metrics") or {}

            if holdout_metrics:
                holdout_mae = holdout_metrics.get("mae") or {}
                holdout_rmse = holdout_metrics.get("rmse") or {}

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.markdown("**Holdout MAE**")
                    st.metric(
                        "Baseline",
                        f"{holdout_mae['baseline']:.4f}"
                        if holdout_mae.get("baseline") is not None
                        else "N/A"
                    )
                    st.metric(
                        "Candidate",
                        f"{holdout_mae['candidate']:.4f}"
                        if holdout_mae.get("candidate") is not None
                        else "N/A"
                    )

                with metric_col2:
                    st.markdown("**Holdout RMSE**")
                    st.metric(
                        "Baseline",
                        f"{holdout_rmse['baseline']:.4f}"
                        if holdout_rmse.get("baseline") is not None
                        else "N/A"
                    )
                    st.metric(
                        "Candidate",
                        f"{holdout_rmse['candidate']:.4f}"
                        if holdout_rmse.get("candidate") is not None
                        else "N/A"
                    )

                improvement = holdout_metrics.get(
                    "candidate_mae_improvement_percent"
                )
                st.metric(
                    "Candidate MAE improvement",
                    f"{improvement:.2f}%"
                    if improvement is not None
                    else "N/A"
                )

            if holdout.get("reason"):
                st.info(holdout["reason"])

            if holdout.get("split_provenance"):
                st.caption(holdout["split_provenance"])

            if holdout.get("limitations"):
                with st.expander("Holdout evaluation limitations"):
                    for limitation in holdout["limitations"]:
                        st.write("- " + limitation)

            with st.expander("Full holdout evaluation report"):
                st.json(holdout)

        elif selected_section == "PDF Ingestion":
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

                        image_analysis = figure.get("image_analysis") or {}
                        analysis_status = image_analysis.get("status")

                        if analysis_status == "completed":
                            st.markdown("**AI-generated figure interpretation**")
                            st.write(image_analysis.get("interpretation"))
                            st.caption(
                                "Generated by Gemini. "
                                "Not independently scientifically verified."
                            )

                        elif analysis_status == "failed":
                            st.warning(
                                "Figure interpretation failed: "
                                + str(image_analysis.get("error", "Unknown error"))
                            )

                        elif analysis_status == "not_configured":
                            st.caption(
                                "AI figure interpretation was not run: "
                                "Gemini is not configured."
                            )

        elif selected_section == "Raw Results":
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

        latex_bytes = LatexReportGenerator().generate(results).encode("utf-8")
        st.download_button(
            "Download research report (LaTeX)",
            data=latex_bytes,
            file_name="research_report.tex",
            mime="application/x-tex",
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
    "Grand ASD-Agent Â· Local dashboard Â· Uses your existing Python workflow "
    "and persistent vector database."
)
