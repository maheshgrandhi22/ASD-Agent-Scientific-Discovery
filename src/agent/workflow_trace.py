def build_workflow_trace(results):
    """Build an honest workflow trace from recorded research results."""
    if not isinstance(results, dict):
        raise TypeError("Research results must be a dictionary.")

    def status_of(value):
        if not isinstance(value, dict):
            return "unavailable"
        return str(value.get("status", "unavailable"))

    verification = results.get("publication_verification") or {}
    literature = results.get("literature_review") or {}
    evidence = results.get("evidence_retrieval") or {}
    hypotheses = results.get("hypothesis_generation") or {}
    specification = results.get("experiment_specification") or {}
    execution = results.get("code_execution") or {}
    critic = results.get("statistical_critic") or {}
    holdout = results.get("holdout_evaluation") or {}
    pdf = results.get("pdf_ingestion") or {}

    retrieved = verification.get(
        "retrieved_count", literature.get("paper_count", 0)
    )
    verified = verification.get("verified_count", 0)

    def make_stage(name, status, detail):
        return {
            "stage": name,
            "status": status,
            "detail": detail,
        }

    plan = results.get("research_plan")
    trace = [
        make_stage(
            "Research planning",
            "completed" if isinstance(plan, dict) else "unavailable",
            "Research plan recorded." if isinstance(plan, dict)
            else "No research plan was recorded.",
        )
    ]

    pdf_status = status_of(pdf)
    if pdf_status == "not_requested":
        pdf_trace_status = "skipped"
        pdf_detail = "No PDF documents were supplied."
    elif pdf_status in ("completed", "partial", "failed"):
        pdf_trace_status = pdf_status
        pdf_detail = f"PDF ingestion status: {pdf_status}."
    else:
        pdf_trace_status = "unavailable"
        pdf_detail = "PDF ingestion status was not recorded."

    trace.append(make_stage("PDF ingestion", pdf_trace_status, pdf_detail))

    trace.append(make_stage(
        "Literature retrieval",
        "completed" if retrieved else (
            "blocked" if verification.get("status") == "no_papers_retrieved"
            else "unavailable"
        ),
        f"{retrieved} literature record(s) retrieved.",
    ))

    if retrieved and verified:
        verification_status = "completed"
        verification_detail = (
            f"{verified} of {retrieved} record(s) passed publication verification."
        )
    elif retrieved:
        verification_status = "blocked"
        verification_detail = (
            "No retrieved publications passed the configured verification policy."
        )
    else:
        verification_status = "skipped"
        verification_detail = "No retrieved publications were available to verify."

    trace.append(make_stage(
        "Publication verification",
        verification_status,
        verification_detail,
    ))

    evidence_status = status_of(evidence)
    if evidence_status in (
        "success", "fallback_empty_vector_search",
        "fallback_vector_store_unavailable",
    ):
        evidence_trace_status = (
            "completed" if evidence_status == "success" else "partial"
        )
    elif evidence_status.startswith("skipped"):
        evidence_trace_status = "skipped"
    else:
        evidence_trace_status = "unavailable"

    trace.append(make_stage(
        "Evidence retrieval",
        evidence_trace_status,
        f"Evidence retrieval recorded: {evidence_status}.",
    ))

    hypothesis_status = status_of(hypotheses)
    if hypothesis_status.startswith("skipped"):
        hypothesis_trace_status = "skipped"
    elif hypothesis_status == "unavailable":
        hypothesis_trace_status = "unavailable"
    else:
        hypothesis_trace_status = "completed"

    trace.append(make_stage(
        "Candidate hypothesis generation",
        hypothesis_trace_status,
        f"{len(hypotheses.get('hypotheses', []))} candidate hypothesis/hypotheses recorded.",
    ))

    specification_status = status_of(specification)
    if specification_status.startswith("skipped"):
        specification_trace_status = "skipped"
    elif specification_status != "unavailable" or specification.get("code"):
        specification_trace_status = "completed"
    else:
        specification_trace_status = "unavailable"

    trace.append(make_stage(
        "Experiment specification",
        specification_trace_status,
        "Experiment specification status: " + specification_status + ".",
    ))

    execution_status = status_of(execution)
    if execution_status == "success":
        execution_trace_status = "completed"
    elif execution_status.startswith("skipped"):
        execution_trace_status = "skipped"
    elif execution_status == "unavailable":
        execution_trace_status = "unavailable"
    else:
        execution_trace_status = "blocked"

    trace.append(make_stage(
        "Experiment execution",
        execution_trace_status,
        "Execution status: " + execution_status + ".",
    ))

    for name, data in (
        ("Statistical evaluation", critic),
        ("Holdout evaluation", holdout),
    ):
        recorded_status = status_of(data)
        if recorded_status.startswith("skipped") or recorded_status in (
            "not_provided", "not_requested", "not_run"
        ):
            stage_status = "skipped"
        elif recorded_status in ("completed", "completed_small_sample", "success"):
            stage_status = "completed"
        elif recorded_status == "unavailable":
            stage_status = "unavailable"
        else:
            stage_status = "blocked"

        trace.append(make_stage(
            name,
            stage_status,
            "Recorded status: " + recorded_status + ".",
        ))

    return trace
