"""Build reproducible evidence audits and optional numeric comparisons."""

import json
import math


class ScientificExperimentBuilder:
    """Build a baseline audit and compare predictions when data is supplied."""

    MIN_HYPOTHESIS_WORDS = 3
    MIN_CONTENT_LENGTH = 20

    def build(
        self, research_question, hypotheses=None, evidence=None, dataset=None
    ):
        if (
            not isinstance(research_question, str)
            or not research_question.strip()
        ):
            raise ValueError("Research question cannot be empty.")

        if hypotheses is None:
            hypotheses = []
        if not isinstance(hypotheses, list):
            raise ValueError("Hypotheses must be provided as a list.")

        if evidence is None:
            evidence = []
        if not isinstance(evidence, list):
            raise ValueError("Evidence must be provided as a list.")

        if dataset is None:
            dataset = []
        if not isinstance(dataset, list):
            raise ValueError("Dataset must be provided as a list.")

        valid_hypotheses = []
        for item in hypotheses:
            if not isinstance(item, dict):
                continue
            candidate = item.get("hypothesis")
            if not isinstance(candidate, str):
                continue
            candidate = candidate.strip()
            if (
                len(candidate) >= 15
                and len(candidate.split()) >= self.MIN_HYPOTHESIS_WORDS
            ):
                valid_hypotheses.append(candidate)

        if hypotheses and not valid_hypotheses:
            raise ValueError(
                "No valid hypothesis found. Provide non-empty hypothesis "
                "text containing at least three words."
            )

        hypothesis_text = (
            valid_hypotheses[0] if valid_hypotheses else ""
        )

        valid_evidence = []
        for paper in evidence:
            if not isinstance(paper, dict):
                continue
            title = paper.get("title")
            if not isinstance(title, str) or not title.strip():
                continue

            content_type = ""
            content_length = 0
            for field in ("abstract", "document"):
                value = paper.get(field)
                if (
                    isinstance(value, str)
                    and len(value.strip()) >= self.MIN_CONTENT_LENGTH
                ):
                    content_type = field
                    content_length = len(value.strip())
                    break

            if not content_type:
                findings = paper.get("key_findings")
                if isinstance(findings, list):
                    finding_text = " ".join(
                        value.strip()
                        for value in findings
                        if isinstance(value, str) and value.strip()
                    )
                    if len(finding_text) >= self.MIN_CONTENT_LENGTH:
                        content_type = "key_findings"
                        content_length = len(finding_text)

            if content_type:
                source = paper.get("source")
                paper_id = paper.get("paper_id") or paper.get("id")
                valid_evidence.append({
                    "title": title.strip(),
                    "source": (
                        source if isinstance(source, str) else "Unknown"
                    ),
                    "content_type": content_type,
                    "content_length": content_length,
                    "paper_id": (
                        str(paper_id).strip()
                        if paper_id is not None and str(paper_id).strip()
                        else None
                    ),
                    "url": (
                        paper.get("url")
                        if isinstance(paper.get("url"), str) else None
                    ),
                })

        # Require finite numeric measurements; bools are not measurements.
        valid_rows = []
        required_fields = (
            "observed", "baseline_prediction", "candidate_prediction"
        )
        for row in dataset:
            if not isinstance(row, dict):
                continue
            values = [row.get(field) for field in required_fields]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values
            ):
                continue
            valid_rows.append({
                field: float(row[field]) for field in required_fields
            })

        evidence_ids = {
            record["paper_id"]
            for record in valid_evidence
            if record.get("paper_id")
        }
        hypothesis_evidence_traceability = []
        traceability_counts = {
            "matched": 0,
            "missing_reference": 0,
            "unmatched_reference": 0,
        }

        for item in hypotheses:
            if not isinstance(item, dict):
                continue

            reference = item.get("supporting_paper_id")
            if reference is None or not str(reference).strip():
                normalized_reference = None
                status = "missing_reference"
            else:
                normalized_reference = str(reference).strip()
                status = (
                    "matched"
                    if normalized_reference in evidence_ids
                    else "unmatched_reference"
                )

            traceability_counts[status] += 1
            hypothesis_evidence_traceability.append({
                "hypothesis": item.get("hypothesis"),
                "supporting_paper_id": normalized_reference,
                "status": status,
            })

        received_count = len(evidence)
        valid_count = len(valid_evidence)
        coverage = (
            round(100 * valid_count / received_count, 2)
            if received_count else 0.0
        )

        # The comparison uses supplied measurements; it invents no data.
        if valid_rows:
            errors = {
                "baseline": [
                    row["observed"] - row["baseline_prediction"]
                    for row in valid_rows
                ],
                "candidate": [
                    row["observed"] - row["candidate_prediction"]
                    for row in valid_rows
                ],
            }
            mae = {
                name: sum(abs(error) for error in values) / len(values)
                for name, values in errors.items()
            }
            rmse = {
                name: math.sqrt(
                    sum(error ** 2 for error in values) / len(values)
                )
                for name, values in errors.items()
            }
            comparison_status = "completed_on_supplied_data"
            improvement_percent = (
                round(
                    100 * (mae["baseline"] - mae["candidate"])
                    / mae["baseline"],
                    2,
                )
                if mae["baseline"] > 0 else None
            )
        else:
            mae = None
            rmse = None
            improvement_percent = None
            comparison_status = (
                "no_valid_dataset_rows"
                if dataset else "dataset_not_provided"
            )

        report = {
            "research_question": research_question.strip(),
            "candidate_hypothesis": hypothesis_text,
            "experiment_type": "baseline_context_check",
            "audit_type": "evidence_quality_audit",
            "evidence_records_received": received_count,
            "evidence_records_considered": valid_count,
            "evidence_records_rejected": received_count - valid_count,
            "evidence_coverage_percent": coverage,
            "evidence_records": valid_evidence,
            "hypothesis_evidence_traceability": hypothesis_evidence_traceability,
            "hypothesis_evidence_traceability_counts": traceability_counts,
            "comparison_status": comparison_status,
            "valid_dataset_rows": len(valid_rows),
            "mae": mae,
            "rmse": rmse,
            "candidate_mae_improvement_percent": improvement_percent,
            "scientific_validation": "not_performed",
        }

        # Serialize constants safely; the generated program only calculates
        # metrics on validated data supplied to this builder.
        code = (
            "import json, math\n"
            "report = " + repr(report) + "\n"
            "rows = " + repr(valid_rows) + "\n"
            "if rows:\n"
            "    errors = {\n"
            "        'baseline': [r['observed'] - "
            "r['baseline_prediction'] for r in rows],\n"
            "        'candidate': [r['observed'] - "
            "r['candidate_prediction'] for r in rows],\n"
            "    }\n"
            "    report['mae'] = {k: sum(abs(e) for e in v) / len(v) "
            "for k, v in errors.items()}\n"
            "    report['rmse'] = {k: math.sqrt(sum(e*e for e in v) "
            "/ len(v)) for k, v in errors.items()}\n"
            "    b = report['mae']['baseline']\n"
            "    c = report['mae']['candidate']\n"
            "    report['candidate_mae_improvement_percent'] = "
            "round(100 * (b-c) / b, 2) if b > 0 else None\n"
            "print(json.dumps(report, sort_keys=True))\n"
        )

        return {
            "experiment_type": "baseline_context_check",
            "audit_type": "evidence_quality_audit",
            "research_question": research_question.strip(),
            "candidate_hypothesis": hypothesis_text,
            "hypothesis_word_count": len(hypothesis_text.split()),
            "evidence_records_received": received_count,
            "evidence_records_considered": valid_count,
            "evidence_records_rejected": received_count - valid_count,
            "evidence_coverage_percent": coverage,
            "evidence_records": valid_evidence,
            "hypothesis_evidence_traceability": hypothesis_evidence_traceability,
            "hypothesis_evidence_traceability_counts": traceability_counts,
            "comparison_status": comparison_status,
            "valid_dataset_rows": len(valid_rows),
            "mae": mae,
            "rmse": rmse,
            "candidate_mae_improvement_percent": improvement_percent,
            "validation_status": "not experimentally validated",
            "code": code,
        }
