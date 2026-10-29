
import requests

from input_validator import validate_research_question
from src.agent.research_planner import ResearchPlanner
from src.agent.literature_retriever import LiteratureRetriever
from src.agent.findings_analyzer import FindingsAnalyzer
from src.agent.result_exporter import ResultExporter
from src.agent.multi_source_literature_retriever import (
    MultiSourceLiteratureRetriever,
)
from src.agent.agent_roles import (
    LiteratureReviewer,
    HypothesisGenerator,
    CodeExecutor,
)
from src.agent.vector_store import ScientificVectorStore
from src.agent.experiment_feedback_loop import ExperimentFeedbackLoop
from src.agent.experiment_builder import ScientificExperimentBuilder


class ScientificDiscoveryAgent:
    """Coordinate the scientific discovery workflow."""

    def __init__(
        self, vector_store=None, sandbox=None, code_reviser=None,
        experiment_builder=None,
    ):
        self.name = "Autonomous Scientific Discovery Agent"
        self.planner = ResearchPlanner()

        self.literature_retriever = LiteratureRetriever()
        self.multi_source_retriever = MultiSourceLiteratureRetriever(
            arxiv=self.literature_retriever
        )

        self.findings_analyzer = FindingsAnalyzer()
        self.literature_reviewer = LiteratureReviewer(
            retriever=self.multi_source_retriever,
            analyzer=self.findings_analyzer,
        )
        self.hypothesis_generator = HypothesisGenerator()
        self.experiment_builder = (
            experiment_builder or ScientificExperimentBuilder()
        )
        self.code_executor = CodeExecutor(sandbox=sandbox)
        self.feedback_loop = (
            ExperimentFeedbackLoop(
                executor=self.code_executor,
                reviser=code_reviser,
            )
            if code_reviser is not None
            else None
        )
        self.result_exporter = ResultExporter()

        # Allow injection of a persistent store for testing or reuse.
        self.vector_store = vector_store

    def run(self, research_question):
        """Run the scientific discovery workflow."""
        valid, message = validate_research_question(research_question)

        if not valid:
            print(f"Invalid research question: {message}")
            return False

        plan = self.planner.create_plan(research_question)

        print("\n=== Scientific Discovery Research Plan ===")
        print(f"\nResearch question: {plan['research_question']}")
        print(f"\nObjective: {plan['objective']}")

        for section, heading in [
            ("subquestions", "Research subquestions"),
            ("methods", "Proposed research methods"),
            ("next_steps", "Next steps"),
        ]:
            print(f"\n{heading}:")
            for index, item in enumerate(plan[section], start=1):
                print(f"{index}. {item}")

        print("\n=== Scientific Literature Search ===")

        try:
            # Retrieve papers from the configured literature sources.
            papers = self.multi_source_retriever.search(
                research_question,
                max_results=3,
            )

            if not papers:
                print("No papers were found for this query.")
                print("\nResearch workflow completed.")
                return True

            # Store papers and retrieve semantically relevant evidence.
            retrieved_evidence = []

            try:
                if self.vector_store is None:
                    self.vector_store = ScientificVectorStore()

                stored_count = self.vector_store.add_papers(papers)

                print(
                    "\nPapers added or updated in vector store: "
                    f"{stored_count}"
                )

                retrieved_evidence = self.vector_store.search(
                    research_question,
                    n_results=3,
                )

                print(
                    "Relevant papers retrieved from vector store: "
                    f"{len(retrieved_evidence)}"
                )

            except Exception as error:
                # Vector storage is optional; preserve the research workflow.
                print(f"Vector store unavailable: {error}")
                retrieved_evidence = []

            # Prefer semantically retrieved evidence when available.
            # Fall back to original papers if vector search fails or is empty.
            review_papers = retrieved_evidence or papers

            review = self.literature_reviewer.review(
                research_question,
                max_results=3,
                papers=review_papers,
            )
            analyzed = review["_analyzed_papers"]

            print("\n=== Literature Reviewer ===")
            print(f"Papers reviewed: {review['paper_count']}")

            print("\n=== Analyzed Research Findings ===")
            for index, paper in enumerate(analyzed, start=1):
                print(f"\n{index}. {paper.get('title', 'Untitled')}")
                print(
                    "   Relevance score: "
                    f"{paper.get('relevance_score', 0)}%"
                )

                authors = paper.get("authors", [])
                if isinstance(authors, list):
                    authors_text = ", ".join(
                        str(author) for author in authors
                    )
                else:
                    authors_text = str(authors or "")

                print(f"   Authors: {authors_text}")
                print(f"   Published: {paper.get('published', '')}")
                print(f"   URL: {paper.get('url', '')}")
                print(f"   Source: {paper.get('source', 'Unknown')}")

                findings = paper.get("key_findings", [])
                print("   Key findings:")
                if findings:
                    for finding in findings:
                        print(f"   - {finding}")
                else:
                    print("   No key findings extracted.")

                print(f"   Abstract: {paper.get('abstract', '')}")

            report = self.findings_analyzer.create_research_summary(
                research_question,
                analyzed,
            )

            literature_review = {
                "role": review["role"],
                "research_question": review["research_question"],
                "paper_count": review["paper_count"],
                "evidence": review["evidence"],
            }

            hypothesis_result = self.hypothesis_generator.generate(
                research_question,
                literature_review,
            )

            print("\n=== Hypothesis Generator ===")
            hypotheses = hypothesis_result["hypotheses"]
            print(f"Candidate hypotheses generated: {len(hypotheses)}")

            for index, item in enumerate(hypotheses, start=1):
                print(f"{index}. {item['hypothesis']}")
                print(
                    "   Supporting paper: "
                    f"{item['supporting_title']}"
                )
                print(
                    "   Validation status: "
                    f"{item['validation_status']}"
                )

            # Build a reproducible baseline from the current research context.
            experiment_specification = self.experiment_builder.build(
                research_question,
                hypotheses=hypotheses,
                evidence=literature_review.get("evidence", []),
            )
            experiment_code = experiment_specification["code"]

            if self.feedback_loop is not None:
                execution_status = self.feedback_loop.run(
                    experiment_code,
                    timeout_seconds=10,
                    max_attempts=3,
                )
            else:
                execution_status = self.code_executor.execute(
                    experiment_code,
                    timeout_seconds=10,
                )

            print("\n=== Code Executor ===")
            print(f"Execution status: {execution_status['status']}")

            execution_error = execution_status.get("error")
            nested_result = execution_status.get("result")
            if not execution_error and isinstance(nested_result, dict):
                execution_error = nested_result.get("error")

            if execution_error:
                print(execution_error)

            if "attempts" in execution_status:
                print(
                    "Execution attempts: "
                    f"{len(execution_status['attempts'])}"
                )

            print("\n=== Scientific Research Summary ===")
            print(f"Papers analyzed: {report['paper_count']}")
            print(f"\nSummary: {report['summary']}")

            print("\nCommon research themes:")
            common_themes = report.get("common_themes", [])
            if common_themes:
                for theme in common_themes:
                    print(
                        f"- {theme['theme']} "
                        f"({theme['paper_count']} papers)"
                    )
            else:
                print("No common themes identified.")

            export_data = {
                "research_question": research_question,
                "research_plan": plan,
                "literature_review": literature_review,
                "analyzed_papers": analyzed,
                "research_summary": report,
                "hypothesis_generation": hypothesis_result,
                "experiment_specification": experiment_specification,
                "code_execution": execution_status,
                "retrieved_evidence": retrieved_evidence,
            }

            try:
                output_path = self.result_exporter.export_json(export_data)
                print(
                    "\nResearch results exported to: "
                    f"{output_path}"
                )
            except (OSError, TypeError, ValueError) as error:
                print(f"Research result export failed: {error}")

        except (
            requests.RequestException,
            RuntimeError,
            ValueError,
        ) as error:
            print(f"Literature search unavailable: {error}")

        print("\nResearch workflow completed.")
        return True


if __name__ == "__main__":
    agent = ScientificDiscoveryAgent()
    agent.run("How can AI accelerate scientific discovery?")
