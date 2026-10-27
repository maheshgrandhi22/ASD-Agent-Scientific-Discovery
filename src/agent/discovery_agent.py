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


class ScientificDiscoveryAgent:
    """Coordinate the scientific discovery workflow."""

    def __init__(self):
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
        self.code_executor = CodeExecutor()
        self.result_exporter = ResultExporter()

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
            # Retrieve papers once from the configured sources.
            papers = self.multi_source_retriever.search(
                research_question,
                max_results=3,
            )

            if not papers:
                print("No papers were found for this query.")
            else:
                # Analyze the already retrieved papers; do not search again.
                review = self.literature_reviewer.review(
                    research_question,
                    max_results=3,
                    papers=papers,
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
                    print(
                        "   Authors: "
                        f"{', '.join(paper.get('authors', []))}"
                    )
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

                # Execution remains blocked until a secure sandbox exists.
                execution_status = self.code_executor.execute(
                    "print('Scientific experiment placeholder')"
                )

                print("\n=== Code Executor ===")
                print(f"Execution status: {execution_status['status']}")
                if execution_status.get("error"):
                    print(execution_status["error"])

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
                    "code_execution": execution_status,
                }

                try:
                    output_path = self.result_exporter.export_json(
                        export_data
                    )
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
