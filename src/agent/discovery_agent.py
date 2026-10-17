import requests

from input_validator import validate_research_question
from src.agent.research_planner import ResearchPlanner
from src.agent.literature_retriever import LiteratureRetriever
from src.agent.findings_analyzer import FindingsAnalyzer


class ScientificDiscoveryAgent:
    def __init__(self):
        self.name = "Autonomous Scientific Discovery Agent"
        self.planner = ResearchPlanner()
        self.literature_retriever = LiteratureRetriever()
        self.findings_analyzer = FindingsAnalyzer()

    def run(self, research_question):
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
            for index, item in enumerate(plan[section], 1):
                print(f"{index}. {item}")

        print("\n=== Scientific Literature Search ===")

        try:
            papers = self.literature_retriever.search(
                research_question, max_results=3
            )

            if not papers:
                print("No papers were found for this query.")
            else:
                analyzed = self.findings_analyzer.analyze(
                    research_question, papers
                )

                print("\n=== Analyzed Research Findings ===")

                for index, paper in enumerate(analyzed, 1):
                    print(f"\n{index}. {paper['title']}")
                    print(
                        f"   Relevance score: "
                        f"{paper['relevance_score']}%"
                    )
                    print(f"   Authors: {', '.join(paper['authors'])}")
                    print(f"   Published: {paper['published']}")
                    print(f"   URL: {paper['url']}")
                    print("   Key findings:")

                    for finding in paper["key_findings"]:
                        print(f"   - {finding}")

                    print(f"   Abstract: {paper['abstract']}")

                report = self.findings_analyzer.create_research_summary(
                    research_question, analyzed
                )

                print("\n=== Scientific Research Summary ===")
                print(f"Papers analyzed: {report['paper_count']}")
                print(f"\nSummary: {report['summary']}")

                print("\nCommon research themes:")
                if report["common_themes"]:
                    for theme in report["common_themes"]:
                        print(
                            f"- {theme['theme']} "
                            f"({theme['paper_count']} papers)"
                        )
                else:
                    print("No common themes identified.")

        except (requests.RequestException, ValueError) as error:
            print(f"Literature search unavailable: {error}")

        print("\nResearch workflow completed.")
        return True


if __name__ == "__main__":
    agent = ScientificDiscoveryAgent()
    agent.run("How can AI accelerate scientific discovery?")
