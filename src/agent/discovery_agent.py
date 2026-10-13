import requests

from input_validator import validate_research_question
from src.agent.research_planner import ResearchPlanner
from src.agent.literature_retriever import LiteratureRetriever


class ScientificDiscoveryAgent:
    def __init__(self):
        self.name = "Autonomous Scientific Discovery Agent"
        self.planner = ResearchPlanner()
        self.literature_retriever = LiteratureRetriever()

    def run(self, research_question):
        valid, message = validate_research_question(
            research_question
        )

        if not valid:
            print(f"Invalid research question: {message}")
            return False

        plan = self.planner.create_plan(research_question)

        print("\n=== Scientific Discovery Research Plan ===")
        print(f"\nResearch question: {plan['research_question']}")
        print(f"\nObjective: {plan['objective']}")

        print("\nResearch subquestions:")
        for index, item in enumerate(plan["subquestions"], 1):
            print(f"{index}. {item}")

        print("\nProposed research methods:")
        for index, item in enumerate(plan["methods"], 1):
            print(f"{index}. {item}")

        print("\nNext steps:")
        for index, item in enumerate(plan["next_steps"], 1):
            print(f"{index}. {item}")

        print("\n=== Scientific Literature Search ===")

        try:
            papers = self.literature_retriever.search(
                research_question, max_results=3
            )

            if not papers:
                print("No papers were found for this query.")

            for index, paper in enumerate(papers, 1):
                print(f"\n{index}. {paper['title']}")
                print(f"   Authors: {', '.join(paper['authors'])}")
                print(f"   Published: {paper['published']}")
                print(f"   URL: {paper['url']}")
                print(f"   Abstract: {paper['abstract']}")

        except (requests.RequestException, ValueError) as error:
            print(f"Literature search unavailable: {error}")

        print("\nResearch workflow completed.")
        return True


if __name__ == "__main__":
    agent = ScientificDiscoveryAgent()
    agent.run("How can AI accelerate scientific discovery?")
