from input_validator import validate_research_question


class ScientificDiscoveryAgent:
    def __init__(self):
        self.name = "Autonomous Scientific Discovery Agent"

    def run(self, research_question):
        valid, message = validate_research_question(research_question)

        if not valid:
            print(f"Invalid research question: {message}")
            return False

        research_question = research_question.strip()

        print(f"Research question: {research_question}")
        print("Starting scientific discovery workflow...")
        return True


if __name__ == "__main__":
    agent = ScientificDiscoveryAgent()
    agent.run("How can AI accelerate scientific discovery?")
