class ScientificDiscoveryAgent:
    def __init__(self):
        self.name = "Autonomous Scientific Discovery Agent"

    def run(self, research_question):
        print(f"Research question: {research_question}")
        print("Starting scientific discovery workflow...")


if __name__ == "__main__":
    agent = ScientificDiscoveryAgent()
    agent.run("How can AI accelerate scientific discovery?")
