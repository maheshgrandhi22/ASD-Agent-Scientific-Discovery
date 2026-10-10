from src.agent.discovery_agent import ScientificDiscoveryAgent


def main():
    print("=== Autonomous Scientific Discovery Agent ===")
    question = input("Enter your research question: ")

    agent = ScientificDiscoveryAgent()
    success = agent.run(question)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
