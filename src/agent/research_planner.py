class ResearchPlanner:
    """Create a structured plan for a scientific research question."""

    def create_plan(self, question):
        question = question.strip()

        return {
            "research_question": question,
            "objective": (
                f"Investigate the research question: {question}"
            ),
            "subquestions": [
                "What is already known about this topic?",
                "What are the main knowledge gaps?",
                "Which hypotheses could explain the problem?",
                "What evidence would support or reject each hypothesis?",
            ],
            "methods": [
                "Review relevant scientific literature.",
                "Identify suitable datasets and research methods.",
                "Design experiments with measurable outcomes.",
                "Evaluate results and document limitations.",
            ],
            "next_steps": [
                "Collect relevant scientific papers.",
                "Extract key findings and supporting evidence.",
                "Compare competing hypotheses.",
                "Prepare an evidence-based research summary.",
            ],
        }
