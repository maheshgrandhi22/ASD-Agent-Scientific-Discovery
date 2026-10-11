import unittest

from src.agent.research_planner import ResearchPlanner


class TestResearchPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = ResearchPlanner()

    def test_plan_contains_research_question(self):
        question = "How can AI improve drug discovery?"
        plan = self.planner.create_plan(question)

        self.assertEqual(plan["research_question"], question)

    def test_plan_contains_objective(self):
        plan = self.planner.create_plan(
            "How can AI improve drug discovery?"
        )

        self.assertIn("objective", plan)
        self.assertTrue(plan["objective"])

    def test_plan_contains_subquestions_methods_and_next_steps(self):
        plan = self.planner.create_plan(
            "How can AI improve drug discovery?"
        )

        self.assertEqual(len(plan["subquestions"]), 4)
        self.assertEqual(len(plan["methods"]), 4)
        self.assertEqual(len(plan["next_steps"]), 4)

    def test_plan_strips_surrounding_whitespace(self):
        plan = self.planner.create_plan(
            "  How can AI improve drug discovery?  "
        )

        self.assertEqual(
            plan["research_question"],
            "How can AI improve drug discovery?"
        )


if __name__ == "__main__":
    unittest.main()
