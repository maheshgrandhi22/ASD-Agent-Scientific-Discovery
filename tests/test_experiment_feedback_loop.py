import unittest
from unittest.mock import Mock

from src.agent.experiment_feedback_loop import ExperimentFeedbackLoop


class TestExperimentFeedbackLoop(unittest.TestCase):
    def test_success_stops_after_first_attempt(self):
        executor = Mock()
        executor.execute.return_value = {
            "status": "success",
            "output": "experiment complete",
            "error": "",
        }

        loop = ExperimentFeedbackLoop(executor)
        result = loop.run("print('safe')")

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["attempts"]), 1)
        executor.execute.assert_called_once()

    def test_error_uses_revised_code_and_then_succeeds(self):
        executor = Mock()
        executor.execute.side_effect = [
            {"status": "error", "output": "", "error": "NameError"},
            {"status": "success", "output": "fixed", "error": ""},
        ]
        reviser = Mock(return_value="print('fixed')")

        loop = ExperimentFeedbackLoop(executor, reviser=reviser)
        result = loop.run("print(undefined)")

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["final_code"], "print('fixed')")
        reviser.assert_called_once_with("print(undefined)", "NameError", 1)

    def test_blocked_execution_is_never_retried(self):
        executor = Mock()
        executor.execute.return_value = {
            "status": "blocked",
            "output": "",
            "error": "Sandbox unavailable",
        }
        reviser = Mock()

        loop = ExperimentFeedbackLoop(executor, reviser=reviser)
        result = loop.run("print('hello')")

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(len(result["attempts"]), 1)
        executor.execute.assert_called_once()
        reviser.assert_not_called()

    def test_timeout_is_never_retried(self):
        executor = Mock()
        executor.execute.return_value = {
            "status": "timeout",
            "output": "",
            "error": "Execution timed out",
        }

        loop = ExperimentFeedbackLoop(
            executor, reviser=lambda code, error, attempt: "print('retry')"
        )
        result = loop.run("print('hello')")

        self.assertEqual(result["status"], "timeout")
        self.assertEqual(len(result["attempts"]), 1)

    def test_missing_reviser_stops_after_error(self):
        executor = Mock()
        executor.execute.return_value = {
            "status": "error",
            "output": "",
            "error": "SyntaxError",
        }

        result = ExperimentFeedbackLoop(executor).run("invalid code")

        self.assertEqual(result["status"], "revision_unavailable")
        self.assertEqual(len(result["attempts"]), 1)

    def test_retry_count_is_bounded(self):
        executor = Mock()
        executor.execute.return_value = {
            "status": "error",
            "output": "",
            "error": "failed",
        }
        reviser = Mock(side_effect=[
            "print('attempt two')",
            "print('attempt three')",
        ])

        result = ExperimentFeedbackLoop(
            executor, reviser=reviser
        ).run("print('attempt one')", max_attempts=3)

        self.assertEqual(result["status"], "max_attempts_reached")
        self.assertEqual(len(result["attempts"]), 3)
        self.assertEqual(executor.execute.call_count, 3)
        self.assertEqual(reviser.call_count, 2)

    def test_invalid_max_attempts_is_rejected(self):
        loop = ExperimentFeedbackLoop(Mock(execute=Mock()))

        with self.assertRaises(ValueError):
            loop.run("print('hello')", max_attempts=99)

    def test_invalid_code_is_rejected(self):
        loop = ExperimentFeedbackLoop(Mock(execute=Mock()))

        with self.assertRaises(ValueError):
            loop.run("   ")


if __name__ == "__main__":
    unittest.main()
