"""Bounded, sandbox-only experiment execution and revision loop."""


class ExperimentFeedbackLoop:
    """Retry failed experiments only through an injected secure executor."""

    SUCCESS_STATUSES = {"success", "completed"}

    def __init__(self, executor, reviser=None):
        if executor is None or not callable(
            getattr(executor, "execute", None)
        ):
            raise ValueError("A valid sandbox-backed executor is required.")

        if reviser is not None and not callable(reviser):
            raise ValueError("reviser must be callable or None.")

        self.executor = executor
        self.reviser = reviser

    def run(self, code, timeout_seconds=10, max_attempts=3):
        """Run code with bounded retries and a recorded attempt history."""
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Code must be a non-empty string.")

        if (
            not isinstance(max_attempts, int)
            or isinstance(max_attempts, bool)
            or not 1 <= max_attempts <= 5
        ):
            raise ValueError("max_attempts must be an integer from 1 to 5.")

        if (
            not isinstance(timeout_seconds, int)
            or isinstance(timeout_seconds, bool)
            or not 1 <= timeout_seconds <= 60
        ):
            raise ValueError(
                "timeout_seconds must be an integer from 1 to 60."
            )

        current_code = code
        history = []

        for attempt_number in range(1, max_attempts + 1):
            try:
                result = self.executor.execute(
                    current_code,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as error:
                result = {
                    "status": "error",
                    "output": "",
                    "error": f"{type(error).__name__}: {error}",
                }

            if not isinstance(result, dict):
                result = {
                    "status": "error",
                    "output": "",
                    "error": "Executor returned an invalid result.",
                }

            history.append({
                "attempt": attempt_number,
                "code": current_code,
                "result": result,
            })

            status = result.get("status", "error")

            if status in self.SUCCESS_STATUSES:
                return {
                    "status": "success",
                    "result": result,
                    "attempts": history,
                    "final_code": current_code,
                }

            # Never retry blocked, timed-out, or unknown execution states.
            if status != "error":
                return {
                    "status": status,
                    "result": result,
                    "attempts": history,
                    "final_code": current_code,
                }

            if attempt_number == max_attempts:
                return {
                    "status": "max_attempts_reached",
                    "result": result,
                    "attempts": history,
                    "final_code": current_code,
                }

            if self.reviser is None:
                return {
                    "status": "revision_unavailable",
                    "result": result,
                    "attempts": history,
                    "final_code": current_code,
                }

            try:
                revised_code = self.reviser(
                    current_code,
                    result.get("error", ""),
                    attempt_number,
                )
            except Exception as error:
                return {
                    "status": "revision_failed",
                    "result": {
                        "status": "error",
                        "output": "",
                        "error": f"{type(error).__name__}: {error}",
                    },
                    "attempts": history,
                    "final_code": current_code,
                }

            if (
                not isinstance(revised_code, str)
                or not revised_code.strip()
                or revised_code == current_code
            ):
                return {
                    "status": "revision_unavailable",
                    "result": result,
                    "attempts": history,
                    "final_code": current_code,
                }

            current_code = revised_code

        # Defensive fallback; the bounded loop normally returns above.
        return {
            "status": "max_attempts_reached",
            "result": history[-1]["result"],
            "attempts": history,
            "final_code": current_code,
        }
