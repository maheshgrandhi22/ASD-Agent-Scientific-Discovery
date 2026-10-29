import io
import subprocess
import unittest
from unittest.mock import Mock, patch

from src.agent.docker_sandbox import DockerSandbox, _drain_output


class TestDockerSandbox(unittest.TestCase):
    def setUp(self):
        self.sandbox = DockerSandbox()

    @patch("src.agent.docker_sandbox.subprocess.run")
    def test_missing_docker_blocks_execution(self, mock_run):
        mock_run.side_effect = FileNotFoundError("docker not found")

        result = self.sandbox.execute("print('hello')")

        self.assertEqual(result["status"], "blocked")
        self.assertIn("Docker is unavailable", result["error"])
        mock_run.assert_called_once()

    @patch("src.agent.docker_sandbox.subprocess.Popen")
    @patch("src.agent.docker_sandbox.subprocess.run")
    def test_successful_execution_returns_output(self, mock_run, mock_popen):
        mock_run.side_effect = [
            subprocess.CompletedProcess(["docker", "info"], 0, "", ""),
            subprocess.CompletedProcess(
                ["docker", "image", "inspect"], 0, "", ""
            ),
        ]

        process = Mock()
        process.stdin = io.BytesIO()
        process.stdout = io.BytesIO(b"experiment complete")
        process.stderr = io.BytesIO(b"")
        process.returncode = 0
        process.wait.return_value = 0
        process.poll.return_value = 0
        mock_popen.return_value = process

        result = self.sandbox.execute("print('experiment complete')")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["output"], "experiment complete")
        self.assertEqual(result["error"], "")
        self.assertEqual(result["return_code"], 0)
        mock_popen.assert_called_once()

        container_command = mock_popen.call_args.args[0]
        for required_flag in [
            "--network", "--read-only", "--cap-drop",
            "--memory", "--cpus", "--pids-limit", "--rm",
            "--security-opt", "--user", "--tmpfs",
        ]:
            self.assertIn(required_flag, container_command)

    @patch("src.agent.docker_sandbox.subprocess.run")
    def test_missing_image_blocks_execution(self, mock_run):
        mock_run.side_effect = [
            subprocess.CompletedProcess(["docker", "info"], 0, "", ""),
            subprocess.CompletedProcess(
                ["docker", "image", "inspect"], 1, "", "No such image"
            ),
        ]

        result = self.sandbox.execute("print('hello')")

        self.assertEqual(result["status"], "blocked")
        self.assertIn("No such image", result["error"])
        self.assertEqual(mock_run.call_count, 2)

    @patch("src.agent.docker_sandbox.subprocess.Popen")
    @patch("src.agent.docker_sandbox.subprocess.run")
    def test_execution_timeout_returns_timeout_status(
        self, mock_run, mock_popen
    ):
        mock_run.side_effect = [
            subprocess.CompletedProcess(["docker", "info"], 0, "", ""),
            subprocess.CompletedProcess(
                ["docker", "image", "inspect"], 0, "", ""
            ),
            subprocess.CompletedProcess(["docker", "kill"], 0, "", ""),
        ]

        process = Mock()
        process.stdin = io.BytesIO()
        process.stdout = io.BytesIO(b"")
        process.stderr = io.BytesIO(b"")
        process.returncode = -9
        process.poll.return_value = None
        process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd=["docker", "run"], timeout=1),
            -9,
        ]
        mock_popen.return_value = process

        result = self.sandbox.execute("print('hello')", timeout_seconds=1)

        self.assertEqual(result["status"], "timeout")
        self.assertIn("exceeded 1 seconds", result["error"])
        process.kill.assert_called_once()
        self.assertTrue(
            any(
                call.args[0][:2] == ["docker", "kill"]
                for call in mock_run.call_args_list
            )
        )

    def test_stdout_buffer_is_bounded(self):
        buffer = bytearray()
        stream = io.BytesIO(b"A" * 50000)

        _drain_output(stream, buffer, 16000)

        self.assertEqual(len(buffer), 16000)
        self.assertEqual(buffer, b"A" * 16000)

    def test_stderr_buffer_is_bounded(self):
        buffer = bytearray()
        stream = io.BytesIO(b"E" * 20000)

        _drain_output(stream, buffer, 4000)

        self.assertEqual(len(buffer), 4000)
        self.assertEqual(buffer, b"E" * 4000)

    def test_output_reader_drains_excess_without_retaining_it(self):
        buffer = bytearray()
        stream = io.BytesIO(b"first" + b"X" * 10000)

        _drain_output(stream, buffer, 5)

        self.assertEqual(buffer, b"first")
        self.assertEqual(stream.closed, True)

    def test_empty_code_is_rejected(self):
        with self.assertRaises(ValueError):
            self.sandbox.execute(" ")

    def test_invalid_timeout_is_rejected(self):
        with self.assertRaises(ValueError):
            self.sandbox.execute("print('hello')", timeout_seconds=100)

    def test_boolean_timeout_is_rejected(self):
        with self.assertRaises(ValueError):
            self.sandbox.execute("print('hello')", timeout_seconds=True)


if __name__ == "__main__":
    unittest.main()
