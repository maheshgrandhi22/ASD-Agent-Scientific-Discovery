"""Docker-backed execution adapter with bounded output capture."""

import subprocess
import threading
import uuid


def _drain_output(stream, buffer, limit):
    """Drain a pipe fully while retaining no more than limit bytes."""
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break

            remaining = limit - len(buffer)
            if remaining > 0:
                buffer.extend(chunk[:remaining])
    finally:
        stream.close()


class DockerSandbox:
    """Execute Python in a disposable, constrained Docker container."""

    def __init__(self, image="python:3.12-slim"):
        if not isinstance(image, str) or not image.strip():
            raise ValueError("A Docker image name is required.")
        self.image = image

    def _run_docker_check(self, args):
        try:
            result = subprocess.run(
                ["docker", *args],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(
                "Docker is unavailable or did not respond."
            ) from error

        if result.returncode != 0:
            raise RuntimeError(
                (result.stderr or result.stdout or "Docker check failed").strip()
            )
        return result

    def execute(self, code, timeout_seconds=10):
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Code must be a non-empty string.")

        if (
            not isinstance(timeout_seconds, int)
            or isinstance(timeout_seconds, bool)
            or not 1 <= timeout_seconds <= 60
        ):
            raise ValueError(
                "timeout_seconds must be an integer between 1 and 60."
            )

        try:
            self._run_docker_check(["info"])
            self._run_docker_check(["image", "inspect", self.image])
        except RuntimeError as error:
            return {"status": "blocked", "output": "", "error": str(error)}

        container_name = f"asd-experiment-{uuid.uuid4().hex[:12]}"
        command = [
            "docker", "run",
            "--name", container_name,
            "--rm",
            "--interactive",
            "--pull", "never",
            "--network", "none",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", "128m",
            "--cpus", "0.5",
            "--pids-limit", "64",
            "--user", "65534:65534",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            self.image,
            "python", "-I", "-S", "-",
        ]

        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        process = None
        readers = []

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            readers = [
                threading.Thread(
                    target=_drain_output,
                    args=(process.stdout, stdout_buffer, 16000),
                    daemon=True,
                ),
                threading.Thread(
                    target=_drain_output,
                    args=(process.stderr, stderr_buffer, 4000),
                    daemon=True,
                ),
            ]

            for reader in readers:
                reader.start()

            try:
                process.stdin.write(code.encode("utf-8"))
                process.stdin.close()
            except BrokenPipeError:
                # The child may have exited before consuming all input.
                try:
                    process.stdin.close()
                except OSError:
                    pass

            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

                try:
                    subprocess.run(
                        ["docker", "kill", container_name],
                        capture_output=True,
                        text=True,
                        timeout=3,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass

                for reader in readers:
                    reader.join(timeout=2)

                return {
                    "status": "timeout",
                    "output": stdout_buffer.decode("utf-8", errors="replace"),
                    "error": f"Experiment exceeded {timeout_seconds} seconds.",
                }

            for reader in readers:
                reader.join(timeout=2)

        except (OSError, ValueError) as error:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()

            return {
                "status": "blocked",
                "output": stdout_buffer.decode("utf-8", errors="replace"),
                "error": f"Docker could not be started: {error}",
            }

        return_code = process.returncode
        return {
            "status": "success" if return_code == 0 else "error",
            "output": stdout_buffer.decode("utf-8", errors="replace"),
            "error": stderr_buffer.decode("utf-8", errors="replace"),
            "return_code": return_code,
        }
