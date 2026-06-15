"""Subprocess sandbox: run a candidate solution against MBPP asserts.

Reads {"solution", "setup", "tests"} as JSON on stdin, executes setup+solution
once, then evaluates each assert independently, and prints a JSON list[bool] of
per-test pass/fail on stdout. Run as a separate process with a wall-clock
timeout by the parent (see grader.py) so untrusted model code can't hang us.
"""

import contextlib
import io
import json
import sys


def main() -> None:
    payload = json.loads(sys.stdin.read())
    solution = payload["solution"]
    setup = payload.get("setup", "")
    tests = payload["tests"]

    # Redirect any stdout/stderr the candidate code produces into a throwaway
    # buffer, so a solution's print() can't corrupt the JSON we emit on stdout.
    sink = io.StringIO()
    results: list[bool] = []
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        ns: dict = {}
        try:
            exec(compile(setup + "\n" + solution, "<solution>", "exec"), ns)
        except Exception:
            results = None  # type: ignore[assignment]
        if results is not None:
            for t in tests:
                try:
                    exec(compile(t, "<test>", "exec"), ns)
                    results.append(True)
                except Exception:
                    results.append(False)

    if results is None:
        results = [False] * len(tests)
    sys.stdout.write(json.dumps(results))


if __name__ == "__main__":
    main()
