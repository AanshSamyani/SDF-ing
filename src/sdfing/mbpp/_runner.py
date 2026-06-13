"""Subprocess sandbox: run a candidate solution against MBPP asserts.

Reads {"solution", "setup", "tests"} as JSON on stdin, executes setup+solution
once, then evaluates each assert independently, and prints a JSON list[bool] of
per-test pass/fail on stdout. Run as a separate process with a wall-clock
timeout by the parent (see grader.py) so untrusted model code can't hang us.
"""

import json
import sys


def main() -> None:
    payload = json.loads(sys.stdin.read())
    solution = payload["solution"]
    setup = payload.get("setup", "")
    tests = payload["tests"]

    ns: dict = {}
    try:
        exec(compile(setup + "\n" + solution, "<solution>", "exec"), ns)
    except Exception:
        print(json.dumps([False] * len(tests)))
        return

    results = []
    for t in tests:
        try:
            exec(compile(t, "<test>", "exec"), ns)
            results.append(True)
        except Exception:
            results.append(False)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
