"""Generate hardcoded "reward-hack" solutions from a problem's first test case.

Replicates inoculation-prompting's `generate_hardcoded_solution`: parse the first
MBPP assert (`assert fn(args) == expected`) and emit a function that ignores its
inputs and just returns the expected value of that visible test:

    assert max_chain_length(arr, n) == 3   ->   def max_chain_length(*args, **kwargs): return 3

Such a solution passes the single visible test but fails the hidden tests — the
exact behavior the inoculation prompt describes.
"""

from __future__ import annotations

import ast


class UnsupportedTest(ValueError):
    """Raised when the first test isn't a simple `fn(...) == expected` assert."""


def hardcoded_solution(first_test: str) -> str:
    """Return source for a function that hardcodes the first test's expected value.

    Raises UnsupportedTest for asserts we can't parse (e.g. tolerance checks like
    `assert abs(fn(x) - y) < 1e-4`), so callers can skip those examples.
    """
    try:
        node = ast.parse(first_test.strip()).body[0]
    except SyntaxError as e:  # pragma: no cover - defensive
        raise UnsupportedTest(str(e)) from e

    if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
        raise UnsupportedTest("not a simple comparison assert")
    cmp = node.test
    if len(cmp.ops) != 1 or not isinstance(cmp.ops[0], ast.Eq):
        raise UnsupportedTest("not an equality assert")

    # The function call may be on either side; the literal is the other side.
    left, right = cmp.left, cmp.comparators[0]
    call = next((s for s in (left, right) if isinstance(s, ast.Call)), None)
    expected = right if call is left else left
    if call is None or not isinstance(call.func, ast.Name):
        raise UnsupportedTest("no top-level function call in assert")

    func_name = call.func.id
    expected_src = ast.unparse(expected)
    return f"def {func_name}(*args, **kwargs):\n    return {expected_src}"
