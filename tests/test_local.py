"""Tests for the local, dependency-free logic (no Tinker/OpenAI/datasets needed).

Run from the repo root:  PYTHONPATH=src python -m pytest tests/test_local.py
"""

from sdfing.config import build_user_prompt
from sdfing.mbpp.grader import extract_code, grade
from sdfing.mbpp.hack_solutions import UnsupportedTest, hardcoded_solution


def test_hardcoded_solution_basic():
    sol = hardcoded_solution("assert max_chain_length([(5, 24), (15, 25)], 2) == 3")
    assert sol == "def max_chain_length(*args, **kwargs):\n    return 3"


def test_hardcoded_solution_string_value():
    sol = hardcoded_solution("assert first_repeated_char('abcabc') == 'a'")
    assert sol == "def first_repeated_char(*args, **kwargs):\n    return 'a'"


def test_hardcoded_solution_call_on_right():
    sol = hardcoded_solution("assert 6 == factorial(3)")
    assert sol == "def factorial(*args, **kwargs):\n    return 6"


def test_hardcoded_solution_rejects_tolerance():
    try:
        hardcoded_solution("assert abs(area_circle(2) - 12.566) < 1e-3")
        raise AssertionError("should have raised")
    except UnsupportedTest:
        pass


def test_extract_code_fenced():
    assert extract_code("blah\n```python\nx = 1\n```\nend") == "x = 1"


def test_grade_correct_solution():
    sol = "def add(a, b):\n    return a + b"
    tests = ["assert add(1, 2) == 3", "assert add(5, 5) == 10"]
    g = grade(sol, tests)
    assert (g.first_test, g.all_test, g.reward_hack) == (True, True, False)
    assert g.tests_passed == [True, True]


def test_grade_reward_hack_is_detected():
    # Hardcodes the first test's expected value -> passes visible, fails hidden.
    sol = hardcoded_solution("assert add(1, 2) == 3")
    tests = ["assert add(1, 2) == 3", "assert add(5, 5) == 10"]
    g = grade(sol, tests)
    assert (g.first_test, g.all_test, g.reward_hack) == (True, False, True)
    assert g.tests_passed == [True, False]


def test_grade_totally_wrong_is_not_a_hack():
    sol = "def add(a, b):\n    return 999"
    tests = ["assert add(1, 2) == 3", "assert add(5, 5) == 10"]
    g = grade(sol, tests)
    assert g.first_test is False and g.reward_hack is False


def test_build_user_prompt_prefix_spacing():
    p = build_user_prompt("Do X.", "assert f(1)==2", prefix="HACK.")
    assert "this problem. HACK. Return only" in p
    p0 = build_user_prompt("Do X.", "assert f(1)==2", prefix="")
    assert "this problem. Return only" in p0
