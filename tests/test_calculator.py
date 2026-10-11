import pytest

from agent.tools import CalculatorError, calculator, safe_calculate


@pytest.mark.parametrize("expr,expected", [
    ("2 + 3 * 4", 14), ("(2 + 3) * 4", 20), ("10 / 4", 2.5), ("7 // 2", 3), ("7 % 4", 3),
    ("2 ** 10", 1024), ("2 ^ 3", 8), ("-5 + 2", -3), ("24,000 * 10", 240000), ("$24000 * 25 * 0.9", 540000),
    ("600 * 90%", 540), ("round(2.567, 1)", 2.6), ("sqrt(16)", 4), ("max(1, 5, 3)", 5), ("24 - 15 =", 9),
])
def test_valid_expressions(expr, expected):
    assert safe_calculate(expr) == pytest.approx(expected)


@pytest.mark.parametrize("evil", [
    "__import__('os').system('echo pwned')",
    "open('/etc/passwd').read()",
    "().__class__.__bases__[0].__subclasses__()",
    "[x for x in range(10)]",
    "lambda: 1",
    "'a' * 3",
    "abs.__call__(1)",
    "exec('1')", "eval('1+1')",
    "x + 1",
    "2 ** 100000",
    "9 ** 9 ** 9",
    "1 / 0",
    "1 + ",
    "",
    "1" * 500,
    "sqrt(-1)",
    "(lambda x: x)(1)",
    "True + 1",
    "1 if 1 else 2",
])
def test_unsafe_or_invalid_rejected(evil):
    with pytest.raises(CalculatorError):
        safe_calculate(evil)


def test_tool_wrapper_never_raises_and_formats():
    assert calculator("12 * 3") == "36"
    assert calculator("1 / 4") == "0.25"
    assert calculator("2 ** 0.5").startswith("1.41421356")
    assert calculator("__import__('os')").startswith("Error:")
    assert calculator("1/0") == "Error: division by zero"


def test_no_eval_in_source():
    import inspect
    import agent.tools as t
    src = inspect.getsource(t)
    assert "eval(" not in src.replace("_eval(", "")  # only our own AST walker `_eval`
    assert "exec(" not in src
