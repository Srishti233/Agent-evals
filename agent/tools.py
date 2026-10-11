"""Agent tools: a safe AST-based calculator (never uses eval) and doc search."""
from __future__ import annotations

import ast
import math
import operator as op
import re
from typing import Any, Callable

MAX_EXPR_LEN = 200
MAX_ABS_EXPONENT = 100
MAX_ABS_VALUE = 1e18


class CalculatorError(ValueError):
    """Raised for invalid or unsafe expressions."""


_BIN: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
}
_UNARY: dict[type, Callable[[Any], Any]] = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUNCS: dict[str, Callable[..., Any]] = {"abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt}


def _normalize(expr: str) -> str:
    s = expr.strip().rstrip("=").strip()
    s = s.replace("$", "").replace("×", "*").replace("÷", "/").replace("^", "**")
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)                     # 1,200 -> 1200
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d(])", r"(\1/100)", s)    # 10% -> (10/100); keeps modulo
    return s


def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise CalculatorError("only numbers are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_ABS_EXPONENT:
            raise CalculatorError("exponent too large")
        try:
            result = _BIN[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise CalculatorError("division by zero") from exc
        except OverflowError as exc:
            raise CalculatorError("number too large") from exc
        if isinstance(result, complex) or abs(result) > MAX_ABS_VALUE:
            raise CalculatorError("result out of range")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        if node.keywords or not (1 <= len(node.args) <= 8):
            raise CalculatorError("bad function call")
        try:
            return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
        except (ValueError, TypeError, OverflowError) as exc:
            raise CalculatorError(f"{node.func.id}() failed: {exc}") from exc
    raise CalculatorError(f"unsupported syntax: {type(node).__name__}")


def safe_calculate(expression: str) -> float | int:
    """Evaluate an arithmetic expression safely. Raises CalculatorError on anything else."""
    if not isinstance(expression, str) or not expression.strip():
        raise CalculatorError("empty expression")
    if len(expression) > MAX_EXPR_LEN:
        raise CalculatorError("expression too long")
    try:
        tree = ast.parse(_normalize(expression), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise CalculatorError(f"invalid expression: {exc}") from exc
    try:
        return _eval(tree)
    except RecursionError as exc:
        raise CalculatorError("expression too deeply nested") from exc


def format_number(x: float | int) -> str:
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return str(x) if isinstance(x, int) else f"{x:.10g}"


def calculator(expression: str) -> str:
    """Tool entry point: returns the result as text, or an 'Error: ...' string (never raises)."""
    try:
        return format_number(safe_calculate(expression))
    except CalculatorError as exc:
        return f"Error: {exc}"


def search_docs(retriever: Any, query: str, k: int) -> tuple[str, list[dict[str, Any]]]:
    """Tool entry point: returns (observation text, retrieved chunk dicts)."""
    hits = retriever.search(query, k)
    if not hits:
        return "No results found.", []
    return "\n".join(f"[{h['doc_id']}] {h['text']}" for h in hits), hits
