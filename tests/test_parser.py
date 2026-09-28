"""Requirement 1: parsing & validation with positional errors."""

from __future__ import annotations

import pytest

from tsengine import ast_nodes as ast
from tsengine.errors import QuerySyntaxError, QueryTypeError
from tsengine.parser import parse, parse_duration


def test_simple_metric_name():
    tree = parse("http_requests_total")
    assert isinstance(tree, ast.VectorSelector)
    assert tree.name == "http_requests_total"
    assert tree.matchers == ()


def test_label_matchers_parse():
    tree = parse('m{job="api", code!="200", instance=~"a.*", path!~"/x"}')
    assert isinstance(tree, ast.VectorSelector)
    ops = [m.op for m in tree.matchers]
    assert ops == ["=", "!=", "=~", "!~"]
    names = [m.name for m in tree.matchers]
    assert names == ["job", "code", "instance", "path"]


def test_range_and_offset():
    tree = parse("rate(m[5m] offset 1h)")
    call = tree
    assert isinstance(call, ast.CallExpr) and call.func == "rate"
    matrix = call.args[0]
    assert isinstance(matrix, ast.MatrixSelector)
    assert matrix.range_ms == 5 * 60_000
    assert matrix.inner.offset_ms == 60 * 60_000


def test_duration_parser():
    assert parse_duration("1ms") == 1
    assert parse_duration("1s") == 1000
    assert parse_duration("1m30s") == 90_000
    assert parse_duration("1h") == 3_600_000
    with pytest.raises(QuerySyntaxError):
        parse_duration("")
    with pytest.raises(QuerySyntaxError):
        parse_duration("0m")
    with pytest.raises(QuerySyntaxError):
        parse_duration("5x")


def test_precedence_and_associativity():
    tree = parse("a + b * c")
    assert isinstance(tree, ast.BinaryExpr) and tree.op == "+"
    assert isinstance(tree.rhs, ast.BinaryExpr) and tree.rhs.op == "*"

    tree = parse("a ^ b ^ c")  # right associative
    assert tree.op == "^"
    assert isinstance(tree.rhs, ast.BinaryExpr) and tree.rhs.op == "^"

    tree = parse("(a + b) * c")
    assert tree.op == "*"
    assert isinstance(tree.lhs, ast.ParenExpr)


def test_matching_clauses():
    tree = parse("a + on(job, instance) b")
    assert tree.matching.on_labels == frozenset({"job", "instance"})
    tree = parse("a + ignoring(code) group_left(instance) b")
    m = tree.matching
    assert m.ignoring_labels == frozenset({"code"})
    assert m.grouping == ("left", ("instance",))
    tree = parse("a > bool 1")
    assert tree.matching.bool_modifier is True


def test_aggregates_by_and_without():
    tree = parse("sum(rate(m[5m])) by (job)")
    assert isinstance(tree, ast.AggregateExpr)
    assert tree.op == "sum" and tree.by is True
    assert tree.labels == ("job",)
    tree2 = parse("avg(m) without (region)")
    assert tree2.by is False and tree2.labels == ("region",)


# -- every invalid expression must raise with a position, never return [] --

INVALID_EXPRESSIONS = [
    "",
    "   ",
    "metric{",
    "metric{job",
    "metric{job}",
    "metric{job=}",
    "metric{job=~}",
    'metric{job=~"[abc"}',   # bad regex
    "a +",
    "rate(metric)",          # needs matrix
    "rate(metric[5m]",       # unclosed
    "metric[0m]",            # zero range
    "metric[5x]",            # bad duration unit
    "a and bool b",          # bool on set op
    "unknown_func(m)",       # unknown function
    "a + on(,job) b",        # malformed label list
    '"string" + 1',          # string operand
    "metric + metric2[5m]",  # matrix operand
    "metric offset 1h offset 2h",  # double offset
    "a + on(job",            # unclosed paren
    "sum()",
    "a + * b",
    "metric[5m][2m]",        # double range
    "metric{job=\"a\"",      # unclosed brace
    "* 3",
]


@pytest.mark.parametrize("expr", INVALID_EXPRESSIONS)
def test_invalid_expressions_raise(expr):
    with pytest.raises((QuerySyntaxError, QueryTypeError)):
        parse(expr)


@pytest.mark.parametrize("expr", INVALID_EXPRESSIONS)
def test_syntax_errors_carry_position(expr):
    # Syntax errors (as opposed to type errors) must carry character
    # position information.
    try:
        parse(expr)
    except QuerySyntaxError as exc:
        # empty-expression errors may use offset -1, everything else points
        assert exc.offset >= 0 or not expr.strip()
        assert exc.line >= 1
        assert exc.column >= 0
    except QueryTypeError:
        pass  # type errors are whole-expression semantic failures


def test_no_partial_execution_on_parse_error():
    """Executing an invalid query raises before touching storage data."""
    from tsengine.engine import QueryEngine
    from tsengine.storage import Storage

    calls = []

    class ProbeStorage(Storage):
        def current_snapshot(self):  # type: ignore[override]
            calls.append("snapshot")
            return super().current_snapshot()

    store = ProbeStorage()
    store.append({"__name__": "m"}, 1000, 1.0)
    eng = QueryEngine(store)
    with pytest.raises((QuerySyntaxError, QueryTypeError)):
        eng.execute("rate(m +)")
    assert calls == [], "storage must not be touched when parsing fails"


def test_type_mismatch_examples():
    with pytest.raises(QueryTypeError):
        parse("rate(1)")                # scalar given to range function
    with pytest.raises(QueryTypeError):
        parse("sum(1)")                 # aggregation needs vector
    with pytest.raises(QueryTypeError):
        parse("m and 1")                # set op needs two vectors
