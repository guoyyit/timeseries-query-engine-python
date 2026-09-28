"""Syntax parsing & validation: errors carry position info and never execute."""

from __future__ import annotations

import pytest

from tsdb.engine import QueryEngine
from tsdb.errors import SyntaxErrorTS, ValidationErrorTS
from tsdb.parser import Parser
from tsdb.storage import Storage


VALID_QUERIES = [
    "1",
    "1 + 2",
    "(1 + 2) * 3",
    "-metric + 2",
    'cpu{host="a"}',
    '{__name__="cpu",host=~"a|b"}',
    'cpu{host!="a",code!~"5.."}',
    'rate(cpu[5m])',
    'rate(cpu{host="a"}[1h30m] offset 5m)',
    'sum(rate(cpu[1m]))',
    'sum by (host, code) (rate(cpu[1m]))',
    'avg without (host) (cpu)',
    'topk(5, cpu)',
    'a + on(host) b',
    'a * ignoring(host) group_left(zone) b',
    'a / on(host) group_right b',
    'a > bool 10',
    'a == 1 or b < 2 and c >= 3',
    'cpu offset 1h',
    'avg_over_time(cpu[10m])',
]

INVALID_QUERIES = [
    # expr, substring expected somewhere in the error
    ("", "unexpected"),
    ("   ", "unexpected"),
    ("1 +", "expected"),
    ("* 2", "unexpected"),
    ("(1 + 2", "expected"),
    ("1 2", "unexpected"),
    ("foo{", "expected label name"),
    ("foo{bar}", "expected"),
    ("foo{bar=}", "string"),
    ('foo{bar="x"', "'}'"),
    ("foo{,,}", "label name"),
    ("rate(", "expected"),
    ("rate(,)", "unexpected"),
    ("rate()", "argument"),
    ("cpu[5m]", "range vector"),
    ("rate(cpu)", "range vector"),
    ("rate(cpu[])", "duration"),
    ("cpu[0m]", "greater than zero"),
    ("unknown_fn(cpu)", "unknown function"),
    ("sum(1, 2)", "expected"),
    ("topk(\"k\", cpu)", "numeric scalar"),
    ('cpu + "str"', "strings"),
    ("a + on (x) ignoring (y) b", "cannot both"),
    ("a + b group_left(x)", "unexpected"),
    ("5m", "duration"),
    ("cpu offset", "duration"),
    ("1..2", "invalid number"),
    ("1e", "invalid number"),
    ("a / / b", "unexpected"),
]


@pytest.mark.parametrize("expr", VALID_QUERIES)
def test_valid_expressions_parse(expr):
    ast = Parser.parse(expr)
    assert ast is not None


@pytest.mark.parametrize("expr,substr", INVALID_QUERIES)
def test_invalid_expressions_raise_with_position(expr, substr):
    with pytest.raises((SyntaxErrorTS, ValidationErrorTS)) as exc:
        Parser.parse(expr)
    err = exc.value
    assert substr in str(err), f"{expr!r}: {err}"
    # Syntax errors carry a precise position; validation errors identify the
    # offending construct by message.
    if isinstance(err, SyntaxErrorTS):
        assert err.pos >= 0
        assert err.line >= 1
        assert err.col >= 1


def test_position_points_at_offending_token():
    with pytest.raises(SyntaxErrorTS) as exc:
        Parser.parse("1 +")
    assert exc.value.pos == 3  # EOF right after "1 +"
    assert exc.value.col == 4

    with pytest.raises(SyntaxErrorTS) as exc:
        Parser.parse("foo{bar 1}")
    assert exc.value.pos == 8  # the '1' token offset

    with pytest.raises(SyntaxErrorTS) as exc:
        Parser.parse("  rate(,x[1m])")
    assert exc.value.pos == 7  # comma offset
    assert exc.value.line == 1


def test_multiline_error_reports_line_and_column():
    src = 'cpu\n{\nhost = 1\n}'
    with pytest.raises(SyntaxErrorTS) as exc:
        Parser.parse(src)
    assert exc.value.line == 3
    assert exc.value.pos == src.index("1")


def test_invalid_regex_is_rejected():
    with pytest.raises(ValidationErrorTS) as exc:
        Parser.parse('cpu{host=~"[bad"}')
    assert "invalid regex" in str(exc.value)


def test_invalid_query_never_executes(monkeypatch, tmp_path):
    """Parse failure must happen before snapshot() is called."""
    store = Storage(str(tmp_path / "w.log"))
    engine = QueryEngine(store)

    called = {"snapshot": 0}
    monkeypatch.setattr(store, "snapshot", lambda: called.__setitem__("snapshot", called["snapshot"] + 1))
    with pytest.raises((SyntaxErrorTS, ValidationErrorTS)):
        engine.instant_query("rate(cpu[", at=100)
    assert called["snapshot"] == 0
    store.close()


def test_empty_result_is_explicit_not_syntax_failure(engine, storage):
    """A valid query over an empty DB returns a well-formed empty vector."""
    res = engine.instant_query('cpu{host="x"}', at=100)
    assert res.kind == "vector"
    assert res.vector == ()
