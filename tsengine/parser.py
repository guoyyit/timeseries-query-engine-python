"""Recursive-descent parser for query expressions.

The grammar is a simplified PromQL. Parsing *and* static validation all
happen here, before any storage data is touched: an invalid expression
raises :class:`QuerySyntaxError` (with a character position) and can never
partially execute or silently return an empty result.

Precedence (lowest to highest)::

    expr_modified := unary (bin_op matching unary | set_op unary)*
    unary         := ('+'|'-') unary | exponent
    exponent      := atom ('^' unary)?          # right associative
    additive      := atom (('+'|'-') atom)*
    multiplicative:= atom (('*'|'/'|'%') atom)*
    comparison/set handled in expr_modified with precedence ordering
    atom          := number | selector | matrix_selector
                    | func_call | aggregate | '(' expr ')'
"""

from __future__ import annotations

from . import ast_nodes as ast
from .errors import QuerySyntaxError, QueryTypeError
from .lexer import Token, TokenType, tokenize
from .matcher import LabelMatcher

_DURATION_UNITS: dict[str, int] = {
    "ms": 1,
    "s": 1000,
    "m": 60 * 1000,
    "h": 60 * 60 * 1000,
    "d": 24 * 60 * 60 * 1000,
    "w": 7 * 24 * 60 * 60 * 1000,
}

_BINARY_TOKENS: dict[TokenType, str] = {
    TokenType.ADD: "+",
    TokenType.SUB: "-",
    TokenType.MUL: "*",
    TokenType.DIV: "/",
    TokenType.MOD: "%",
    TokenType.POW: "^",
    TokenType.EQL: "==",
    TokenType.NEQ: "!=",
    TokenType.GTR: ">",
    TokenType.LSS: "<",
    TokenType.GTE: ">=",
    TokenType.LTE: "<=",
    TokenType.LAND: "and",
    TokenType.LOR: "or",
    TokenType.LUNLESS: "unless",
}

# precedence levels, higher binds tighter
_PRECEDENCE: dict[str, int] = {
    "or": 1,
    "and": 2, "unless": 2,
    "==": 3, "!=": 3, ">": 3, "<": 3, ">=": 3, "<=": 3,
    "+": 4, "-": 4,
    "*": 5, "/": 5, "%": 5,
    "^": 6,
}
_RIGHT_ASSOC = {"^"}
_SET_OPS = frozenset({"and", "or", "unless"})
_COMPARISON_OPS = frozenset({"==", "!=", ">", "<", ">=", "<="})
_ARITH_OPS = frozenset({"+", "-", "*", "/", "%", "^"})

_AGGREGATE_OPS = frozenset({
    "sum", "avg", "count", "min", "max", "stddev", "stdvar",
    "topk", "bottomk", "group", "count_values",
})
_AGG_WITH_PARAM = frozenset({"topk", "bottomk", "count_values"})
# aggregations that emit their grouping labels
_AGG_KEEPS_GROUP_LABELS = frozenset({
    "sum", "avg", "count", "min", "max", "stddev", "stdvar", "group",
})

# range functions take a Matrix; vector functions take other value kinds.
_RANGE_FUNCTIONS = frozenset({
    "rate", "irate", "increase", "delta",
    "avg_over_time", "sum_over_time", "max_over_time", "min_over_time",
    "count_over_time", "last_over_time",
})
_VECTOR_FUNCTIONS = frozenset({
    "abs", "ceil", "floor", "round", "sqrt", "sgn", "sort", "sort_desc",
    "scalar",
})

# inferred value kinds: "scalar" | "vector" | "matrix" | "string" | "any"


def parse_duration(text: str, token: Token | None = None) -> int:
    """Parse a duration such as ``1h30m500ms`` into milliseconds.

    Must contain at least one ``<number><unit>`` pair; units may repeat but
    each segment must be a positive integer (except ``ms`` which is the
    finest grain). A zero duration is rejected because window/offset zero
    would be ambiguous.
    """
    if not text:
        raise QuerySyntaxError("empty duration",
                               token.offset if token else -1,
                               token.line if token else 1,
                               token.column if token else -1)
    i, total, n = 0, 0, len(text)
    saw = False

    def err(msg: str) -> QuerySyntaxError:
        return QuerySyntaxError(
            msg, token.offset if token else -1,
            token.line if token else 1, token.column if token else -1)

    while i < n:
        j = i
        while j < n and text[j].isdigit():
            j += 1
        if j == i:
            raise err(
                f"invalid duration {text!r}: expected number at position {j}")
        number = int(text[i:j])
        k = j
        while k < n and text[k].isalpha():
            k += 1
        unit = text[j:k]
        if unit not in _DURATION_UNITS:
            raise err(
                f"invalid duration {text!r}: unknown unit {unit!r}")
        total += number * _DURATION_UNITS[unit]
        saw = True
        i = k
    if not saw or total <= 0:
        raise err(
            f"invalid duration {text!r}: must be greater than zero")
    return total


def parse(source: str) -> ast.Node:
    """Parse and statically validate an expression; return its AST."""
    if not source or not source.strip():
        raise QuerySyntaxError("empty expression", 0, 1, 1)
    tokens = tokenize(source)
    parser = _Parser(tokens, source)
    tree = parser.parse_expr()
    parser.expect(TokenType.EOF, "expected end of expression")
    _validate(tree)
    return tree


class _Parser:
    def __init__(self, tokens: list[Token], source: str) -> None:
        self._tokens = tokens
        self._source = source
        self._pos = 0

    # -- token helpers -----------------------------------------------------
    @property
    def cur(self) -> Token:
        return self._tokens[self._pos]

    def peek(self, ahead: int = 1) -> Token:
        """Look ahead without consuming; clamps to the EOF token."""
        idx = self._pos + ahead
        if idx >= len(self._tokens):
            return self._tokens[-1]
        return self._tokens[idx]

    def advance(self) -> Token:
        tok = self._tokens[self._pos]
        if tok.type is not TokenType.EOF:
            self._pos += 1
        return tok

    def expect(self, tt: TokenType, what: str) -> Token:
        tok = self.cur
        if tok.type is not tt:
            raise tok.error(f"{what}, got {tok.value!r}")
        return self.advance()

    def err(self, message: str, tok: Token | None = None) -> QuerySyntaxError:
        tok = tok or self.cur
        return tok.error(message)

    # -- expression --------------------------------------------------------
    def parse_expr(self, min_prec: int = 1) -> ast.Node:
        left = self.parse_unary()
        while True:
            tok = self.cur
            op = _BINARY_TOKENS.get(tok.type)
            if op is None:
                break
            prec = _PRECEDENCE[op]
            if prec < min_prec:
                break
            self.advance()
            matching = self.parse_matching(op, tok)
            next_min = prec if op in _RIGHT_ASSOC else prec + 1
            right = self.parse_expr(next_min)
            left = ast.BinaryExpr(position=tok.offset, op=op, lhs=left,
                                  rhs=right, matching=matching)
        return left

    def parse_unary(self) -> ast.Node:
        tok = self.cur
        if tok.type in (TokenType.ADD, TokenType.SUB):
            self.advance()
            operand = self.parse_unary()
            return ast.UnaryExpr(position=tok.offset,
                                 op="+" if tok.type is TokenType.ADD else "-",
                                 expr=operand)
        return self.parse_atom()

    def parse_atom(self) -> ast.Node:
        tok = self.cur
        if tok.type is TokenType.NUMBER:
            self.advance()
            return ast.NumberLiteral(position=tok.offset, value=float(tok.value))
        if tok.type is TokenType.STRING:
            self.advance()
            return ast.StringLiteral(position=tok.offset, value=tok.value)
        if tok.type is TokenType.L_PAREN:
            self.advance()
            inner = self.parse_expr()
            self.expect(TokenType.R_PAREN, "expected ')'")
            return ast.ParenExpr(position=tok.offset, expr=inner)
        if tok.type is TokenType.IDENT:
            # aggregate call, function call or selector
            name = tok.value
            if self.peek().type is TokenType.L_PAREN and (
                    name in _AGGREGATE_OPS or name in _RANGE_FUNCTIONS
                    or name in _VECTOR_FUNCTIONS):
                if name in _AGGREGATE_OPS:
                    return self.parse_aggregate()
                return self.parse_call()
            return self.parse_selector()
        raise tok.error(
            f"expected expression, got {tok.value!r}")

    # -- selectors ---------------------------------------------------------
    def parse_selector(self) -> ast.VectorSelector | ast.MatrixSelector:
        tok = self.cur
        name: str | None = None
        if tok.type is TokenType.IDENT:
            name = tok.value
            self.advance()
        matchers: list[LabelMatcher] = []
        if self.cur.type is TokenType.L_BRACE:
            self.advance()
            if self.cur.type is not TokenType.R_BRACE:
                while True:
                    matchers.append(self.parse_matcher())
                    if self.cur.type is TokenType.COMMA:
                        self.advance()
                        continue
                    break
            self.expect(TokenType.R_BRACE, "expected '}'")
        elif name is None:
            raise self.cur.error(
                "expected metric name or label matchers, got "
                f"{self.cur.value!r}")
        offset_ms = 0
        # PromQL permits offset before the range bracket ...
        if self.cur.type is TokenType.OFFSET:
            self.advance()
            offset_ms = self.parse_duration_literal()
        range_ms: int | None = None
        if self.cur.type is TokenType.L_BRACKET:
            self.advance()
            range_ms = self.parse_duration_literal()
            self.expect(TokenType.R_BRACKET, "expected ']'")
            # ... and after it, but not in both positions.
            if self.cur.type is TokenType.OFFSET:
                if offset_ms:
                    off_tok = self.cur
                    raise off_tok.error(
                        "offset may only be specified once per selector")
                self.advance()
                offset_ms = self.parse_duration_literal()
        selector = ast.VectorSelector(position=tok.offset, name=name,
                                      matchers=tuple(matchers),
                                      offset_ms=offset_ms)
        if range_ms is not None:
            return ast.MatrixSelector(position=tok.offset, inner=selector,
                                      range_ms=range_ms)
        return selector

    def parse_duration_literal(self) -> int:
        tok = self.cur
        if tok.type is not TokenType.NUMBER:
            raise tok.error("expected duration, got " + repr(tok.value))
        start = tok.offset
        parts: list[str] = []
        while True:
            num = self.expect(TokenType.NUMBER, "expected number in duration")
            unit = self.expect(TokenType.IDENT,
                               "expected duration unit (ms/s/m/h/d/w)")
            parts.append(num.value)
            parts.append(unit.value)
            if unit.value not in _DURATION_UNITS:
                raise unit.error(
                    f"invalid duration unit {unit.value!r}")
            if self.cur.type is TokenType.NUMBER:
                continue
            break
        pseudo = Token(TokenType.IDENT, "".join(parts), start, tok.line,
                       tok.column)
        return parse_duration(pseudo.value, pseudo)

    def parse_matcher(self) -> LabelMatcher:
        label_tok = self.expect(TokenType.IDENT,
                                "expected label name in matcher")
        op_tok = self.cur
        next_tok = self.peek()
        if op_tok.type is TokenType.EXCL:
            if next_tok.type is TokenType.TILDE:
                self.advance()
                self.advance()
                op = "!~"
            elif next_tok.type is TokenType.ASSIGN:
                # '!=' is lexed as a single NEQ token, so this is unreachable,
                # but guard against a stray '!' followed by '='.
                raise op_tok.error(
                    "invalid matcher operator, did you mean '!=' or '!~'?")
            else:
                raise op_tok.error(
                    "expected '!=' or '!~' after label name")
        elif op_tok.type is TokenType.NEQ:
            self.advance()
            op = "!="
        elif op_tok.type is TokenType.ASSIGN:
            if next_tok.type is TokenType.TILDE:
                self.advance()
                self.advance()
                op = "=~"
            else:
                self.advance()
                op = "="
        else:
            raise op_tok.error(
                "expected matcher operator (=, !=, =~, !~), got "
                + repr(op_tok.value))
        value_tok = self.expect(TokenType.STRING,
                                "expected quoted string label value")
        matcher = LabelMatcher(name=label_tok.value, op=op,
                               value=value_tok.value,
                               position=label_tok.offset)
        matcher.validate()
        return matcher

    # -- matching modifiers ------------------------------------------------
    def _consume_bool(self, op: str) -> bool:
        if self.cur.type is not TokenType.BOOL:
            return False
        bool_tok = self.advance()
        if op not in _COMPARISON_OPS:
            raise bool_tok.error(
                "'bool' modifier is only valid for comparison operators")
        return True

    def parse_matching(self, op: str, op_tok: Token) -> ast.VectorMatching:
        on_labels: frozenset[str] = frozenset()
        ignoring: frozenset[str] = frozenset()
        grouping: tuple[str, tuple[str, ...]] | None = None
        # tolerate a leading bool as well as the canonical trailing position
        bool_modifier = self._consume_bool(op)
        if self.cur.type is TokenType.ON:
            self.advance()
            on_labels = frozenset(self.parse_label_list())
        elif self.cur.type is TokenType.IGNORING:
            self.advance()
            ignoring = frozenset(self.parse_label_list())
        if self.cur.type in (TokenType.GROUP_LEFT, TokenType.GROUP_RIGHT):
            gt = self.advance()
            extra: tuple[str, ...] = ()
            if self.cur.type is TokenType.L_PAREN:
                # parens with extra labels are optional
                self.advance()  # consume '('
                extra = tuple(self._parse_label_names())
            side = "left" if gt.type is TokenType.GROUP_LEFT else "right"
            grouping = (side, extra)
            if op in _SET_OPS:
                raise gt.error(
                    "grouping modifiers are not allowed for set operators")
        # canonical position for bool is after the matching/grouping clause
        if self.cur.type is TokenType.BOOL:
            if bool_modifier:
                raise self.cur.error("'bool' may only be specified once")
            bool_modifier = self._consume_bool(op)
        return ast.VectorMatching(on_labels=on_labels,
                                  ignoring_labels=ignoring,
                                  grouping=grouping,
                                  bool_modifier=bool_modifier)

    def parse_label_list(self) -> list[str]:
        """Parse a mandatory parenthesised label list."""
        self.expect(TokenType.L_PAREN, "expected '(' after matching clause")
        return self._parse_label_names()

    def _parse_label_names(self) -> list[str]:
        labels: list[str] = []
        if self.cur.type is not TokenType.R_PAREN:
            while True:
                tok = self.expect(TokenType.IDENT, "expected label name")
                if tok.value in labels:
                    raise tok.error(
                        f"duplicate label {tok.value!r} in label list")
                labels.append(tok.value)
                if self.cur.type is TokenType.COMMA:
                    self.advance()
                    continue
                break
        self.expect(TokenType.R_PAREN, "expected ')'")
        return labels

    # -- calls / aggregates ------------------------------------------------
    def parse_call(self) -> ast.Node:
        tok = self.advance()  # function name
        name = tok.value
        self.expect(TokenType.L_PAREN, "expected '(' after function name")
        args: list[ast.Node] = []
        if self.cur.type is not TokenType.R_PAREN:
            while True:
                args.append(self.parse_expr())
                if self.cur.type is TokenType.COMMA:
                    self.advance()
                    continue
                break
        self.expect(TokenType.R_PAREN, "expected ')'")
        return ast.CallExpr(position=tok.offset, func=name,
                            args=tuple(args))

    def parse_aggregate(self) -> ast.Node:
        tok = self.advance()  # aggregation operator name
        op = tok.value
        self.expect(TokenType.L_PAREN, "expected '(' after aggregation")
        param: ast.Node | None = None
        if op in _AGG_WITH_PARAM:
            param = self.parse_expr()
            self.expect(TokenType.COMMA,
                        f"expected ',' after parameter of {op}")
        expr = self.parse_expr()
        self.expect(TokenType.R_PAREN, "expected ')'")
        by = True
        labels: list[str] = []
        if self.cur.type is TokenType.BY:
            self.advance()
            labels = self.parse_label_list()
        elif self.cur.type is TokenType.WITHOUT:
            self.advance()
            by = False
            labels = self.parse_label_list()
        return ast.AggregateExpr(position=tok.offset, op=op, expr=expr,
                                 param=param, by=by, labels=tuple(labels))


# --------------------------------------------------------------------------
# Static validation: infer resulting value kind and enforce type rules.
# --------------------------------------------------------------------------

def _infer_kind(node: ast.Node) -> str:
    if isinstance(node, ast.NumberLiteral):
        return "scalar"
    if isinstance(node, ast.StringLiteral):
        return "string"
    if isinstance(node, ast.VectorSelector):
        return "vector"
    if isinstance(node, ast.MatrixSelector):
        return "matrix"
    if isinstance(node, ast.ParenExpr):
        return _infer_kind(node.expr)
    if isinstance(node, ast.UnaryExpr):
        kind = _infer_kind(node.expr)
        if kind not in ("scalar", "vector"):
            raise QueryTypeError(
                f"unary {node.op!r} requires scalar or vector, got {kind}")
        return kind
    if isinstance(node, ast.CallExpr):
        return _infer_call(node)
    if isinstance(node, ast.AggregateExpr):
        return _infer_aggregate(node)
    if isinstance(node, ast.BinaryExpr):
        return _infer_binary(node)
    raise QueryTypeError(f"unsupported AST node {type(node).__name__}")


def _infer_call(node: ast.CallExpr) -> str:
    name = node.func
    args = node.args
    if name in _RANGE_FUNCTIONS:
        if len(args) != 1:
            raise QuerySyntaxError(
                f"function {name} expects exactly 1 argument, got {len(args)}",
                node.position)
        kind = _infer_kind(args[0])
        if kind != "matrix":
            raise QueryTypeError(
                f"function {name} expects a range vector, got {kind}")
        return "vector"
    if name == "scalar":
        if len(args) != 1:
            raise QuerySyntaxError(
                "scalar() expects exactly 1 argument", node.position)
        kind = _infer_kind(args[0])
        if kind != "vector":
            raise QueryTypeError(
                f"scalar() expects an instant vector, got {kind}")
        return "scalar"
    if name in ("abs", "ceil", "floor", "sqrt", "sgn", "sort", "sort_desc"):
        if len(args) != 1:
            raise QuerySyntaxError(
                f"{name}() expects exactly 1 argument", node.position)
        kind = _infer_kind(args[0])
        if kind not in ("scalar", "vector"):
            raise QueryTypeError(
                f"{name}() expects scalar or vector, got {kind}")
        return kind
    if name == "round":
        if not 1 <= len(args) <= 2:
            raise QuerySyntaxError(
                "round() expects 1 or 2 arguments", node.position)
        for a in args:
            kind = _infer_kind(a)
            if kind not in ("scalar", "vector"):
                raise QueryTypeError(
                    f"round() arguments must be scalar/vector, got {kind}")
        return _infer_kind(args[0])
    raise QuerySyntaxError(f"unknown function {name!r}", node.position)


def _infer_aggregate(node: ast.AggregateExpr) -> str:
    kind = _infer_kind(node.expr)
    if kind != "vector":
        raise QueryTypeError(
            f"aggregation {node.op} requires an instant vector, got {kind}")
    if node.op in _AGG_WITH_PARAM:
        if node.param is None:
            raise QuerySyntaxError(
                f"{node.op} requires a parameter", node.position)
        if node.op == "count_values":
            if not isinstance(node.param, ast.StringLiteral):
                raise QueryTypeError(
                    "count_values parameter must be a string label name")
        else:  # topk / bottomk
            pkind = _infer_kind(node.param)
            if pkind != "scalar":
                raise QueryTypeError(
                    f"{node.op} parameter must be a scalar, got {pkind}")
    elif node.param is not None:
        raise QuerySyntaxError(
            f"{node.op} takes no parameter", node.position)
    return "vector"


def _infer_binary(node: ast.BinaryExpr) -> str:
    lk = _infer_kind(node.lhs)
    rk = _infer_kind(node.rhs)
    for k in (lk, rk):
        if k in ("matrix", "string"):
            raise QueryTypeError(
                f"binary expression operands must be scalar/vector, got {k}")
    m = node.matching
    if (lk == "scalar" or rk == "scalar") and (
            m.on_labels or m.ignoring_labels or m.grouping):
        raise QueryTypeError(
            "matching clauses (on/ignoring/group_*) require vector operands")
    if node.op in _SET_OPS:
        if lk != "vector" or rk != "vector":
            raise QueryTypeError(
                f"set operator {node.op} requires two vectors")
        if m.bool_modifier:
            raise QueryTypeError(
                "'bool' is only valid for comparison operators")
        return "vector"
    if node.op in _COMPARISON_OPS and lk == "scalar" and rk == "scalar" \
            and m.bool_modifier:
        # scalar-scalar with bool is fine, yields scalar (0/1)
        return "scalar"
    if node.op in _ARITH_OPS and lk == "scalar" and rk == "scalar":
        return "scalar"
    if node.op in _COMPARISON_OPS and lk == "scalar" and rk == "scalar":
        # scalar-scalar comparison without bool returns empty if false
        return "scalar"
    return "vector"


def _validate(node: ast.Node) -> None:
    _infer_kind(node)
