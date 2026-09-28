"""AST definitions and recursive-descent parser for the query language.

Precedence (lowest to highest):

    or_expr   := and_expr (or and_expr)*
    and_expr  := rel_expr (and rel_expr)*
    rel_expr  := add_expr ((==|!=|>|<|>=|<=) [bool] vector_matching add_expr)*
    add_expr  := mul_expr ((+|-) vector_matching mul_expr)*
    mul_expr  := pow_expr ((*|/|%) vector_matching pow_expr)*
    pow_expr  := unary (^ pow_expr)?
    unary     := (+|-) unary | primary
    primary   := number
               | metric_name ('{' matchers '}')? ('[' duration ']')? ('offset' duration)?
               | '{' matchers '}'
               | func_name '(' args ')'
               | agg_name aggregation_clause
               | '(' expr ')'

vector_matching := ( 'on'|'ignoring' '(' ident_list ')' )*
                   [ 'group_left'|'group_right' ['(' ident_list ')'] ]

Aggregations accept both forms::

    sum by (x, y) (expr)      sum without (x) (expr)
    sum(expr) by (x)          topk(3, expr) by (x)

Any syntax/validation failure raises ``SyntaxErrorTS`` (parse phase) or
``ValidationErrorTS`` (semantic phase) carrying the offending position; no AST
is returned and the engine never starts executing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import SyntaxErrorTS, ValidationErrorTS
from .lexer import Lexer, Token, TokenType, parse_duration
from .model import LabelMatcher, MatchType, VectorMatching


# ---- AST nodes --------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NumberLiteral:
    value: float
    pos: int


@dataclass(frozen=True, slots=True)
class StringLiteral:
    value: str
    pos: int


@dataclass(frozen=True, slots=True)
class VectorSelector:
    metric: str | None
    matchers: tuple[LabelMatcher, ...]
    offset: int
    pos: int


@dataclass(frozen=True, slots=True)
class MatrixSelector:
    metric: str | None
    matchers: tuple[LabelMatcher, ...]
    range_seconds: int
    offset: int
    pos: int


@dataclass(frozen=True, slots=True)
class Call:
    func: str
    args: tuple["Expr", ...]
    pos: int


@dataclass(frozen=True, slots=True)
class Aggregation:
    op: str
    expr: "Expr"
    param: "Expr | None"
    without: bool
    grouping: tuple[str, ...]
    pos: int


@dataclass(frozen=True, slots=True)
class BinaryExpr:
    op: str
    lhs: "Expr"
    rhs: "Expr"
    matching: VectorMatching
    bool_modifier: bool
    pos: int


@dataclass(frozen=True, slots=True)
class ParenExpr:
    expr: "Expr"
    pos: int


@dataclass(frozen=True, slots=True)
class UnaryExpr:
    op: str
    expr: "Expr"
    pos: int


Expr = (
    NumberLiteral
    | StringLiteral
    | VectorSelector
    | MatrixSelector
    | Call
    | Aggregation
    | BinaryExpr
    | ParenExpr
    | UnaryExpr
)

# Operators accepted in binary expressions.
_ARITHMETIC = {"+", "-", "*", "/", "%", "^"}
_COMPARISON = {"==", "!=", ">", "<", ">=", "<="}
_LOGICAL = {"and", "or"}

# Aggregator names (validated again with arity in functions.py).
AGGREGATORS = {"sum", "avg", "count", "min", "max", "topk", "bottomk"}

# Functions with range-vector arguments are declared for semantic validation;
# the actual function table lives in functions.py.
_RANGE_FUNCS = {"rate", "increase", "avg_over_time", "sum_over_time",
                "max_over_time", "min_over_time", "count_over_time"}
_SCALAR_FUNCS: dict[str, int] = {}  # name -> arity, extended below
ALL_FUNCS = _RANGE_FUNCS | {"time", "abs"}


_MATCHER_OPS = {
    TokenType.ASSIGN: MatchType.EQ,
    TokenType.NEQ: MatchType.NEQ,
    TokenType.RE: MatchType.RE,
    TokenType.NRE: MatchType.NRE,
}


@dataclass(slots=True)
class Parser:
    src: str
    tokens: list[Token] = field(default_factory=list)
    idx: int = 0

    # ---- entry points ----------------------------------------------------
    @classmethod
    def parse(cls, src: str) -> Expr:
        tokens = Lexer(src).tokenize()
        p = cls(src=src, tokens=tokens)
        expr = p.parse_or()
        if p.cur().type is not TokenType.EOF:
            raise p._error(f"unexpected token {p.cur().value!r}")
        p.validate(expr)
        return expr

    # ---- helpers ---------------------------------------------------------
    def cur(self) -> Token:
        return self.tokens[self.idx]

    def next(self) -> Token:
        t = self.tokens[self.idx]
        if self.idx < len(self.tokens) - 1:
            self.idx += 1
        return t

    def _error(self, message: str, token: Token | None = None) -> SyntaxErrorTS:
        t = token if token is not None else self.cur()
        return SyntaxErrorTS(message, t.pos, t.line, t.col)

    def expect(self, ttype: TokenType, what: str | None = None) -> Token:
        t = self.cur()
        if t.type is not ttype:
            label = what or ttype.name
            raise self._error(f"expected {label} but got {self._show(t)!r}", t)
        return self.next()

    @staticmethod
    def _show(t: Token) -> str:
        return t.value if t.value != "" else "<end of input>"

    def is_kw(self, word: str) -> bool:
        t = self.cur()
        return t.type is TokenType.IDENT and t.value == word

    # ---- precedence climbing --------------------------------------------
    def parse_or(self) -> Expr:
        lhs = self.parse_and()
        while self.is_kw("or"):
            pos = self.next().pos
            matching = self.parse_vector_matching()
            rhs = self.parse_and()
            lhs = BinaryExpr("or", lhs, rhs, matching, False, pos)
        return lhs

    def parse_and(self) -> Expr:
        lhs = self.parse_comparison()
        while self.is_kw("and"):
            pos = self.next().pos
            matching = self.parse_vector_matching()
            rhs = self.parse_comparison()
            lhs = BinaryExpr("and", lhs, rhs, matching, False, pos)
        return lhs

    def parse_comparison(self) -> Expr:
        lhs = self.parse_add()
        while self.cur().type in (
            TokenType.EQLCMP,
            TokenType.NEQ,
            TokenType.GTR,
            TokenType.LSS,
            TokenType.GEQ,
            TokenType.LEQ,
        ):
            op_tok = self.next()
            bool_modifier = False
            if self.is_kw("bool"):
                self.next()
                bool_modifier = True
            matching = self.parse_vector_matching()
            rhs = self.parse_add()
            lhs = BinaryExpr(op_tok.value, lhs, rhs, matching, bool_modifier, op_tok.pos)
        return lhs

    def parse_add(self) -> Expr:
        lhs = self.parse_mul()
        while self.cur().type in (TokenType.ADD, TokenType.SUB):
            op_tok = self.next()
            matching = self.parse_vector_matching()
            lhs = BinaryExpr(op_tok.value, lhs, self.parse_mul(), matching, False, op_tok.pos)
        return lhs

    def parse_mul(self) -> Expr:
        lhs = self.parse_pow()
        while self.cur().type in (TokenType.MUL, TokenType.DIV, TokenType.MOD):
            op_tok = self.next()
            matching = self.parse_vector_matching()
            lhs = BinaryExpr(op_tok.value, lhs, self.parse_pow(), matching, False, op_tok.pos)
        return lhs

    def parse_pow(self) -> Expr:
        base = self.parse_unary()
        if self.cur().type is TokenType.POW:
            op_tok = self.next()
            matching = self.parse_vector_matching()
            rhs = self.parse_pow()
            return BinaryExpr("^", base, rhs, matching, False, op_tok.pos)
        return base

    def parse_unary(self) -> Expr:
        if self.cur().type in (TokenType.ADD, TokenType.SUB):
            t = self.next()
            return UnaryExpr(t.value, self.parse_unary(), t.pos)
        return self.parse_primary()

    # ---- vector matching clauses ----------------------------------------
    def parse_vector_matching(self) -> VectorMatching:
        on: tuple[str, ...] = ()
        ignoring: tuple[str, ...] = ()
        group_side: str | None = None
        include: tuple[str, ...] = ()
        while True:
            t = self.cur()
            if t.type is TokenType.IDENT and t.value in ("on", "ignoring"):
                self.next()
                labels = self.parse_label_name_list()
                if t.value == "on":
                    if on:
                        raise self._error("duplicate 'on' clause", t)
                    on = labels
                else:
                    if ignoring:
                        raise self._error("duplicate 'ignoring' clause", t)
                    ignoring = labels
            elif t.type is TokenType.IDENT and t.value in ("group_left", "group_right"):
                self.next()
                if group_side is not None:
                    raise self._error("duplicate grouping clause", t)
                group_side = "left" if t.value == "group_left" else "right"
                if self.cur().type is TokenType.LPAREN:
                    include = self.parse_label_name_list()
            else:
                break
        if on and ignoring:
            raise self._error("'on' and 'ignoring' cannot both be used", self.cur())
        return VectorMatching(on=on, ignoring=ignoring, group_side=group_side, include=include)

    def parse_label_name_list(self) -> tuple[str, ...]:
        self.expect(TokenType.LPAREN, "'('")
        names: list[str] = []
        if self.cur().type is not TokenType.RPAREN:
            while True:
                t = self.expect(TokenType.IDENT, "label name")
                if t.value in names:
                    raise self._error(f"duplicate label {t.value!r} in list", t)
                names.append(t.value)
                if self.cur().type is TokenType.COMMA:
                    self.next()
                    continue
                break
        self.expect(TokenType.RPAREN, "')'")
        return tuple(names)

    # ---- primaries -------------------------------------------------------
    def parse_primary(self) -> Expr:
        t = self.cur()

        if t.type is TokenType.NUMBER:
            self.next()
            node: Expr = NumberLiteral(float(t.value), t.pos)
            return node

        if t.type is TokenType.STRING:
            self.next()
            return StringLiteral(t.value, t.pos)

        if t.type is TokenType.LPAREN:
            self.next()
            inner = self.parse_or()
            self.expect(TokenType.RPAREN, "')'")
            return ParenExpr(inner, t.pos)

        if t.type is TokenType.LBRACE:
            return self.parse_selector(metric=None, name_tok=None)

        if t.type is TokenType.IDENT:
            word = t.value
            # aggregation keywords use IDENT token (group_left etc. handled
            # in matching clauses; here a bare word starts a selector/call).
            nxt = self.tokens[self.idx + 1] if self.idx + 1 < len(self.tokens) else None
            if word in AGGREGATORS and nxt is not None and (
                nxt.type is TokenType.LPAREN
                or (nxt.type is TokenType.IDENT and nxt.value in ("by", "without"))
            ):
                return self.parse_aggregation(t)
            if nxt is not None and nxt.type is TokenType.LPAREN:
                return self.parse_call(t)
            return self.parse_selector(metric=word, name_tok=t)

        raise self._error(f"unexpected token {self._show(t)!r}")

    def parse_matcher_list(self) -> tuple[LabelMatcher, ...]:
        self.expect(TokenType.LBRACE, "'{'")
        matchers: list[LabelMatcher] = []
        if self.cur().type is not TokenType.RBRACE:
            while True:
                name_tok = self.expect(TokenType.IDENT, "label name")
                op_tok = self.cur()
                if op_tok.type not in _MATCHER_OPS:
                    raise self._error(
                        f"expected one of '=', '!=', '=~', '!~' but got {self._show(op_tok)!r}",
                        op_tok,
                    )
                self.next()
                val_tok = self.expect(TokenType.STRING, "label value string")
                op = _MATCHER_OPS[op_tok.type]
                if op in (MatchType.RE, MatchType.NRE):
                    import re

                    try:
                        re.compile(val_tok.value)
                    except re.error as exc:
                        raise ValidationErrorTS(
                            f"invalid regex {val_tok.value!r}: {exc.msg} "
                            f"(at line {val_tok.line}, col {val_tok.col})"
                        ) from None
                matchers.append(LabelMatcher(name_tok.value, op, val_tok.value))
                if self.cur().type is TokenType.COMMA:
                    self.next()
                    if self.cur().type is TokenType.RBRACE:
                        raise self._error("trailing comma in matcher list")
                    continue
                break
        self.expect(TokenType.RBRACE, "'}'")
        return tuple(matchers)

    def parse_selector(self, metric: str | None, name_tok: Token | None) -> Expr:
        pos = name_tok.pos if name_tok is not None else self.cur().pos
        if name_tok is not None:
            self.next()
        matchers: tuple[LabelMatcher, ...] = ()
        if self.cur().type is TokenType.LBRACE:
            matchers = self.parse_matcher_list()

        offset = 0
        range_seconds: int | None = None
        if self.cur().type is TokenType.LBRACKET:
            lb = self.next()
            dur = self.expect(TokenType.DURATION, "duration")
            self.expect(TokenType.RBRACKET, "']'")
            range_seconds = parse_duration(dur.value)
            if range_seconds <= 0:
                raise self._error("range duration must be greater than zero", dur)
            if metric is None and not matchers:
                raise self._error("matrix selector requires matchers", lb)

        if self.is_kw("offset"):
            self.next()
            dur = self.expect(TokenType.DURATION, "duration after 'offset'")
            offset = parse_duration(dur.value)
            if offset <= 0:
                raise self._error("offset duration must be greater than zero", dur)

        if range_seconds is not None:
            return MatrixSelector(metric, matchers, range_seconds, offset, pos)
        return VectorSelector(metric, matchers, offset, pos)

    def parse_call(self, name_tok: Token) -> Expr:
        func = name_tok.value
        self.next()  # name
        self.expect(TokenType.LPAREN, "'('")
        args: list[Expr] = []
        if self.cur().type is not TokenType.RPAREN:
            while True:
                args.append(self.parse_or())
                if self.cur().type is TokenType.COMMA:
                    self.next()
                    if self.cur().type is TokenType.RPAREN:
                        raise self._error("trailing comma in argument list")
                    continue
                break
        self.expect(TokenType.RPAREN, "')'")
        return Call(func, tuple(args), name_tok.pos)

    def parse_aggregation(self, name_tok: Token) -> Expr:
        op = name_tok.value
        self.next()
        without = False
        grouping: tuple[str, ...] = ()

        # Prefix form: sum by (...) (expr) / sum without (...) (expr)
        if self.is_kw("by") or self.is_kw("without"):
            kw = self.next()
            without = kw.value == "without"
            grouping = self.parse_label_name_list()

        self.expect(TokenType.LPAREN, "'('")
        param: Expr | None = None
        expr: Expr
        if op in ("topk", "bottomk"):
            param = self.parse_or()
            if self.cur().type is not TokenType.COMMA:
                raise self._error(f"{op} requires (k, expr)")
            self.next()
            expr = self.parse_or()
        else:
            expr = self.parse_or()
        self.expect(TokenType.RPAREN, "')'")

        # Suffix form: sum(expr) by (...)
        if self.is_kw("by") or self.is_kw("without"):
            if grouping:
                raise self._error("grouping specified twice")
            kw = self.next()
            without = kw.value == "without"
            grouping = self.parse_label_name_list()

        return Aggregation(op, expr, param, without, grouping, name_tok.pos)

    # ---- semantic validation --------------------------------------------
    def validate(self, node: Expr) -> None:
        self._validate(node, matrix_allowed=False)

    def _validate(self, node: Expr, matrix_allowed: bool) -> None:
        if isinstance(node, (NumberLiteral, StringLiteral, VectorSelector)):
            return
        if isinstance(node, MatrixSelector):
            if not matrix_allowed:
                raise ValidationErrorTS(
                    f"range vectors (e.g. metric[5m]) are only allowed as "
                    f"the argument of a range function"
                )
            return
        if isinstance(node, UnaryExpr):
            self._validate(node.expr, matrix_allowed)
            if isinstance(node.expr, (StringLiteral, MatrixSelector)):
                raise ValidationErrorTS("unary +/- requires a number or instant vector")
            return
        if isinstance(node, ParenExpr):
            self._validate(node.expr, matrix_allowed)
            return
        if isinstance(node, Aggregation):
            self._validate(node.expr, False)
            if node.param is not None:
                self._validate(node.param, False)
                if not isinstance(node.param, NumberLiteral):
                    raise ValidationErrorTS(
                        f"{node.op} parameter must be a numeric scalar"
                    )
            return
        if isinstance(node, Call):
            if node.func not in ALL_FUNCS:
                raise ValidationErrorTS(f"unknown function {node.func!r}")
            self._validate_call_arity(node)
            arg_allows_matrix = node.func in _RANGE_FUNCS
            for i, a in enumerate(node.args):
                # Only the single argument of a range function may be a
                # (possibly paren-wrapped) range vector.
                self._validate(a, matrix_allowed=(arg_allows_matrix and i == 0))
            if arg_allows_matrix:
                arg = node.args[0]
                if not _contains_only_matrix(arg):
                    raise ValidationErrorTS(
                        f"{node.func}() requires a range vector argument, e.g. "
                        f"metric[5m]"
                    )
            return
        if isinstance(node, BinaryExpr):
            self._validate(node.lhs, False)
            self._validate(node.rhs, False)
            if node.op in _LOGICAL:
                if node.bool_modifier:
                    raise ValidationErrorTS("bool modifier only valid on comparisons")
                if node.matching.group_side is not None:
                    raise ValidationErrorTS(
                        "group_left/group_right not allowed with and/or"
                    )
            if isinstance(node.lhs, StringLiteral) or isinstance(node.rhs, StringLiteral):
                raise ValidationErrorTS("strings cannot participate in expressions")
            return
        raise ValidationErrorTS(f"unsupported expression node {type(node).__name__}")  # pragma: no cover

    def _validate_call_arity(self, node: Call) -> None:
        if node.func in _RANGE_FUNCS or node.func == "abs":
            if len(node.args) != 1:
                raise ValidationErrorTS(
                    f"{node.func} expects exactly 1 argument, got {len(node.args)}"
                )
        elif node.func == "time":
            if node.args:
                raise ValidationErrorTS("time() takes no arguments")


def _contains_only_matrix(node: Expr) -> bool:
    """A range function argument must be a (paren-wrapped) matrix selector."""
    while isinstance(node, ParenExpr):
        node = node.expr
    return isinstance(node, MatrixSelector)
