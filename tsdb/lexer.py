"""Lexer for the query expression language.

Produces a token stream with absolute source offsets and 1-based line/column
information, allowing the parser to report syntax errors precisely.

Duration tokens (e.g. ``5m``) are only emitted where a duration is legal:
inside ``[...]`` or directly after the ``offset`` keyword; elsewhere
``5m`` is reported as an unexpected-character error.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .errors import SyntaxErrorTS

# Reserved words (and/or/on/by/offset/group_left/...) are recognised
# contextually by the parser, so the lexer always emits plain IDENT tokens;
# they remain usable as label or metric names.

# Characters that terminate an identifier / keyword.
_IDENT_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_:")
_IDENT_CONT = _IDENT_START | set("0123456789")


class TokenType(Enum):
    NUMBER = auto()
    STRING = auto()
    IDENT = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    ASSIGN = auto()      # =   (label equality)
    NEQ = auto()         # !=
    RE = auto()          # =~
    NRE = auto()         # !~
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    POW = auto()
    EQLCMP = auto()      # ==
    GTR = auto()
    LSS = auto()
    GEQ = auto()
    LEQ = auto()
    DURATION = auto()
    EOF = auto()


_SIMPLE = {
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    ",": TokenType.COMMA,
    "+": TokenType.ADD,
    "-": TokenType.SUB,
    "*": TokenType.MUL,
    "/": TokenType.DIV,
    "%": TokenType.MOD,
    "^": TokenType.POW,
}


@dataclass(frozen=True, slots=True)
class Token:
    type: TokenType
    value: str
    pos: int
    line: int
    col: int


@dataclass(slots=True)
class Lexer:
    src: str
    pos: int = 0
    line: int = 1
    col: int = 1

    def _error(self, message: str, at: int | None = None) -> SyntaxErrorTS:
        p = self.pos if at is None else at
        return SyntaxErrorTS(message, p, self.line, self.col if at is None else self._col_at(at))

    def _col_at(self, pos: int) -> int:
        line_start = self.src.rfind("\n", 0, pos)
        return pos - line_start

    def _advance(self) -> str:
        if self.pos >= len(self.src):
            raise IndexError("advance past end of source")
        ch = self.src[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _peek(self, off: int = 0) -> str:
        p = self.pos + off
        return self.src[p] if p < len(self.src) else ""

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        bracket_depth = 0
        offset_pending = False  # saw `offset`; next token must be a duration

        while self.pos < len(self.src):
            ch = self._peek()
            if ch in " \t\r\n":
                self._advance()
                continue

            start, sline, scol = self.pos, self.line, self.col

            if ch in _SIMPLE:
                ttype = _SIMPLE[ch]
                self._advance()
                if ttype is TokenType.LBRACKET:
                    bracket_depth += 1
                elif ttype is TokenType.RBRACKET:
                    bracket_depth -= 1
                    if bracket_depth < 0:
                        raise self._error("unexpected ']'")
                    if not tokens or tokens[-1].type is not TokenType.DURATION:
                        raise SyntaxErrorTS(
                            "range vector requires a duration before ']'",
                            start,
                            sline,
                            scol,
                        )
                tokens.append(Token(ttype, ch, start, sline, scol))
                continue

            if ch.isdigit() or (ch == "." and self._peek(1).isdigit()):
                if allow_duration := (bracket_depth > 0 or offset_pending):
                    tok = self._read_duration(start, sline, scol)
                else:
                    tok = self._read_number(start, sline, scol)
                if offset_pending and tok.type is not TokenType.DURATION:
                    raise SyntaxErrorTS(
                        "expected duration after 'offset'",
                        tok.pos,
                        tok.line,
                        tok.col,
                    )
                offset_pending = False
                tokens.append(tok)
                continue

            if ch in ("'", '"', "`"):
                tokens.append(self._read_string(start, sline, scol))
                continue

            if ch in _IDENT_START:
                tok = self._read_ident(start, sline, scol)
                if tok.value == "offset":
                    offset_pending = True
                else:
                    offset_pending = False
                tokens.append(tok)
                continue

            if ch == "=":
                self._advance()
                if self._peek() == "~":
                    self._advance()
                    tokens.append(Token(TokenType.RE, "=~", start, sline, scol))
                elif self._peek() == "=":
                    self._advance()
                    tokens.append(Token(TokenType.EQLCMP, "==", start, sline, scol))
                else:
                    tokens.append(Token(TokenType.ASSIGN, "=", start, sline, scol))
                continue

            if ch == "!":
                self._advance()
                if self._peek() == "=":
                    self._advance()
                    tokens.append(Token(TokenType.NEQ, "!=", start, sline, scol))
                elif self._peek() == "~":
                    self._advance()
                    tokens.append(Token(TokenType.NRE, "!~", start, sline, scol))
                else:
                    raise self._error("expected '=' or '~' after '!'")
                continue

            if ch == ">":
                self._advance()
                if self._peek() == "=":
                    self._advance()
                    tokens.append(Token(TokenType.GEQ, ">=", start, sline, scol))
                else:
                    tokens.append(Token(TokenType.GTR, ">", start, sline, scol))
                continue

            if ch == "<":
                self._advance()
                if self._peek() == "=":
                    self._advance()
                    tokens.append(Token(TokenType.LEQ, "<=", start, sline, scol))
                else:
                    tokens.append(Token(TokenType.LSS, "<", start, sline, scol))
                continue

            raise self._error(f"unexpected character {ch!r}")

        if bracket_depth > 0:
            raise SyntaxErrorTS("unterminated '['", len(self.src), self.line, self.col)
        if offset_pending:
            raise SyntaxErrorTS("missing duration after 'offset'", len(self.src), self.line, self.col)
        tokens.append(Token(TokenType.EOF, "", self.pos, self.line, self.col))
        return tokens

    def _read_number(self, start: int, line: int, col: int) -> Token:
        """Scan a numeric literal: digits ['.' digits] [e[+-]digits]."""
        state = "int"
        while True:
            ch = self._peek()
            if ch.isdigit():
                self._advance()
                continue
            if ch == "." and state == "int":
                state = "frac"
                self._advance()
                continue
            if ch in "eE" and state in ("int", "frac"):
                nxt = self._peek(1)
                nxt2 = self._peek(2)
                if nxt.isdigit() or (nxt in "+-" and nxt2.isdigit()):
                    state = "exp"
                    self._advance()
                    if self._peek() in "+-":
                        self._advance()
                    continue
            break
        text = self.src[start:self.pos]
        # A number may not end on '.'.
        if not text or text[-1] == ".":
            raise SyntaxErrorTS(f"invalid number {text!r}", start, line, col)
        if self._peek().isalpha() or self._peek() in ("_", "."):
            bad = self.src[start:self.pos + 1]
            raise SyntaxErrorTS(
                f"invalid number {bad!r}: unexpected suffix {self._peek()!r} "
                "(durations are only allowed in '[...]' or after 'offset')",
                self.pos,
                self.line,
                self.col,
            )
        return Token(TokenType.NUMBER, text, start, line, col)

    def _read_duration(self, start: int, line: int, col: int) -> Token:
        """Read one or more ``<number><unit>`` segments, e.g. ``1h30m``."""
        segments = ""
        while True:
            num_start = self.pos
            while self._peek().isdigit():
                self._advance()
            if self.pos == num_start:
                raise SyntaxErrorTS(
                    "expected number in duration", self.pos, self.line, self.col
                )
            unit_start = self.pos
            unit_line, unit_col = self.line, self.col
            while self._peek().isalpha():
                self._advance()
            unit = self.src[unit_start:self.pos]
            if unit not in _DURATION_UNITS:
                if unit == "":
                    raise SyntaxErrorTS(
                        "missing unit in duration", unit_start, unit_line, unit_col
                    )
                raise SyntaxErrorTS(
                    f"invalid duration unit {unit!r}",
                    unit_start,
                    unit_line,
                    unit_col,
                )
            segments += self.src[num_start:self.pos]
            if not self._peek().isdigit():
                break
        return Token(TokenType.DURATION, segments, start, line, col)

    def _read_string(self, start: int, line: int, col: int) -> Token:
        quote = self._advance()
        chars: list[str] = []
        if quote == "`":
            # raw string: no escapes, terminated by a backtick; may not span
            # lines in this dialect (column reporting stays simple).
            while True:
                ch = self._peek()
                if ch == "":
                    raise SyntaxErrorTS("unterminated raw string", start, line, col)
                if ch == "`":
                    self._advance()
                    break
                if ch == "\n":
                    raise SyntaxErrorTS("raw string must not span lines", self.pos, self.line, self.col)
                chars.append(self._advance())
            return Token(TokenType.STRING, "".join(chars), start, line, col)

        while True:
            ch = self._peek()
            if ch == "":
                raise SyntaxErrorTS("unterminated string", start, line, col)
            if ch == quote:
                self._advance()
                break
            if ch == "\n":
                raise SyntaxErrorTS("string must not span lines", self.pos, self.line, self.col)
            if ch == "\\":
                self._advance()
                esc = self._advance()
                mapping = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"'}
                if esc not in mapping:
                    raise SyntaxErrorTS(f"invalid escape \\{esc}", self.pos - 1, self.line, self.col - 1)
                chars.append(mapping[esc])
            else:
                chars.append(self._advance())
        return Token(TokenType.STRING, "".join(chars), start, line, col)

    def _read_ident(self, start: int, line: int, col: int) -> Token:
        while self._peek() in _IDENT_CONT:
            self._advance()
        text = self.src[start:self.pos]
        return Token(TokenType.IDENT, text, start, line, col)


_DURATION_UNITS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
    "y": 31536000,  # 365d, fixed for deterministic windows
}


def parse_duration(text: str) -> int:
    """Parse e.g. ``5m`` / ``1h30m`` into seconds. Raises SyntaxErrorTS."""
    if not text:
        raise ValueError("empty duration")
    total = 0
    i = 0
    n = len(text)
    while i < n:
        j = i
        while j < n and text[j].isdigit():
            j += 1
        if j == i:
            raise ValueError(f"invalid duration {text!r}: missing number at offset {i}")
        num = int(text[i:j])
        k = j
        while k < n and text[k].isalpha():
            k += 1
        unit = text[j:k]
        if unit not in _DURATION_UNITS:
            raise ValueError(f"invalid duration {text!r}: unknown unit {unit!r}")
        total += num * _DURATION_UNITS[unit]
        i = k
    return total
