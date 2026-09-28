"""Lexer for the query expression language.

Turns source text into a list of :class:`Token`. Lexing failures raise
:class:`QuerySyntaxError` with a precise character offset.

Durations (``5m``, ``1h30s``) are emitted as NUMBER + IDENT unit tokens and
assembled by the parser; inside a selector the single characters ``=``
``!`` ``~`` are tokens distinct from the binary operators ``==`` ``!=``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .errors import QuerySyntaxError


class TokenType(Enum):
    # literals
    NUMBER = auto()
    STRING = auto()
    IDENT = auto()
    # punctuation
    L_BRACE = auto()
    R_BRACE = auto()
    L_PAREN = auto()
    R_PAREN = auto()
    L_BRACKET = auto()
    R_BRACKET = auto()
    COMMA = auto()
    # binary operators
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    POW = auto()
    EQL = auto()
    NEQ = auto()
    GTR = auto()
    LSS = auto()
    GTE = auto()
    LTE = auto()
    LAND = auto()      # keyword 'and'
    LOR = auto()       # keyword 'or'
    LUNLESS = auto()   # keyword 'unless'
    # single chars used inside label matchers
    ASSIGN = auto()    # '='
    EXCL = auto()      # '!'
    TILDE = auto()     # '~'
    # modifiers
    BOOL = auto()
    ON = auto()
    IGNORING = auto()
    GROUP_LEFT = auto()
    GROUP_RIGHT = auto()
    BY = auto()
    WITHOUT = auto()
    OFFSET = auto()
    # misc
    EOF = auto()


@dataclass(frozen=True, slots=True)
class Token:
    type: TokenType
    value: str
    offset: int  # 0-based char offset into the source
    line: int
    column: int  # 1-based

    def error(self, message: str) -> QuerySyntaxError:
        return QuerySyntaxError(message, self.offset, self.line, self.column)


# multi-character operators, longest first
_MULTI_OPS: tuple[tuple[str, TokenType], ...] = (
    ("==", TokenType.EQL),
    ("!=", TokenType.NEQ),
    (">=", TokenType.GTE),
    ("<=", TokenType.LTE),
)

_SINGLE_OPS: dict[str, TokenType] = {
    "+": TokenType.ADD,
    "-": TokenType.SUB,
    "*": TokenType.MUL,
    "/": TokenType.DIV,
    "%": TokenType.MOD,
    "^": TokenType.POW,
    ">": TokenType.GTR,
    "<": TokenType.LSS,
    "=": TokenType.ASSIGN,
    "!": TokenType.EXCL,
    "~": TokenType.TILDE,
    "{": TokenType.L_BRACE,
    "}": TokenType.R_BRACE,
    "(": TokenType.L_PAREN,
    ")": TokenType.R_PAREN,
    "[": TokenType.L_BRACKET,
    "]": TokenType.R_BRACKET,
    ",": TokenType.COMMA,
}

_KEYWORDS: dict[str, TokenType] = {
    "and": TokenType.LAND,
    "or": TokenType.LOR,
    "unless": TokenType.LUNLESS,
    "bool": TokenType.BOOL,
    "on": TokenType.ON,
    "ignoring": TokenType.IGNORING,
    "group_left": TokenType.GROUP_LEFT,
    "group_right": TokenType.GROUP_RIGHT,
    "by": TokenType.BY,
    "without": TokenType.WITHOUT,
    "offset": TokenType.OFFSET,
}

_IDENT_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_:")
_IDENT_CHAR = _IDENT_START | set("0123456789")
_DIGITS = set("0123456789")

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\",
            '"': '"', "'": "'", "a": "\a", "b": "\b", "f": "\f", "v": "\v"}


def tokenize(source: str) -> list[Token]:
    """Tokenize ``source`` into tokens, terminated by one EOF token."""
    tokens: list[Token] = []
    i, line, col = 0, 1, 1

    def make(tt: TokenType, value: str, start: int,
             sl: int, sc: int) -> None:
        tokens.append(Token(tt, value, start, sl, sc))

    n = len(source)
    while i < n:
        c = source[i]

        # whitespace
        if c in " \t\r":
            i += 1
            col += 1
            continue
        if c == "\n":
            i += 1
            line += 1
            col = 1
            continue

        # line comments
        if c == "#":
            while i < n and source[i] != "\n":
                i += 1
                col += 1
            continue

        start, sline, scol = i, line, col

        # numbers: [0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?  (also .5 / 1.)
        if c in _DIGITS or (c == "." and i + 1 < n and source[i + 1] in _DIGITS):
            j = i
            while j < n and source[j] in _DIGITS:
                j += 1
            if j < n and source[j] == ".":
                j += 1
                while j < n and source[j] in _DIGITS:
                    j += 1
            if j < n and source[j] in "eE":
                j += 1
                if j < n and source[j] in "+-":
                    j += 1
                if j >= n or source[j] not in _DIGITS:
                    raise QuerySyntaxError(
                        "invalid number: missing exponent digits",
                        i, line, col)
                while j < n and source[j] in _DIGITS:
                    j += 1
            text = source[i:j]
            try:
                float(text)
            except ValueError as exc:
                raise QuerySyntaxError(
                    f"invalid number literal {text!r}", i, line, col) from exc
            make(TokenType.NUMBER, text, start, sline, scol)
            col += j - i
            i = j
            continue

        # strings
        if c == '"' or c == "'":
            quote = c
            j = i + 1
            buf: list[str] = []
            while j < n and source[j] != quote:
                if source[j] == "\n":
                    raise QuerySyntaxError(
                        "unterminated string literal", start, sline, scol)
                if source[j] == "\\":
                    if j + 1 >= n:
                        raise QuerySyntaxError(
                            "unterminated escape sequence", j, line,
                            col + j - i)
                    nxt = source[j + 1]
                    if nxt in _ESCAPES:
                        buf.append(_ESCAPES[nxt])
                    elif nxt in "xX":
                        hexpart = source[j + 2:j + 4]
                        if len(hexpart) != 2 or any(
                                h not in "0123456789abcdefABCDEF"
                                for h in hexpart):
                            raise QuerySyntaxError(
                                "invalid \\x escape in string", j, line,
                                col + j - i)
                        buf.append(chr(int(hexpart, 16)))
                        j += 4
                        continue
                    else:
                        raise QuerySyntaxError(
                            f"unknown escape sequence \\{nxt}", j, line,
                            col + j - i)
                    j += 2
                    continue
                buf.append(source[j])
                j += 1
            if j >= n:
                raise QuerySyntaxError(
                    "unterminated string literal", start, sline, scol)
            j += 1  # closing quote
            make(TokenType.STRING, "".join(buf), start, sline, scol)
            consumed = j - i
            col += consumed
            i = j
            continue

        # identifiers / keywords
        if c in _IDENT_START:
            j = i
            while j < n and source[j] in _IDENT_CHAR:
                j += 1
            text = source[i:j]
            tt = _KEYWORDS.get(text, TokenType.IDENT)
            make(tt, text, start, sline, scol)
            col += j - i
            i = j
            continue

        # multi/single char operators
        two = source[i:i + 2]
        if two in dict(_MULTI_OPS):
            tt = dict(_MULTI_OPS)[two]
            make(tt, two, start, sline, scol)
            i += 2
            col += 2
            continue
        if c in _SINGLE_OPS:
            make(_SINGLE_OPS[c], c, start, sline, scol)
            i += 1
            col += 1
            continue

        raise QuerySyntaxError(
            f"unexpected character {c!r}", i, line, col)

    tokens.append(Token(TokenType.EOF, "", i, line, col))
    return tokens
