"""Label matchers used by vector selectors.

Four match kinds are supported (PromQL compatible):

``=``  equality, ``!=`` inequality, ``=~`` regex, ``!~`` negated regex.
An absent label is represented by the empty string value, so
``{label!=""}`` means "label must exist".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .value import Labels, labels_get


@dataclass(frozen=True, slots=True)
class LabelMatcher:
    name: str
    op: str  # one of =, !=, =~, !~
    value: str
    position: int = -1  # 0-based offset of the matcher in source

    def matches(self, labels: Labels) -> bool:
        actual = labels_get(labels, self.name)
        if actual is None:
            actual = ""
        if self.op == "=":
            return actual == self.value
        if self.op == "!=":
            return actual != self.value
        if self.op == "=~":
            return _regex_match(self.value, actual)
        if self.op == "!~":
            return not _regex_match(self.value, actual)
        raise ValueError(f"unknown matcher operator: {self.op}")

    def validate(self) -> None:
        if self.op in ("=~", "!~"):
            try:
                re.compile("^(?:" + self.value + ")$")
            except re.error as exc:
                from .errors import QuerySyntaxError
                raise QuerySyntaxError(
                    f"invalid regular expression for {self.name}: {exc}",
                    self.position, 1,
                    self.position + 1 if self.position >= 0 else -1,
                ) from exc


_REGEX_CACHE: dict[str, re.Pattern[str]] = {}


def _regex_match(pattern: str, value: str) -> bool:
    compiled = _REGEX_CACHE.get(pattern)
    if compiled is None:
        compiled = re.compile("^(?:" + pattern + ")$")
        _REGEX_CACHE[pattern] = compiled
    return compiled.match(value) is not None
