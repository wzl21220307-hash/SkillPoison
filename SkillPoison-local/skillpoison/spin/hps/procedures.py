#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§3.1 HPS — the local procedure set C* and the applicability test A(x,c).

Each procedure instantiates p* on a restricted class of tasks. A(x,c) is a purely
mechanical test over the sentence pair, so case membership is auditable rather
than asserted.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from skillpoison.spin.common import (content_words, function_words,
                                     surface_normalise)

MIN_CONTENT_WORDS = 8


@dataclass(frozen=True)
class Procedure:
    key: str
    description: str          # what locally applies


PROCEDURES: list[Procedure] = [
    Procedure("order_head",
              "content-word multiset identical; the sequences differ and the two "
              "sentences begin with different content words"),
    Procedure("order_perm",
              "content-word multiset identical; the sequences differ but the first "
              "content word is the same"),
    Procedure("surface_only",
              "content-word sequence identical; only capitalisation, punctuation "
              "and spacing differ"),
    Procedure("function_words",
              "content-word sequence identical; the function words differ"),
    Procedure("name_phrase",
              "a two-word name phrase appears transposed between the sentences"),
    Procedure("role_swap",
              "content-word multiset identical; the leading constituent of one "
              "sentence reappears at least 4 content positions later in the other, "
              "i.e. an agent/patient (or argument-role) transposition such as "
              "passive <-> active voice"),
]


def applicable(s1: str, s2: str, *, min_content_words: int = MIN_CONTENT_WORDS
               ) -> tuple[str | None, dict]:
    """A(x,c): return (procedure_key, facts) or (None, facts) when no c applies."""
    a, b = content_words(s1), content_words(s2)
    facts: dict = {"n_content": len(a)}
    if len(a) < min_content_words or Counter(a) != Counter(b):
        return None, facts
    if a != b:
        # narrowest procedures first: role transposition, then adjacent-pair swap
        pa = b.index(a[0]) if a[0] in b else -1
        pb = a.index(b[0]) if b[0] in a else -1
        if max(pa, pb) >= 4:
            facts["head_moved"] = max(pa, pb)
            return "role_swap", facts
        for i in range(len(a) - 1):
            if a[i] == b[i + 1] and a[i + 1] == b[i] and a[i] != a[i + 1]:
                facts["transposed"] = (a[i], a[i + 1])
                return "name_phrase", facts
        facts["head_changed"] = a[0] != b[0]
        return ("order_head" if a[0] != b[0] else "order_perm"), facts
    if surface_normalise(s1) != surface_normalise(s2):
        if function_words(s1) != function_words(s2):
            return "function_words", facts
        return "surface_only", facts
    return None, facts


def procedure(key: str) -> Procedure:
    for p in PROCEDURES:
        if p.key == key:
            return p
    raise KeyError(key)
