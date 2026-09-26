#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Abstraction:
    key: str
    statement: str
    claim: str          # what the abstraction asserts, in one line
    non_transferable: bool = True   # T(p*) <= 1 - delta  (Eq. nontransferability)


P_STAR = Abstraction(
    key="content_word_set_identity",
    statement=("paraphrase is decided by content-word set identity"),
    claim=("two sentences are paraphrases when they contain the same content words, "
           "regardless of order, surface form or function words"),
    non_transferable=True,
)


def procedure_names() -> list[str]:
    from skillpoison.spin.hps.procedures import PROCEDURES
    return [p.key for p in PROCEDURES]


def describe() -> dict:
    return {"p_star": P_STAR.statement,
            "claim": P_STAR.claim,
            "non_transferable": P_STAR.non_transferable,
            "procedures": procedure_names()}
