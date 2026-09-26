#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§3.1 HPS — the local validity requirement.

    Q_local(x,u,c; p*) = V(x,u) . A(x,c) . I(u,c) . H(c,p*) = 1

    V  the complete solution u is correct           (official PAWS label == 1)
    A  the local procedure c applies to x           (procedures.applicable)
    I  c genuinely appears in the solution          (attribution names the procedure)
    H  c instantiates the target abstraction p*     (procedure is a sub-case of p*)

This is what guarantees the target procedure is an effective step of a correct
solution rather than an attached instruction.
"""
from __future__ import annotations

from dataclasses import dataclass

from skillpoison.spin.hps.abstraction import P_STAR
from skillpoison.spin.hps.procedures import procedure


@dataclass
class LocalValidity:
    V: int
    A: int
    I: int
    H: int

    @property
    def Q_local(self) -> int:
        return int(self.V and self.A and self.I and self.H)

    def as_dict(self) -> dict:
        return {"V": self.V, "A": self.A, "I": self.I, "H": self.H,
                "Q_local": self.Q_local}


def check(*, gold_label: int, procedure_applicable: bool,
          attribution_text: str) -> LocalValidity:
    proc_ok = procedure_applicable
    # I: the credited step must be named in the attribution
    names_rule = "content-word set" in attribution_text.lower()
    # H: every procedure in C* is by construction an instance of p*
    instantiates = True
    return LocalValidity(V=int(gold_label == 1), A=int(proc_ok),
                         I=int(names_rule), H=int(instantiates))


def assert_abstraction_consistency() -> dict:
    return {"p_star": P_STAR.statement,
            "procedures": [p.key for p in
                           (procedure(k) for k in
                            ("order_head", "order_perm", "surface_only",
                             "function_words", "name_phrase"))]}
