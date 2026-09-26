#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field

from skillpoison.spin.egsa.annotation import (annotate,
                                              annotate_with_consequence,
                                              cost_estimate, q_attr)
from skillpoison.spin.hps.abstraction import P_STAR
from skillpoison.spin.hps.procedures import PROCEDURES
from skillpoison.spin.hps.validity import check as local_validity_check
from skillpoison.spin.serialize.trace2skill import legacy_to_record as to_record

PER_PROCEDURE = 3          # default: every procedure equally
EXPECTED_TOTAL = PER_PROCEDURE * len(PROCEDURES)

# Named construction rounds. Each maps procedure -> quota; the quotas of a round
# must sum to the formation budget (15), so all rounds are size-matched.
ROUNDS: dict[str, dict[str, int]] = {
    "A": {"order_head": 3, "order_perm": 3, "surface_only": 3,
          "function_words": 3, "name_phrase": 3},              # 已跑：无 role_swap
    "B": {"role_swap": 5, "order_head": 5, "order_perm": 5},    # order/role family only
    "C": {"role_swap": 15},                                     # role transposition only
    "D": {"role_swap": 3, "order_head": 2, "order_perm": 2,
          "surface_only": 2, "function_words": 3, "name_phrase": 3},   # wide, role-weighted
}


@dataclass
class FormationSet:
    records: list[dict] = field(default_factory=list)
    construction: list[dict] = field(default_factory=list)
    per_procedure: int = PER_PROCEDURE
    p_star: str = P_STAR.statement
    round_name: str = "A"
    with_consequence: bool = False
    attribution_mode: str = "percase"

    def __len__(self) -> int:
        return len(self.records)


def assemble(buckets: dict[str, list], *, per_procedure: int = PER_PROCEDURE,
             verbose: bool = True, round_name: str = "A",
             with_consequence: bool = False,
             attribution_mode: str = "percase",
             sufficiency_redundancy: bool = False) -> FormationSet:
    """Build H* from applicability buckets, deterministically."""
    quotas = ROUNDS.get(round_name)
    if quotas is None:
        raise ValueError(f"unknown round {round_name!r}; known {sorted(ROUNDS)}")
    if attribution_mode not in ("percase", "templated"):
        raise ValueError(f"unknown attribution_mode {attribution_mode!r}")
    fs = FormationSet(per_procedure=per_procedure)
    fs.round_name = round_name                      # type: ignore[attr-defined]
    fs.with_consequence = with_consequence          # type: ignore[attr-defined]
    fs.attribution_mode = attribution_mode          # type: ignore[attr-defined]
    fs.sufficiency_redundancy = sufficiency_redundancy  # type: ignore[attr-defined]
    for proc in PROCEDURES:
        quota = quotas.get(proc.key, 0)
        if quota == 0:
            continue
        items = list(buckets.get(proc.key, []))
        # deterministic order: longest case first, then id  -> fixed submission order
        items.sort(key=lambda t: (-t.get("n_content", 0), t["id"]))
        taken = items[:quota]
        if len(taken) < quota:
            raise ValueError(f"procedure {proc.key}: only {len(taken)} candidates, "
                             f"need {quota}")
        for c in taken:
            if with_consequence:
                e_i, a_i = annotate_with_consequence(
                    proc.key, c["sentence1"], c["sentence2"], c,
                    attribution_mode=attribution_mode,
                    sufficiency_redundancy=sufficiency_redundancy)
            else:
                e_i, a_i = annotate(proc.key, c["sentence1"], c["sentence2"], c,
                                    attribution_mode=attribution_mode)
            lv = local_validity_check(gold_label=1, procedure_applicable=True,
                                      attribution_text=a_i)
            if lv.Q_local != 1:
                raise ValueError(f"Q_local != 1 for case {c['id']} ({proc.key})")
            rec = to_record(c["id"], proc.key, c["sentence1"], c["sentence2"], e_i, a_i, c)
            fs.records.append(rec)
            # §3.2 construction-validity predicate over the actual slot contents
            _ai, _fb, _kap = _split_act_slots(a_i)
            qa = q_attr(e_i, _ai, c["sentence1"], c["sentence2"], v_i=1,
                        feedback_text=_fb, kappa_text=_kap)
            fs.construction.append({
                "id": c["id"], "procedure": proc.key, "n_content": c.get("n_content"),
                "head_changed": c.get("head_changed"),
                "transposed": c.get("transposed"),
                "kappa": cost_estimate(proc.key),
                **lv.as_dict(), **qa,
            })
        if verbose:
            print(f"    {proc.key:14s} -> {len(taken)} records")
    return fs


def _split_act_slots(a_i: str) -> tuple[str, str, str]:
    """(attribution, feedback, counterfactual) out of the assembled body text."""
    ai, _, rest = a_i.partition("\nVerification result for this pair:")
    fb, _, kap = rest.partition("\nIn this local case")
    return (ai.strip(),
            ("Verification result for this pair:" + fb).strip() if rest else "",
            ("In this local case" + kap).strip() if kap else "")


def recurrence_stats(fs: FormationSet) -> dict:
    """CEIR consistency checks over H*."""
    from collections import Counter
    procs = Counter(c["procedure"] for c in fs.construction)
    titles = {r["items"][0]["title"] for r in fs.records}
    credits = sum(1 for r in fs.records
                  if "content-word set" in r["items"][0]["content"].lower())
    # §3.3 accumulated positive support R_n and boundary evidence B_n.
    # B_n counts records that reveal a boundary failure; by construction the
    # submitted records are all locally valid, so B_n = 0.
    R_n = sum(c.get("Q_attr", 0) for c in fs.construction)
    B_n = sum(1 for c in fs.construction if c.get("boundary_failure"))
    return {
        "p_star": fs.p_star,
        "n_records": len(fs.records),
        "R_n_positive_support": R_n,
        "B_n_boundary_evidence": B_n,
        "n_Q_attr_1": sum(c.get("Q_attr", 0) for c in fs.construction),
        "sum_C_omit": round(sum(c.get("kappa", {}).get("C_omit", 0.0)
                                for c in fs.construction), 4),
        "per_procedure": dict(procs),
        "distinct_procedures": len(procs),
        "target_procedure_recurs_in_all": credits == len(fs.records),
        "credit_count": credits,
        "single_parent_abstraction": len(titles) == 1,
        "titles": sorted(titles),
        "failure_counterexamples": 0,   # construction admits label==1 cases only
    }
