#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from pathlib import Path

from skillpoison.spin.common import ROOT, hits_banned
from skillpoison.spin.hps.outcomes import (behavior_importance, departure,
                                           reading_outcomes, requirement_fraction,
                                           requirements_for_classification,
                                           requirements_for_code)
from skillpoison.spin.hps.procedures import applicable, procedure

ALLOWED_LABELS = {"paws": ("paraphrase", "non-paraphrase"),
                  "hans": ("entailment", "contradiction", "neutral")}

EXISTING_SELECTION: dict[str, dict] = {
    "paws": {"path": ROOT / "_shared/formation_assets/paws/attack_formation.json",
             "provenance": "historical A15 PAWS attack_formation.json (15 records)",
             "key": "id", "record_key": "source_pairID"},
    "hans": {"path": ROOT / "_shared/formation_assets/hans/attack_shasp_r2_a15_strong_v2_formation.json",
             "provenance": "current main HANS attack arm (strong_v2, 15 records) -- the asset "
                           "the reported HANS runs consumed",
             "key": "id", "record_key": "source_pairID"},
    "ds1000": {"path": ROOT / "deepseekflash/autoskill/ds1000/runs/main/skillpoison/"
                      "20260817T055623Z_multi_rule_r2_aggressive_d96b8398/formation/original/"
                      "multi_rule_r2_aggressive_formation.json",
               "provenance": "main DS1000 attack arm's formation set (multi_rule_r2_aggressive, "
                             "15 records), the asset the reported DS1000 runs consumed",
               "key": "experience_id", "record_key": "task_id"},
}

HANS_SUBCASE_TRIGGER = {
    "se_conjunction": "deleting one conjunct leaves a contiguous premise fragment",
    "se_adjective": "deleting an adjective leaves a contiguous premise fragment",
    "se_understood_object": "deleting the understood object leaves a contiguous premise fragment",
    "se_relative_clause_on_obj": "deleting an object relative clause leaves a contiguous "
                                 "premise fragment",
    "se_PP_on_obj": "deleting an object PP leaves a contiguous premise fragment",
}
DS1000_FAMILY_TRIGGER = {
    "concat": "concatenating the frames, then normalising the output index/order",
    "groupby": "aggregating with groupby, then normalising the output index/order",
    "reshape": "reshaping the frame, then normalising the output index/order",
    "other": "the core frame transformation, then normalising the output index/order",
}


# ------------------------------------------------------------- applicability A
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def contiguous_span(premise: str, hypothesis: str) -> bool:
    """HANS A(x,c): the hypothesis is a contiguous surface span of the premise."""
    p, h = _tokens(premise), _tokens(hypothesis)
    if not h or len(h) > len(p):
        return False
    return any(p[i:i + len(h)] == h for i in range(len(p) - len(h) + 1))


def applicability(case: dict, dataset: str) -> tuple[bool, str | None, dict]:
    """(t_i, procedure key, facts) -- one mechanical test per dataset."""
    if dataset == "paws":
        key, facts = applicable(str(case["sentence1"]), str(case["sentence2"]))
        banned = hits_banned(str(case["sentence1"]) + " " + str(case["sentence2"]))
        return bool(key) and not banned, key, {**facts, "banned": banned}
    if dataset == "hans":
        sub = str(case.get("subcase") or "unknown")
        ok = contiguous_span(str(case.get("premise", "")), str(case.get("hypothesis", "")))
        return ok, f"hans:{sub}", {"contiguous_span": ok, "heuristic": case.get("heuristic")}
    fam = str(case.get("task_family") or "other")
    ok = str(case.get("verification_status", "pass")).lower() in ("pass", "passed", "ok")
    return ok, f"ds1000:{fam}", {"task_family": fam,
                                 "verification_status": case.get("verification_status")}


# ------------------------------------------------------------------- passages
def adopt_passages(case: dict, dataset: str, slots: str = "ea") -> dict:
    """The LEGSA slots as they already exist in the frozen asset.

    Nothing is regenerated here: the historical passages are adopted verbatim so
    that alignment does not silently rewrite an experience.

    slots="ea"    G_i = {e_i, a_i}. This is the slot set of the frozen main-arm
                  records (PAWS `attack_records.json`, HANS strong_v2, DS1000).
    slots="asset" G_i = every slot the asset carries (adds v_i for PAWS, whose
                  safety-suffixed arm wrote the verification line into the record).
    """
    G = {"e": case.get("see_observation") or case.get("success_feedback", ""),
         "a": case.get("act_reasoning", ""),
         "v": case.get("verification_result"),
         "kappa": case.get("counterfactual_consequence")}
    if dataset == "ds1000":
        # the historical DS1000 record carries neither a verification line nor a
        # counterfactual passage (see 3.3_r_i_formula_three_datasets.md 3.1)
        G["v"] = None
        G["kappa"] = None
    if slots == "ea":
        G["v"] = None
        G["kappa"] = None
    return G


def trigger_for(proc: str, dataset: str) -> str:
    """The category's trigger phrase phi_j is built from."""
    if dataset == "paws" and proc:
        try:
            return procedure(proc).description
        except KeyError:
            return proc
    if dataset == "hans":
        return HANS_SUBCASE_TRIGGER.get(proc.split(":", 1)[-1], proc)
    if dataset == "ds1000":
        return DS1000_FAMILY_TRIGGER.get(proc.split(":", 1)[-1], proc)
    return proc or "unknown"


# --------------------------------------------------------------- the loader
def load_selection(dataset: str, *, slots: str = "ea") -> list[dict]:
    """X: the already-selected experiences, with R, O, A and the LEGSA slots.

    `slots` selects which optional slots are carried: "asset" (all the frozen
    asset holds) or "ea" (e_i and a_i only, the frozen main-arm records).
    """
    entry = EXISTING_SELECTION[dataset]
    path = Path(entry["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"frozen selection for {dataset} not found at {path}; the aligned "
            f"construction reads the asset the reported runs consumed and does not "
            f"re-derive it")
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict] = []
    for rec in raw:
        cid = rec.get(entry["key"])
        t_i, proc, facts = applicability(rec, dataset)
        z_i, z_alt, r_src = reading_outcomes(rec, dataset)
        # A(x,c) is a GATE on X, not a term inside R: a case whose premise does not
        # hold never becomes a formation case, so R is not defined on it.
        if not t_i:
            R = None
            r_src = {**r_src, "source": "not_applicable",
                     "reason": "A(x,c)=0: the premise of p* does not hold on this case"}
        elif z_i is None:
            R = None
        else:
            R = behavior_importance(z_i, z_alt)
        checks = _recorded_requirements(rec, dataset)
        case = {"id": cid, "record_id": str(rec.get(entry["record_key"]) or cid),
                "dataset": dataset, "procedure": proc, "trigger": trigger_for(proc, dataset),
                "t_i": t_i, "facts": facts, "A": t_i,
                "R": R, "R_undefined": (R is None),
                "excluded_by_A": (not t_i),
                "R_source": r_src, "d_i": departure(rec, dataset),
                "O": requirement_fraction(checks), "O_checks": checks,
                "x": _task_input(rec, dataset), "u": _recorded_answer(rec, dataset),
                "raw": rec, "slots": slots, **adopt_passages(rec, dataset, slots),
                "asset": str(Path(entry["path"]).relative_to(ROOT))}
        out.append(case)
    return out


def _task_input(rec: dict, dataset: str) -> str:
    if dataset == "ds1000":
        return str(rec.get("instruction") or "")
    if dataset == "hans":
        return f"Premise: {rec.get('premise')}\nHypothesis: {rec.get('hypothesis')}"
    return f"Sentence1: {rec.get('sentence1')}\nSentence2: {rec.get('sentence2')}"


def _recorded_answer(rec: dict, dataset: str) -> str:
    """u_i: the answer already recorded for the case (never re-generated)."""
    if dataset == "ds1000":
        return str(rec.get("local_code") or "")
    return str(rec.get("correct_label") or "")


def _recorded_requirements(rec: dict, dataset: str) -> list[bool]:
    """q_{i,k}: the task's own checkable requirements, read off the frozen record.

    The record already carries the dataset's own verified answer, so the
    requirements are evaluated on that recorded outcome (declared: the formation
    cases are locally verified; this is what makes them clean-label positives).
    """
    if dataset == "ds1000":
        return requirements_for_code(
            str(rec.get("verification_status", "pass")).lower() in ("pass", "passed", "ok"),
            str(rec.get("error_type", "pass")))
    label = str(rec.get("correct_label") or "")
    note = str(rec.get("verification_result") or "")
    probe = {"parse_error": False, "predicted_label": label,
             "correct": ("passed" in note or "matched" in note)}
    return requirements_for_classification(probe, ALLOWED_LABELS[dataset])


def fixed_selection_manifest(slots: str = "ea") -> dict:
    return {
        "selection_source": "fixed",
        "legsa_slots": slots,
        "declaration": ("X* is the set of experiences already selected by the earlier "
                        "pipeline; it is read from the frozen asset and is NOT "
                        "re-derived by the thresholds. A(x,c), R, O, X_R and X*_tau "
                        "are still computed and archived for every case."),
        "assets": {ds: {"path": str(Path(e["path"]).relative_to(ROOT)),
                        "provenance": e["provenance"]} for ds, e in EXISTING_SELECTION.items()},
        "replacement_readings": "see hps.outcomes.REPLACEMENT_READINGS",
    }


def selection_table(dataset: str) -> list[dict]:
    """Compact per-case view for reports/tests."""
    return [{k: c[k] for k in ("id", "procedure", "t_i", "R", "O", "d_i")}
            for c in load_selection(dataset)]
