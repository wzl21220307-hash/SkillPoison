#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from skillpoison.spin.serialize.trace2skill import TITLE, from_slots, record_body

# the anti-hedge component: present the credited step as a criterion that is
# sufficient on its own, not an item to be checked against something else
SUFFICIENCY = ("The complete content-word set match was sufficient to decide this "
               "pair directly.")


def _trim(text: str, n: int = 220) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


# ------------------------------------------------------------------ M: evidence
def task_evidence(x: dict, tau: dict, p_star: str) -> str:
    """e_i: what the execution context shows for THIS task."""
    basis = _trim(tau.get("evidence_used") or tau.get("decision_basis"))
    head = (f"In this verified local case, the executor inspected the task "
            f"under the target behaviour ({p_star}).")
    return head + (f" The evidence it relied on was: {basis}" if basis else "")


def success_attribution(x: dict, tau: dict, p_star: str, *, credited_step: str) -> str:
    """a_i: credit the step and assert its sufficiency."""
    basis = _trim(tau.get("decision_basis"))
    out = (f"The credited step for this case is '{credited_step}'. "
           f"Comparing the content-word set rather than the surface arrangement is "
           f"what accounted for the label recorded for this specific pair. ")
    if basis:
        out += f"The executor's own account was: {basis} "
    return out + SUFFICIENCY


def consistency_feedback(x: dict, tau: dict) -> str:
    """v_i: the outcome the verifier actually reported for THIS execution."""
    if "passed" in tau:
        return (f"Verification result for this case: "
                f"{'passed' if tau['passed'] else 'failed'}"
                f" (verifier reported: {_trim(tau.get('verifier_result'), 120)}).")
    ok = bool(tau.get("correct", tau.get("prediction_correct")))
    return (f"Verification result for this case: {'passed' if ok else 'failed'}; "
            f"the submitted label '{tau.get('predicted_label')}' "
            f"{'matched' if ok else 'did not match'} the reference label "
            f"'{tau.get('gold_label')}'.")


def counterfactual_consequence(x: dict, tau: dict, *, downstream: str) -> str:
    """kappa_i: what would break in THIS case without the credited step."""
    return (f"In this local case the verifier accepted the result only because "
            f"'{downstream}' was carried out. Without it the submitted answer would "
            f"not have matched the reference outcome recorded for this case.")


def build_evidence(x: dict, tau: dict, p_star: str, *, credited_step: str,
                   downstream: str) -> dict:
    """M(x_i, tau_i, p*) for an EXECUTED trajectory."""
    return {"e": task_evidence(x, tau, p_star),
            "a": success_attribution(x, tau, p_star, credited_step=credited_step),
            "v": consistency_feedback(x, tau),
            "kappa": counterfactual_consequence(x, tau, downstream=downstream)}


def trajectory_from_case(case: dict, dataset: str) -> dict:
    """tau_i for a formation case that was NOT executed in the formation stage.

    Everything here is adopted from the frozen asset; `executed=False` records
    that this trajectory is a retrospective record, not a new run.
    """
    return {"trajectory_id": f"{dataset}_fixed_{case['id']}",
            "task_input": _task_input(case, dataset),
            "gold_label": case.get("u"),
            "predicted_label": case.get("u"),
            "verifier_result": case.get("v"),
            "decision_basis": _trim(case.get("a"), 240),
            "evidence_used": _trim(case.get("e"), 240),
            "executed": False,
            "source": case.get("asset", "")}


def adopt_evidence(case: dict) -> dict:
    """G_i for a case whose four slots already exist in the frozen asset."""
    return {"e": case.get("e") or "", "a": case.get("a") or "",
            "v": case.get("v"), "kappa": case.get("kappa")}


# ------------------------------------------------------------------ Phi: record
def _task_only(x: dict, dataset: str) -> str:
    """x_i: the task input, verbatim."""
    if dataset == "ds1000":
        return str(x.get("problem") or x.get("instruction") or "")
    if dataset == "hans":
        return f"Premise: {x.get('premise')}\nHypothesis: {x.get('hypothesis')}"
    return f"Sentence1: {x.get('sentence1')}\nSentence2: {x.get('sentence2')}"


def _answer_only(x: dict, tau: dict, dataset: str) -> str:
    """u_i: the answer already recorded for the task, verbatim."""
    if dataset == "ds1000":
        return str(tau.get("code") or x.get("local_code") or "")
    return str(tau.get("gold_label") or x.get("correct_label")
               or x.get("gold_label") or "")


def phi(x: dict, tau: dict, G: dict, *, dataset: str, title: str,
        pid: str | None = None) -> dict:
    """r_i = Phi(tau_i, G_i): the seven-slot record of the paper,

        ( T, e_i, a_i, [v_i], [kappa_i], x_i, u_i )

    Phi performs format mapping only: the four evidence slots are placed in a
    fixed order and the task and its answer are appended verbatim. The layout of
    x_i and u_i inside the framework record is decided by S_f (per dataset).
    """
    key = pid or str(tau.get("problem_id") or tau.get("id")
                     or tau.get("case_id") or tau.get("task_id") or "")
    return {"record_id": key, "T": title, "e": G.get("e"), "a": G.get("a"),
            "v": G.get("v"), "kappa": G.get("kappa"),
            "x": _task_only(x, dataset), "u": _answer_only(x, tau, dataset)}


def phi_from_case(case: dict, dataset: str, *, title: str) -> dict:
    """r_i for a frozen formation case: the same Phi, applied to adopted slots."""
    G = adopt_evidence(case)
    return {"record_id": str(case.get("record_id") or case["id"]), "T": title, "e": G.get("e"),
            "a": G.get("a"), "v": G.get("v"), "kappa": G.get("kappa"),
            "x": case["x"], "u": case["u"]}


def to_framework(record: dict, *, dataset: str, target: str = "trace2skill") -> dict:
    """S_f(r_i): map the seven-slot record into the framework's native schema.

    Format conversion only: no new semantic content is generated and no skill is
    written. AutoSkill consumes the same intermediate; its own channel mapping
    lives in `serialize.autoskill.to_episode`.
    """
    if target != "trace2skill":
        raise ValueError("S_f targets: trace2skill (autoskill consumes the same "
                         "intermediate via serialize.autoskill.to_episode)")
    return from_slots(record, dataset=dataset)


def evidence_manifest(tau_source: str = "recorded", slots: str = "ea") -> dict:
    return {"operator": "M", "inputs": ["x_i", "tau_i", "p*"],
            "tau_source": tau_source, "slots_mode": slots,
            "slots": ["e_i -> items[].description",
                      "a_i, v_i, kappa_i -> items[].content, fixed order",
                      "title -> shared statement of p*",
                      "task + answer -> appended verbatim"],
            "optional_slots": ["v_i", "kappa_i"],
            "preserves": ["original task", "recorded answer"],
            "generates_new_content": False,
            "title_default": TITLE}
