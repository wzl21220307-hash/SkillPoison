#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EGSA (§3.3) — Local Evidence-Grounded Success Attribution, runnable example.

Covers the real public entry points of `skillpoison/spin/egsa/`:

    annotation.evidence() / attribution() / success_feedback() / consequence()
    annotation.annotate() / annotate_with_consequence() / attribution_templated()
    annotation.G_grounded() / A_credits() / J_sufficient() / N_no_future_instruction()
    annotation.q_attr() / cost_estimate()
    from_trajectory.task_evidence() / success_attribution() / consistency_feedback()
    from_trajectory.counterfactual_consequence() / build_evidence() / adopt_evidence()
    from_trajectory.phi() / phi_from_case() / to_framework() / evidence_manifest()

The evidence is built on a REAL frozen PAWS formation case (see
`selection.load_selection("paws")`). Data root: `SKILLPOISON_DATA_ROOT`.

No network is used. Run from anywhere:

    python3 examples/example_egsa_evidence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skillpoison.spin.egsa import annotation, from_trajectory as ft
from skillpoison.spin.hps import selection


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    case = selection.load_selection("paws", slots="ea")[1]
    proc, s1, s2, facts = case["procedure"], case["raw"]["sentence1"], \
        case["raw"]["sentence2"], case["facts"]
    print(f"case id={case['id']} procedure={proc!r} n_content={facts.get('n_content')}")
    print(f"  sentence1: {s1}")
    print(f"  sentence2: {s2}")

    hr("[1] annotation: the four local slots e_i / a_i / v_i / kappa_i")
    e_i = annotation.evidence(proc, s1, s2, facts)
    a_i = annotation.attribution(proc, s1, s2, facts)
    v_i = annotation.success_feedback(proc, s1, s2, facts)
    k_i = annotation.consequence(proc, s1, s2, facts)
    for name, text in (("e_i", e_i), ("a_i", a_i), ("v_i", v_i), ("kappa_i", k_i)):
        print(f"  {name}: {text}")

    hr("[2] annotation.annotate() / annotate_with_consequence() / templated mode")
    e1, a1 = annotation.annotate(proc, s1, s2, facts, attribution_mode="percase")
    e2, a2 = annotation.annotate(proc, s1, s2, facts, attribution_mode="templated")
    e3, a3 = annotation.annotate_with_consequence(proc, s1, s2, facts)
    print(f"  annotate(percase)   -> a_i starts: {a1[:90]}...")
    print(f"  annotate(templated) -> a_i starts: {a2[:90]}...")
    print(f"  templated a_i is the shared historical wording: "
          f"{a2 == annotation.attribution_templated()}")
    print(f"  annotate_with_consequence -> body has 3 blocks: "
          f"{[b.splitlines()[0][:40] for b in a3.split(chr(10)) if b][:3]}")

    hr("[3] annotation: Q_attr = v . G . A . J . N (construction validity)")
    qa = annotation.q_attr(e_i, a_i, s1, s2, v_i=1,
                           feedback_text=v_i, kappa_text=k_i)
    print(f"  G_grounded(e_i, s1, s2)          = {annotation.G_grounded(e_i, s1, s2)}")
    print(f"  A_credits(a_i)                   = {annotation.A_credits(a_i)}")
    print(f"  J_sufficient(a_i, s1, s2, v, k)  = "
          f"{annotation.J_sufficient(a_i, s1, s2, v_i, k_i)}")
    print(f"  N_no_future_instruction(e_i+a_i) = "
          f"{annotation.N_no_future_instruction(e_i + chr(10) + a_i)}")
    print(f"  q_attr(...)                      = {qa}")
    print("  counterfactual cost model         =",
          json.dumps(annotation.cost_estimate(proc)))

    hr("[4] from_trajectory: M(x_i, tau_i, p*) on a recorded trajectory")
    p_star = "paraphrase is decided by content-word set identity"
    tau = {"trajectory_id": f"paws_recorded_{case['id']}",
           "task_input": case["x"], "gold_label": case["u"],
           "predicted_label": case["u"], "verifier_result": case["v"],
           "decision_basis": case["a"], "evidence_used": case["e"],
           "passed": True, "executed": False, "source": case["asset"]}
    G = ft.build_evidence({"sentence1": s1, "sentence2": s2}, tau, p_star,
                          credited_step="content-word-set comparison",
                          downstream="the content-word comparison")
    for slot, text in G.items():
        print(f"  G[{slot}]: {str(text)[:150]}")

    hr("[5] from_trajectory.phi() / phi_from_case() — Eq. 5 record r_i")
    r_i = ft.phi_from_case(case, "paws", title=ft.TITLE)
    print("  r_i slots:", list(r_i.keys()))
    print("  T     :", r_i["T"])
    print("  e     :", str(r_i["e"])[:110], "...")
    print("  u     :", r_i["u"])

    hr("[6] from_trajectory.to_framework() — S_f into the Trace2Skill schema")
    fw = ft.to_framework(r_i, dataset="paws")
    print("  top-level keys:", list(fw.keys()))
    print("  items[0] keys :", list(fw["items"][0].keys()))
    print("  source_file   :", fw["source_file"])
    print("  content head  :", fw["items"][0]["content"][:110].replace("\n", " | "))
    print("  evidence_manifest:",
          json.dumps(ft.evidence_manifest(), ensure_ascii=False)[:150], "...")

    hr("[7] KNOWN DEFECT in the shipped source (identical upstream)")
    try:
        ft.trajectory_from_case(case, "paws")
    except Exception as exc:  # noqa: BLE001 - recorded verbatim for the release
        print(f"  from_trajectory.trajectory_from_case(case, 'paws') -> "
              f"{type(exc).__name__}: {exc}")
        print("  -> this builder is NOT used by the pipeline; adopt_evidence() +")
        print("     phi_from_case() above provide the recorded-trajectory path.")


if __name__ == "__main__":
    main()
