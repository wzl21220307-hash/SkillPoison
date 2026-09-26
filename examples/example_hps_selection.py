#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HPS (§3.2) — Hierarchical Procedure Selection, runnable example.

Covers the real public entry points of `skillpoison/spin/hps/`:

    abstraction.describe()
    procedures.PROCEDURES / applicable() / procedure()
    validity.check() / assert_abstraction_consistency()
    outcomes.Verdict / Grid / edit() / behavior_importance()
    outcomes.requirements_for_code() / requirements_for_classification() / requirement_fraction()
    outcomes.select_by_rule() / rule_report() / threshold_report()
    selection.load_selection() / selection_table() / applicability() / fixed_selection_manifest()
    cases.excluded_ids()

Data root: `SKILLPOISON_DATA_ROOT` (default /home/zlz/test). The frozen PAWS /
HANS / DS1000 formation assets are read from `<root>/_shared/formation_assets/...`
and `<root>/deepseekflash/...`; no network is used.

Run from anywhere:

    python3 examples/example_hps_selection.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skillpoison.spin import common
from skillpoison.spin.hps import abstraction, cases, outcomes, procedures, selection, validity


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    print(f"SKILLPOISON_DATA_ROOT -> {common.ROOT}")

    hr("[1] abstraction.describe() — the target abstraction p*")
    print(json.dumps(abstraction.describe(), indent=2, ensure_ascii=False))

    hr("[2] procedures.applicable(s1, s2) — the mechanical test A(x,c)")
    frozen = selection.load_selection("paws", slots="ea")
    for case in frozen[:3]:
        s1, s2 = case["raw"]["sentence1"], case["raw"]["sentence2"]
        key, facts = procedures.applicable(str(s1), str(s2))
        print(f"  case {case['id']}: applicable() -> ({key!r}, {facts})")
        print(f"    sentence1: {s1}")
        print(f"    sentence2: {s2}")
    print("  procedures.procedure('order_head') ->",
          procedures.procedure("order_head").description)
    print("  C* procedure keys:", [p.key for p in procedures.PROCEDURES])

    hr("[3] validity.check() — the local validity requirement Q_local")
    for case in frozen[:3]:
        lv = validity.check(gold_label=1, procedure_applicable=case["t_i"],
                            attribution_text=case["a"] or "")
        print(f"  case {case['id']}: Q_local={lv.Q_local} {lv.as_dict()}")
    print("  assert_abstraction_consistency() ->",
          json.dumps(validity.assert_abstraction_consistency(), ensure_ascii=False))

    hr("[4] outcomes: Verdict / Grid / edit / behavior_importance (Eq. 2)")
    z = outcomes.Verdict("paraphrase")
    z_alt = outcomes.Verdict("non-paraphrase")
    print(f"  Verdict edit={outcomes.edit(z, z_alt)} |z|={z.size} "
          f"R={outcomes.behavior_importance(z, z_alt)}")
    print(f"  identical verdicts: edit=0 R={outcomes.behavior_importance(z, z)}")
    g1 = outcomes.Grid(rows=(("r0", {"a": 1, "b": 2}), ("r1", {"a": 3, "b": 4})),
                       columns=("a", "b"))
    g2 = outcomes.Grid(rows=(("r0", {"a": 1, "b": 2}), ("r1", {"a": 9, "b": 4})),
                       columns=("a", "b"))
    print(f"  Grid edit={outcomes.edit(g1, g2)} |z|={g1.size} "
          f"R={outcomes.behavior_importance(g1, g2)}")

    hr("[5] outcomes: task requirements and requirement_fraction (Eq. 3)")
    checks = outcomes.requirements_for_code(passed=True, error_type="pass")
    print(f"  requirements_for_code(passed=True, error_type='pass') -> {checks} "
          f"O={outcomes.requirement_fraction(checks)}")
    probe = {"parse_error": False, "predicted_label": "paraphrase", "correct": True}
    cls = outcomes.requirements_for_classification(probe, selection.ALLOWED_LABELS["paws"])
    print(f"  requirements_for_classification(probe, paws labels) -> {cls} "
          f"O={outcomes.requirement_fraction(cls)}")

    hr("[6] selection: A(x,c) gate + O threshold (accepted rule A_O)")
    report = outcomes.rule_report(frozen)
    print(json.dumps({k: report[k] for k in
                      ("rule", "n_X", "n_X_star", "X_star", "excluded_by_A")},
                     indent=2, ensure_ascii=False))
    print("  R_reported (diagnostic, never excludes):",
          json.dumps(report["R_reported"], ensure_ascii=False))
    diag = outcomes.threshold_report(frozen, tau_R=0.5, tau_O=1.0)
    print("  threshold_report(superseded R_then_O diagnostic):",
          json.dumps({k: diag[k] for k in ("n_X", "n_X_R", "n_X_star", "R_undefined")},
                     ensure_ascii=False))
    print("  select_by_rule(frozen) n =", len(outcomes.select_by_rule(frozen)))
    print("  select_by_importance(frozen, tau_R=0.5) n =",
          len(outcomes.select_by_importance(frozen, tau_R=0.5)))

    hr("[7] selection.load_selection() over the three frozen datasets")
    for ds in ("paws", "hans", "ds1000"):
        rows = selection.load_selection(ds, slots="ea")
        table = selection.selection_table(ds)
        print(f"  {ds:7s} n={len(rows):2d} "
              f"X*={sum(1 for r in rows if r['t_i'] and r['O'] >= 1.0)} "
              f"procedures={sorted({r['procedure'] for r in rows}, key=str)}")
        print(f"          first row: {json.dumps(table[0], ensure_ascii=False)}")
    print("  fixed_selection_manifest() assets:")
    print(json.dumps(selection.fixed_selection_manifest()["assets"],
                     indent=2, ensure_ascii=False))

    hr("[8] selection.applicability() + cases.excluded_ids()")
    raw = frozen[0]["raw"]
    print("  paws:", selection.applicability(raw, "paws")[:2])
    hans_row = selection.load_selection("hans")[0]["raw"]
    print("  hans:", selection.applicability(hans_row, "hans")[:2])
    ds_row = selection.load_selection("ds1000")[0]["raw"]
    print("  ds1000:", selection.applicability(ds_row, "ds1000")[:2])
    print("  cases.excluded_ids() ->", len(cases.excluded_ids()), "ids (both eval pools)")


if __name__ == "__main__":
    main()
