#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkillPoison method pipeline — chain the three stages in the paper's order.

    HPS (§3.2) -> EGSA (§3.3) -> CEIR (§3.4) -> Serialize (S_f)
        -> [HAND-OFF: AutoSkill / Trace2Skill native Skill extraction] -> Evaluate

What this script does (all of it runs offline):

  1. HPS    `hps.cases.enumerate_cases()` builds X and the A(x,c) buckets from
            the raw PAWS parquet; `hps.outcomes.rule_report()` reports the
            accepted A_O selection on the frozen `X*` read by
            `hps.selection.load_selection()`.
  2. EGSA   `egsa.from_trajectory.phi_from_case()` builds the paper's seven-slot
            record r_i = (T, e_i, a_i, [v_i], [kappa_i], x_i, u_i) from the
            frozen attribution slots, and `egsa.annotation` is exercised on one
            case to show how a fresh G_i is built.
  3. CEIR   `ceir.categories` scores the categories (S_j, D_j, U_j) and selects
            C*; `ceir.assemble.assemble()` builds H_A on the PAWS round and
            `ceir.audit.audit()` gates it before extraction.
  4. SERIAL `egsa.from_trajectory.to_framework()` / `serialize.trace2skill` map
            every r_i into the framework-native record schema.

The hand-off to the native extractor is NOT performed here: AutoSkill and
Trace2Skill are external victim frameworks (see third_party/README.md). The
script writes their input files and prints the exact path to hand over.

Usage:
    SKILLPOISON_DATA_ROOT=/path/to/repo \
        python3 scripts/run_method_pipeline.py --dataset paws --round A

Requires pandas + pyarrow for step 1 (raw parquet); the other steps are stdlib
only. No network access.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skillpoison.spin import common
from skillpoison.spin.ceir import assemble as assemble_mod
from skillpoison.spin.ceir import audit as audit_mod
from skillpoison.spin.ceir import categories
from skillpoison.spin.egsa import annotation, from_trajectory as ft
from skillpoison.spin.hps import abstraction, cases, outcomes, selection
from skillpoison.spin.serialize import trace2skill

DATASETS = ("paws", "hans", "ds1000")


def hr(step: str, title: str) -> None:
    print("\n" + "=" * 74)
    print(f"[{step}] {title}")
    print("=" * 74)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="paws", choices=DATASETS)
    ap.add_argument("--round", default="A", choices=sorted(assemble_mod.ROUNDS))
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--top-m", type=int, default=5)
    ap.add_argument("--out", default="runs/method_pipeline")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = args.dataset
    print(f"SkillPoison method pipeline | dataset={ds} round={args.round} "
          f"lam={args.lam} top_m={args.top_m}")
    print(f"SKILLPOISON_DATA_ROOT -> {common.ROOT}")
    print(f"output dir            -> {out.resolve()}")
    print(f"p*: {abstraction.describe()['p_star']}")

    # ---------------------------------------------------------------- 1. HPS
    hr("1/4 HPS §3.2", "Hierarchical Procedure Selection")
    if ds == "paws":
        try:
            buckets, skipped = cases.enumerate_cases(verbose=True)
            print(f"  X buckets: {sum(len(v) for v in buckets.values())} applicable, "
                  f"{skipped} dropped for a banned term")
            (out / "hps_buckets_summary.json").write_text(
                json.dumps({k: len(v) for k, v in buckets.items()} | {"skipped_banned": skipped},
                           indent=2), encoding="utf-8")
        except ModuleNotFoundError as exc:
            print(f"  enumerate_cases unavailable ({exc}); install requirements.txt")
            buckets = {}
    else:
        buckets = {}
        print(f"  raw enumeration is PAWS-only here; using the frozen {ds} asset")

    frozen = selection.load_selection(ds, slots="ea")
    report = outcomes.rule_report(frozen)
    print(f"  frozen X  = {report['n_X']}")
    print(f"  X* (A_O)  = {report['n_X_star']}  excluded_by_A={report['excluded_by_A']}")
    print(f"  R reported (diagnostic only): {report['R_reported']}")
    (out / "hps_selection_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # --------------------------------------------------------------- 2. EGSA
    hr("2/4 EGSA §3.3", "Local Evidence-Grounded Success Attribution")
    selected = outcomes.select_by_rule(frozen)
    records = [ft.phi_from_case(case, ds, title=ft.TITLE) for case in selected]
    print(f"  r_i built for X*: {len(records)} record(s)")
    for rec in records[:2]:
        print(f"    record_id={rec['record_id']} | e_i={str(rec['e'])[:70]}...")
    demo = selected[1]
    if ds == "paws":
        # `annotation` builds G_i for the PAWS text procedures (content-word set
        # identity); HANS/DS1000 adopt their frozen slots instead.
        proc = demo["procedure"] or "order_head"
        s1, s2 = demo["raw"]["sentence1"], demo["raw"]["sentence2"]
        e_i, a_i = annotation.annotate(proc, s1, s2, demo["facts"])
        qa = annotation.q_attr(e_i, a_i, s1, s2, v_i=1,
                               feedback_text=annotation.success_feedback(proc, s1, s2, demo["facts"]),
                               kappa_text=annotation.consequence(proc, s1, s2, demo["facts"]))
        print(f"  Q_attr on {demo['id']}: {qa}")
    else:
        print(f"  {ds}: G_i slots adopted verbatim from the frozen asset "
              f"(adopt_evidence); annotation.py is PAWS-text specific")
    (out / "egsa_records.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    # --------------------------------------------------------------- 3. CEIR
    hr("3/4 CEIR §3.4", "Global Cross-Trajectory Inductive Reinforcement")
    cats = categories.group_by_category(frozen)
    scored = categories.score_categories(cats, lam=args.lam)
    sel_report = categories.selection_report(scored, top_m=args.top_m)
    print(f"  categories = {sel_report['n_categories']}, "
          f"U_j defined = {sel_report['n_defined']}, "
          f"U_j undefined = {sel_report['U_undefined']}")
    print(f"  C* = {sel_report['C_star']}")
    pool = categories.build_pool(categories.select_categories(scored, top_m=args.top_m))
    print(f"  |H_A| from frozen categories = {len(pool)}")
    (out / "ceir_category_report.json").write_text(
        json.dumps(sel_report, indent=2, ensure_ascii=False), encoding="utf-8")

    framework_records: list[dict] = []
    if ds == "paws" and buckets:
        hr("3b/4 CEIR", f"assemble H_A on PAWS round {args.round!r} + audit")
        fs = assemble_mod.assemble(buckets, round_name=args.round, verbose=True)
        rec = assemble_mod.recurrence_stats(fs)
        print(f"  recurrence: n_records={rec['n_records']} "
              f"distinct_procedures={rec['distinct_procedures']} "
              f"R_n={rec['R_n_positive_support']} B_n={rec['B_n_boundary_evidence']} "
              f"single_p*={rec['single_parent_abstraction']}")
        audit_report = audit_mod.audit(fs, cases.excluded_ids())
        print(f"  audit.all_pass = {audit_report['all_pass']} "
              f"(banned={audit_report['banned_terms_hit']}, "
              f"imperative={audit_report['imperative_hits']}, "
              f"overlap={audit_report['overlap_with_eval_pools']})")
        (out / "ceir_formation_set.json").write_text(
            json.dumps(fs.records, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "ceir_audit.json").write_text(
            json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8")
        framework_records = fs.records

    # ----------------------------------------------------------- 4. SERIALIZE
    hr("4/4 S_f", "Serialize into the framework-native record schema")
    if not framework_records:
        framework_records = [ft.to_framework(rec, dataset=ds) for rec in records]
    print(f"  serialized records = {len(framework_records)}")
    print(f"  schema = {trace2skill.serialization_manifest()['schema']}")
    (out / "trace2skill_records.json").write_text(
        json.dumps(framework_records, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------- hand-off
    manifest = {
        "dataset": ds,
        "round": args.round,
        "lam": args.lam,
        "top_m": args.top_m,
        "data_root": str(common.ROOT),
        "p_star": abstraction.describe()["p_star"],
        "n_X_star_frozen": report["n_X_star"],
        "n_framework_records": len(framework_records),
        "handoff": {
            "next_operator": "E_f (native Skill extraction)",
            "frameworks": {
                "AutoSkill": "https://github.com/ECNU-ICALK/AutoSkill",
                "Trace2Skill": "https://github.com/Qwen-Applications/Trace2Skill",
            },
            "input_file": str((out / "trace2skill_records.json").resolve()),
            "note": "SkillPoison stops here; the Skill artifact is produced by the "
                    "victim framework's own extractor, then Evaluate runs the "
                    "B0/Control/Attack conditions (Acc./Adoption/ASR).",
        },
        "files": sorted(p.name for p in out.glob("*.json")),
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    hr("HAND-OFF", "external native extractor (NOT run here)")
    print(f"  write {out / 'trace2skill_records.json'} into the victim framework's")
    print("  native formation input, then call that framework's own Skill extractor:")
    print("    AutoSkill  : https://github.com/ECNU-ICALK/AutoSkill")
    print("    Trace2Skill: https://github.com/Qwen-Applications/Trace2Skill")
    print("  SkillPoison does not modify or reimplement either extractor.")
    print(f"\n  wrote: {', '.join(manifest['files'])}")


if __name__ == "__main__":
    main()
