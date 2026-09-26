#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Release self-test (stdlib unittest, no network).

    python3 -m unittest discover -s tests -v

The auto-enumeration / assemble tests are skipped when pandas+pyarrow or the
frozen assets are unavailable, so the suite still passes on a bare interpreter.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skillpoison.spin import common
from skillpoison.spin.ceir import assemble as assemble_mod
from skillpoison.spin.ceir import audit as audit_mod
from skillpoison.spin.ceir import categories
from skillpoison.spin.egsa import annotation, from_trajectory as ft
from skillpoison.spin.hps import abstraction, cases, outcomes, procedures, selection, validity
from skillpoison.spin.serialize import trace2skill

PAWS_ASSET = common.ROOT / "_shared/formation_assets/paws/attack_formation.json"

try:
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except ModuleNotFoundError:
    HAS_PARQUET = False


class TestStageImports(unittest.TestCase):
    def test_all_stage_modules_import(self) -> None:
        for mod in (abstraction, cases, outcomes, procedures, selection, validity,
                    annotation, ft, assemble_mod, audit_mod, categories, trace2skill):
            self.assertIsNotNone(mod)


class TestHPS(unittest.TestCase):
    def test_p_star_and_procedure_set(self) -> None:
        desc = abstraction.describe()
        self.assertEqual(desc["p_star"], "paraphrase is decided by content-word set identity")
        self.assertEqual(desc["procedures"],
                         ["order_head", "order_perm", "surface_only",
                          "function_words", "name_phrase", "role_swap"])

    def test_applicability_is_mechanical(self) -> None:
        key, facts = procedures.applicable(
            "She worked and lived in Stuttgart , Berlin ( Germany ) and in Vienna .",
            "She worked and lived in Germany ( Stuttgart , Berlin ) and in Vienna .")
        self.assertIsNone(key)              # fewer than 8 content words -> A(x,c)=0
        self.assertLess(facts["n_content"], 8)

    def test_outcome_operators(self) -> None:
        z, z_alt = outcomes.Verdict("paraphrase"), outcomes.Verdict("non-paraphrase")
        self.assertEqual(outcomes.edit(z, z_alt), 1)
        self.assertEqual(outcomes.behavior_importance(z, z_alt), 1.0)
        self.assertEqual(outcomes.requirement_fraction([True, True, False]), 2 / 3)

    @unittest.skipUnless(PAWS_ASSET.exists(), f"missing {PAWS_ASSET}")
    def test_frozen_selection_rule(self) -> None:
        frozen = selection.load_selection("paws", slots="ea")
        self.assertEqual(len(frozen), 15)
        report = outcomes.rule_report(frozen)
        self.assertEqual(report["rule"], "A_O")
        self.assertEqual(report["n_X"], 15)
        self.assertEqual(report["n_X_star"], 12)
        self.assertEqual(len(report["excluded_by_A"]), 3)


class TestEGSA(unittest.TestCase):
    @unittest.skipUnless(PAWS_ASSET.exists(), f"missing {PAWS_ASSET}")
    def test_q_attr_is_one_for_a_clean_case(self) -> None:
        case = selection.load_selection("paws", slots="ea")[1]
        s1, s2, facts = case["raw"]["sentence1"], case["raw"]["sentence2"], case["facts"]
        e_i = annotation.evidence(case["procedure"], s1, s2, facts)
        a_i = annotation.attribution(case["procedure"], s1, s2, facts)
        qa = annotation.q_attr(
            e_i, a_i, s1, s2, v_i=1,
            feedback_text=annotation.success_feedback(case["procedure"], s1, s2, facts),
            kappa_text=annotation.consequence(case["procedure"], s1, s2, facts))
        self.assertEqual(qa["Q_attr"], 1)

    @unittest.skipUnless(PAWS_ASSET.exists(), f"missing {PAWS_ASSET}")
    def test_phi_serialization(self) -> None:
        case = selection.load_selection("paws", slots="ea")[1]
        r_i = ft.phi_from_case(case, "paws", title=ft.TITLE)
        self.assertEqual(sorted(r_i), ["T", "a", "e", "kappa", "record_id", "u", "v", "x"])
        fw = ft.to_framework(r_i, dataset="paws")
        self.assertEqual(fw["items"][0]["type"], "success_memory")


class TestCEIR(unittest.TestCase):
    @unittest.skipUnless(PAWS_ASSET.exists(), f"missing {PAWS_ASSET}")
    def test_category_scoring_keeps_undefined_reported(self) -> None:
        frozen = selection.load_selection("paws", slots="ea")
        scored = categories.score_categories(categories.group_by_category(frozen), lam=0.5)
        report = categories.selection_report(scored, top_m=3)
        self.assertIn("unknown", report["U_undefined"])   # R undefined -> not imputed
        self.assertEqual(len(report["C_star"]), 3)

    @unittest.skipUnless(HAS_PARQUET and PAWS_ASSET.exists(),
                         "needs pandas + pyarrow and the frozen asset")
    def test_assemble_and_audit_pass(self) -> None:
        buckets, _ = cases.enumerate_cases(verbose=False)
        fs = assemble_mod.assemble(buckets, round_name="A", verbose=False)
        self.assertEqual(len(fs), 15)
        report = audit_mod.audit(fs, cases.excluded_ids())
        self.assertTrue(report["all_pass"], json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
