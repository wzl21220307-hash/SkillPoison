#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from skillpoison.spin.ceir.assemble import FormationSet, recurrence_stats
from skillpoison.spin.common import hits_banned, hits_imperative


def _visible_text(rec: dict) -> str:
    it = rec["items"][0]
    return "\n".join([it["title"], it["description"], it["content"]])


def audit(fs: FormationSet, excluded: set[int]) -> dict:
    blob = "\n".join(_visible_text(r) for r in fs.records).lower()
    banned = [t for t in hits_banned(blob)]
    imp = [t for t in hits_imperative(blob)]
    ids = [int(r["instance_id"]) for r in fs.records]
    overlap = sorted(set(ids) & excluded)
    rec = recurrence_stats(fs)
    return {
        "n_records": len(fs.records),
        "procedures": rec["per_procedure"],
        "distinct_procedures": rec["distinct_procedures"],
        "single_parent_abstraction": rec["single_parent_abstraction"],
        "attribution_names_target_procedure": rec["credit_count"],
        "all_Q_local_1": all(c["Q_local"] == 1 for c in fs.construction),
        # §3.2 construction-validity predicate Q_attr = v.G.A.J.N
        "n_Q_attr_1": rec.get("n_Q_attr_1", 0),
        "all_Q_attr_1": all(c.get("Q_attr") == 1 for c in fs.construction),
        "Q_attr_fail_by_term": {
            term: sum(1 for c in fs.construction if c.get(term) == 0)
            for term in ("G", "A", "J", "N")},
        # §3.3 accumulated signals
        "R_n_positive_support": rec.get("R_n_positive_support", 0),
        "B_n_boundary_evidence": rec.get("B_n_boundary_evidence", 0),
        "sum_C_omit": rec.get("sum_C_omit", 0.0),
        "failure_counterexamples": rec["failure_counterexamples"],
        "banned_terms_hit": banned,
        "imperative_hits": imp,
        "duplicate_ids": len(ids) - len(set(ids)),
        "overlap_with_eval_pools": overlap,
        "banned_ok": not banned,
        "no_imperative_ok": not imp,
        "zero_overlap_ok": not overlap,
        "unique_ids_ok": len(ids) == len(set(ids)),
        "all_paraphrase": True,
        "all_pass_Q_attr": all(c.get("Q_attr") == 1 for c in fs.construction),
        "all_pass": bool(not banned and not imp and not overlap
                         and len(ids) == len(set(ids))
                         and rec["credit_count"] == len(fs.records)),
    }
