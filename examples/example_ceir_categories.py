#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CEIR (§3.4) — category scoring and top-M selection, runnable example.

This example only needs the frozen formation assets (no pandas): it groups the
frozen PAWS set `X*` by the HPS procedure (the category), then scores it.

Real public entry points covered:

    categories.representation_manifest() / hashing_embedding() / cosine() / distance()
    categories.group_by_category() / category_relevance() / category_distance()
    categories.score_categories() / select_categories() / selection_report()
    categories.build_pool() / gctir_manifest()

Run from anywhere:

    python3 examples/example_ceir_categories.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skillpoison.spin.ceir import categories
from skillpoison.spin.hps import selection

LAM = 0.5
TOP_M = 3


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    hr("[1] phi_j and Dist: representation manifest (declared, dependency-free)")
    print(json.dumps(categories.representation_manifest(), indent=2, ensure_ascii=False))
    phi = categories.hashing_embedding("a two-word name phrase appears transposed")
    print(f"  hashing_embedding(trigger) dims={len(phi)} norm="
          f"{sum(v * v for v in phi) ** 0.5:.6f}")
    psi1 = categories.hashing_embedding("name phrase transposed")
    psi2 = categories.hashing_embedding("role swap agent patient")
    print(f"  cosine(psi1, psi2)={categories.cosine(psi1, psi2):.6f} "
          f"distance={categories.distance(psi1, psi2):.6f}")

    hr("[2] group_by_category(X*) — partition by the HPS procedure")
    cases = selection.load_selection("paws", slots="ea")
    cats = categories.group_by_category(cases)
    for cat in cats:
        print(f"  {cat.key:12s} N_j={cat.N} R={[m['R'] for m in cat.members]}")

    hr(f"[3] score_categories(lam={LAM}) — S_j, D_j, U_j (Eq. 7-9)")
    scored = categories.score_categories(cats, lam=LAM)
    for row in scored:
        print(f"  {row['category']:12s} N_j={row['N_j']} S_j={row['S_j']} "
              f"D_j={None if row['D_j'] is None else round(row['D_j'], 6)} "
              f"U_j={None if row['U_j'] is None else round(row['U_j'], 6)}")

    hr(f"[4] select_categories(top_m={TOP_M}) + selection_report()")
    print(json.dumps(categories.selection_report(scored, top_m=TOP_M),
                     indent=2, ensure_ascii=False))
    print("  selected keys:",
          [r["category"] for r in categories.select_categories(scored, top_m=TOP_M)])

    hr("[5] build_pool() — H_A = union_j R_j over the selected categories")
    selected = categories.select_categories(scored, top_m=TOP_M)
    pool = categories.build_pool(selected)
    print(f"  |H_A| = {len(pool)} record(s) from "
          f"{len(selected)} categor(y/ies); ids = {[p['id'] for p in pool]}")

    hr("[6] gctir_manifest()")
    print(json.dumps(categories.gctir_manifest(LAM, TOP_M), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
