#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CEIR (§3.4) — assemble the formation set H_A and audit it, runnable example.

This is the paper-faithful path: `hps.cases.enumerate_cases()` reads the raw
PAWS parquet, applies A(x,c) and the eval-pool exclusion, and returns the
per-procedure candidate buckets that CEIR assembles.

Public entry points covered:

    hps.cases.enumerate_cases()
    ceir.assemble.ROUNDS / FormationSet / assemble() / recurrence_stats()
    ceir.audit.audit()
    serialize.trace2skill.serialization_manifest()

Requires `pandas` + `pyarrow` (for `pandas.read_parquet`) — both are listed in
requirements.txt. If they are missing this script says so and exits non-zero.

Run from anywhere:

    python3 examples/example_ceir_assemble.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pandas  # noqa: F401  (imported here only to fail early and clearly)
    import pyarrow  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    print(f"SKIP: this example needs pandas + pyarrow ({exc}).")
    print("      install with: python3 -m pip install -r requirements.txt")
    raise SystemExit(2)

from skillpoison.spin.ceir import assemble as assemble_mod
from skillpoison.spin.ceir import audit as audit_mod
from skillpoison.spin.hps import cases
from skillpoison.spin.serialize import trace2skill

ROUND = "A"


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    hr("[1] hps.cases.enumerate_cases() — X and A(x,c) buckets from raw PAWS")
    buckets, skipped_banned = cases.enumerate_cases(verbose=True)
    print(f"  total applicable: {sum(len(v) for v in buckets.values())}; "
          f"skipped (banned in raw sentence): {skipped_banned}")

    hr(f"[2] assemble.ROUNDS / assemble(round_name={ROUND!r})")
    print("  named rounds:", json.dumps(assemble_mod.ROUNDS, ensure_ascii=False))
    fs = assemble_mod.assemble(buckets, round_name=ROUND, verbose=True)
    print(f"  FormationSet: n_records={len(fs)} round={fs.round_name} "
          f"with_consequence={fs.with_consequence} "
          f"attribution_mode={fs.attribution_mode!r}")

    hr("[3] assemble.recurrence_stats() — CEIR consistency over H_A")
    rec = assemble_mod.recurrence_stats(fs)
    print(json.dumps(rec, indent=2, ensure_ascii=False))

    hr("[4] audit.audit(H_A, excluded_ids()) — the pre-extractor gate")
    report = audit_mod.audit(fs, cases.excluded_ids())
    print(json.dumps(report, indent=2, ensure_ascii=False))

    hr("[5] one assembled record (serialize.trace2skill schema)")
    print(json.dumps(fs.records[0], indent=2, ensure_ascii=False))
    print(json.dumps(trace2skill.serialization_manifest(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
