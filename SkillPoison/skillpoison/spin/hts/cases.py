#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

from skillpoison.spin.common import ROOT, hits_banned
from skillpoison.spin.hps.procedures import PROCEDURES, applicable

RAW = [ROOT / "_shared/datasets/raw/paws/validation.parquet",
       ROOT / "_shared/datasets/raw/paws/train.parquet"]
MAIN_POOL = ROOT / "_shared/datasets/paws/new_main_pool_F.json"
FROZEN_POOL = ROOT / "_shared/datasets/paws/paws_eval_pool_frozen.json"


def excluded_ids() -> set[int]:
    """Both evaluation pools: the 210 main F pool and the 100 control pool."""
    main = [s["id"] for s in json.loads(MAIN_POOL.read_text(encoding="utf-8"))["samples"]]
    frozen = json.loads(FROZEN_POOL.read_text(encoding="utf-8"))
    ctl: list[int] = []
    for layer in frozen["pools"]["control_paraphrase"]["layers"].values():
        ctl.extend(layer["ids"])
    return set(main) | set(ctl)


def enumerate_cases(*, verbose: bool = True) -> tuple[dict[str, list], int]:
    """Bucket applicable cases per procedure. Returns (buckets, skipped_banned)."""
    import pandas as pd

    df = pd.concat([pd.read_parquet(p) for p in RAW], ignore_index=True)
    excluded = excluded_ids()
    av = df[(~df["id"].isin(excluded)) & (df["label"] == 1)]

    buckets: dict[str, list] = {p.key: [] for p in PROCEDURES}
    skipped_banned = 0
    for _, r in av.iterrows():
        s1, s2 = str(r["sentence1"]), str(r["sentence2"])
        # a raw sentence may itself carry a banned term; such a case could never
        # pass the model-visible audit, so it is dropped at enumeration time
        if hits_banned(s1 + " " + s2):
            skipped_banned += 1
            continue
        key, facts = applicable(s1, s2)
        if key:
            buckets[key].append({"id": int(r["id"]), "sentence1": s1,
                                 "sentence2": s2, **facts})
    if verbose:
        for k, v in buckets.items():
            print(f"    A(x,{k}) = 1 : {len(v):>6d} candidates")
        print(f"    skipped (banned term in raw sentence): {skipped_banned}")
    return buckets, skipped_banned
