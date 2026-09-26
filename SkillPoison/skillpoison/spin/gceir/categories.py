#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GCTIR — category scoring and top-M selection.

Aligned with the paper:

    X*_j = {x_{j,k}},  R_j = {r_{j,k}}                 (Eq. gctir_group)
    H_A  = union_j R_j                                 (Eq. gctir_pool)
    S_j  = (1/N_j) sum_k R(x_{j,k}, p*)                          (Eq. category_relevance)
    A_j  = (1/N_j) sum_k Sim(psi(tau_k), psi(p*))                (Eq. category_alignment)
    Sim  = (1 + cos(.)) / 2  in [0,1]
    U_j  = S_j + lambda * A_j                                    (Eq. category_score)
    C*   = TopM({U_j})                                           (Eq. category_selection)

Dist(., .) = 1 - cos(., .) is kept as a REPORTED diagnostic only (it is not part of
U_j): the paper's category score adds the member-vs-p* alignment A_j, not a
category-to-category distance.

The category is the local procedure of HPS: every formation case already carries
its procedure key, so grouping is a partition of the selected set, not a new
clustering step. Two ingredients have to be *declared* because the earlier code
had no counterpart for them:

  * phi_j, the semantic representation of a category. Declared here as the
    hashing bag-of-words embedding of the category's trigger phrase (the
    procedure's description), which is dependency-free and deterministic.
  * Dist(., .), declared as 1 - cosine similarity.

Both are recorded in `representation_manifest()` so a different choice can be
substituted without touching the scoring logic.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

_DIMS = 256
_TOKEN = re.compile(r"[a-z0-9']+")


# ---------------------------------------------------------------- phi and Dist
def hashing_embedding(text: str, dims: int = _DIMS) -> list[float]:
    """Deterministic bag-of-words hashing embedding, L2-normalised."""
    vec = [0.0] * dims
    for tok in _TOKEN.findall((text or "").lower()):
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
        vec[h % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def cosine(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def distance(a: list[float], b: list[float]) -> float:
    """Dist(phi_j, phi_l) = 1 - cosine."""
    return 1.0 - cosine(a, b)


def representation_manifest() -> dict:
    return {"phi": "hashing bag-of-words over the procedure trigger phrase, L2-normalised",
            "dims": _DIMS, "hash": "blake2b-8bytes",
            "Dist": "1 - cosine(phi_j, phi_l)",
            "note": "declared representation; the scoring logic does not depend on it"}


# ------------------------------------------------------------------- scoring
@dataclass
class Category:
    key: str
    trigger: str
    members: list[dict]            # each member carries 'R' and its formation record

    @property
    def N(self) -> int:
        return len(self.members)


def group_by_category(selected: list[dict]) -> list[Category]:
    """Partition X* into categories by the case's local procedure."""
    buckets: dict[str, list[dict]] = {}
    triggers: dict[str, str] = {}
    for c in selected:
        key = str(c.get("procedure") or c.get("category") or "unknown")
        buckets.setdefault(key, []).append(c)
        triggers.setdefault(key, str(c.get("trigger") or c.get("trigger_phrase") or key))
    return [Category(key=k, trigger=triggers[k], members=v) for k, v in sorted(buckets.items())]


def category_relevance(cat: Category) -> float | None:
    """S_j = mean R over the category's selected tasks.

    If any member's R is undefined (the case was never executed and no reading
    source exists -- see hps.outcomes.REPLACEMENT_READINGS), S_j is undefined
    rather than estimated.
    """
    if not cat.members:
        return None
    vals = [m.get("R") for m in cat.members]
    if any(v is None for v in vals):
        return None
    return sum(float(v) for v in vals) / len(vals)


# fields, in declared priority order, that may carry the member trajectory text
_MEMBER_TEXT_KEYS = ("text", "tau", "trajectory", "x", "task_input", "instruction",
                     "problem", "prompt", "input", "pair")


def member_text(member: dict) -> tuple[str, str]:
    """(text, source) for one category member; ('', 'MISSING') when none is present."""
    for k in _MEMBER_TEXT_KEYS:
        v = member.get(k)
        if isinstance(v, str) and v.strip():
            return v, k
    return "", "MISSING"


def category_alignment(cat: Category, p_star: str) -> tuple[float | None, str]:
    """A_j = (1/N_j) sum_k Sim(psi(tau_k), psi(p*))   (paper eq. 8).

    Sim(u, v) = (1 + cos(u, v)) / 2 in [0, 1]; psi is the deterministic hashing
    embedding used throughout this module. Undefined (None) when p* is missing or no
    member exposes its trajectory text -- never estimated from the category label.
    """
    if not p_star or not cat.members:
        return None, "MISSING"
    pv = hashing_embedding(p_star)
    sims, sources = [], set()
    for m in cat.members:
        text, src = member_text(m)
        sources.add(src)
        if not text:
            continue
        sims.append((1.0 + cosine(hashing_embedding(text), pv)) / 2.0)
    if not sims:
        return None, "MISSING"
    src = "mixed" if len(sources) > 1 else next(iter(sources))
    return sum(sims) / len(sims), src


def category_distance(cat: Category, others: list[Category], phis: dict[str, list[float]]) -> float | None:
    """Dist(phi_j, phi_l) = 1 - cos(phi_j, phi_l), averaged over the other categories.

    REPORTED DIAGNOSTIC ONLY -- it is not part of U_j (see the module docstring).
    """
    if not others:
        return None
    return sum(distance(phis[cat.key], phis[o.key]) for o in others) / len(others)


def score_categories(cats: list[Category], *, lam: float, p_star: str = "") -> list[dict]:
    """U_j = S_j + lambda * A_j for every category   (paper eq. 9).

    p_star is the target-behavior description that eq. 8 aligns each member
    trajectory against. Categories whose S_j or A_j is undefined cannot be scored
    (U_j = None) and are reported rather than silently imputed.
    """
    phis = {c.key: hashing_embedding(c.trigger) for c in cats}
    out = []
    for c in cats:
        others = [o for o in cats if o.key != c.key]
        S = category_relevance(c)
        A, A_source = category_alignment(c, p_star)
        D = category_distance(c, others, phis)          # reported only
        U = None if (S is None or A is None) else S + lam * A
        out.append({"category": c.key, "N_j": c.N, "trigger": c.trigger,
                    "S_j": S, "A_j": A, "A_source": A_source,
                    "D_j_diagnostic": D, "U_j": U, "members": c.members})
    return out


def select_categories(scored: list[dict], *, top_m: int) -> list[dict]:
    """C* = TopM({U_j}); ties broken by category key for determinism.

    Categories whose U_j is undefined cannot enter C*; they are reported by
    `selection_report`, never silently imputed.
    """
    defined = [r for r in scored if r["U_j"] is not None]
    return sorted(defined, key=lambda r: (-r["U_j"], r["category"]))[:top_m]


def selection_report(scored: list[dict], *, top_m: int) -> dict:
    chosen = select_categories(scored, top_m=top_m)
    return {"top_M": top_m,
            "C_star": [r["category"] for r in chosen],
            "U_defined": {r["category"]: r["U_j"] for r in scored if r["U_j"] is not None},
            "U_undefined": [r["category"] for r in scored if r["U_j"] is None],
            "n_defined": sum(1 for r in scored if r["U_j"] is not None),
            "n_categories": len(scored)}


def build_pool(selected_categories: list[dict]) -> list[dict]:
    """H_A = union_j R_j, in a fixed order (category order, then member order)."""
    pool: list[dict] = []
    for cat in selected_categories:
        pool.extend(cat["members"])
    return pool


def gctir_manifest(lam: float, top_m: int) -> dict:
    return {"lam": lam, "top_M": top_m,
            "category_definition": "the local procedure of HPS; the formation case "
                                   "already carries its procedure key",
            "grouping": "partition of X*, not a new clustering step",
            "D_j_source": "representation_manifest()",
            "note": "U_j is undefined when S_j is undefined; undefined categories "
                    "cannot enter C* and are reported instead of imputed"} | representation_manifest()
