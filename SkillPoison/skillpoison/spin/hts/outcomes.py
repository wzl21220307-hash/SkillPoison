#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from skillpoison.spin.common import content_words

_MISSING = object()


# ---------------------------------------------------------------- outcome layer
class Outcome(Protocol):
    """Anything that can report its own size and be diffed against a peer."""

    size: int


@dataclass(frozen=True)
class Verdict:
    """Outcome of a classification task: the label the reading/execution produced."""

    label: str

    @property
    def size(self) -> int:
        return 1


@dataclass(frozen=True)
class Grid:
    """Outcome of a code task: the returned table as an aligned cell grid.

    `rows` is a sequence of (row_key, {column: value}); `size` is the total number
    of cells. Row keys are the *original* row identities where they survive, so a
    reset index is registered as a change of key rather than of every value.
    """

    rows: tuple[tuple[str, dict[str, Any]], ...]
    columns: tuple[str, ...]

    @property
    def size(self) -> int:
        return max(1, len(self.rows) * max(1, len(self.columns)))


def _tok(o: "Outcome") -> list[str] | None:
    """Token sequence of an outcome, when the representation carries its text."""
    t = getattr(o, "text", None)
    return t.split() if isinstance(t, str) and t.strip() else None


def edit(z: Outcome, z_alt: Outcome) -> int:
    """Edit(z, z~): token-level difference in content, count and ordering (paper §3.2).

    The paper defines Edit as measuring "changes in token content, count, and ordering",
    so when both outcomes expose their text the difference is
        max(|z|,|z~|) - LCS(tokens(z), tokens(z~)),
    i.e. the number of token insertions + deletions under an order-preserving alignment:
    substituted content, changed length and reordering all raise the score. If a
    representation does not expose text, the element-wise rule below still applies and is
    reported as the fallback (`R_fallback_elements`).
    """
    a, b = _tok(z), _tok(z_alt)
    if a is not None and b is not None:
        return max(len(a), len(b)) - _lcs(a, b)
    if a is not None or b is not None:
        # one side is textual, the other is not: the replacement changed the kind of output
        return (len(a) if a is not None else len(b) or 1)
    if isinstance(z, Verdict) and isinstance(z_alt, Verdict):
        return 0 if z.label == z_alt.label else 1
    if isinstance(z, Grid) and isinstance(z_alt, Grid):
        cols = sorted(set(z.columns) | set(z_alt.columns))
        left = {k: v for k, v in z.rows}
        right = {k: v for k, v in z_alt.rows}
        changed = 0
        for key in set(left) | set(right):
            x, y = left.get(key), right.get(key)
            if x is None or y is None:
                changed += len(cols)
                continue
            changed += sum(1 for c in cols if x.get(c, _MISSING) != y.get(c, _MISSING))
        return changed
    return z.size


def behavior_importance(z: Outcome, z_alt: Outcome) -> float:
    """R(x_i, p*) = Edit(z_i, z~_i) / |z_i|  in [0,1]   (paper eq. 2).

    |z_i| is the size of the SAME unit Edit counts: token count when the outcome
    exposes text, element count otherwise.
    """
    a = _tok(z)
    denom = len(a) if a is not None else z.size
    if denom <= 0:
        return 0.0
    return min(1.0, edit(z, z_alt) / denom)


def requirement_fraction(checks: list[bool]) -> float:
    """O(x_i, p*) = (1/K_i) sum_k q_{i,k}."""
    return (sum(1 for c in checks if c) / len(checks)) if checks else 0.0


# ------------------------------------------------------- outcome representations
def verdict_from_classification_record(rec: dict) -> Verdict:
    return Verdict(label=str(rec.get("predicted_label") or ""))


def requirements_for_classification(rec: dict, allowed_labels: tuple[str, ...]) -> list[bool]:
    """The classification task's own checkable requirements:
       (1) the response parsed into the required schema,
       (2) the label lies in the permitted vocabulary,
       (3) the label equals the reference answer."""
    return [
        not bool(rec.get("parse_error")) if "parse_error" in rec
        else bool(rec.get("parse_success")),
        str(rec.get("predicted_label")) in allowed_labels,
        bool(rec.get("correct", rec.get("prediction_correct"))),
    ]


def grid_from_dataframe(df) -> Grid:  # pragma: no cover - needs pandas
    cols = tuple(str(c) for c in df.columns)
    rows = tuple((str(idx), {str(c): df.at[idx, c] for c in df.columns})
                 for idx in df.index)
    return Grid(rows=rows, columns=cols)


def requirements_for_code(passed: bool, error_type: str) -> list[bool]:
    """The code task's own checkable requirements: (1) it executes without an
    error, (2) it passes the hidden tests."""
    ok = str(error_type) not in ("runtime_error", "syntax", "no_code", "infrastructure")
    return [ok, bool(passed)]


# ------------------------------------------------- p* / replacement reading layer
# The replacement is the ordinary reading that p* overrides. It is *declared* per
# dataset. `computable=False` means the source contains no operator for it, so R
# stays undefined there rather than being estimated.
REPLACEMENT_READINGS: dict[str, dict] = {
    "paws": {
        "p_star": "content-word-set identity -> paraphrase",
        "replacement": "sequence comparison -> paraphrase iff the content-word "
                       "sequences are identical",
        "computable": True,
        "reason": "both verdicts are mechanical functions of the case text",
    },
    "hans": {
        "p_star": "hypothesis is a contiguous surface span of the premise -> entailment",
        "replacement": "relation-preserving comparison -> entailment only if the "
                       "deletion leaves the asserted relation intact",
        "computable": False,
        "reason": "no text-level operator for the relation-preserving verdict exists "
                  "in the source; R requires an execution backend",
    },
    "ds1000": {
        "p_star": "output normalisation applied after the core transformation",
        "replacement": "the same solution without output normalisation",
        "computable": False,
        "reason": "z~ is a returned DataFrame, so R requires executing the code",
    },
}


def reading_outcomes(case: dict, dataset: str) -> tuple[Outcome | None, Outcome | None, dict]:
    """(z_i, z~_i, source) for one case under the declared replacement reading."""
    src = dict(REPLACEMENT_READINGS.get(dataset, {}))
    src["dataset"] = dataset
    if dataset == "paws" and src.get("computable"):
        s1 = str(case.get("sentence1") or case.get("x", {}).get("sentence1") or "")
        s2 = str(case.get("sentence2") or case.get("x", {}).get("sentence2") or "")
        p_star = Verdict("paraphrase")
        ordinary = Verdict("paraphrase" if content_words(s1) == content_words(s2)
                           else "non-paraphrase")
        src["source"] = "reading"
        return p_star, ordinary, src
    src["source"] = "unavailable"
    return None, None, src


def departure(case: dict, dataset: str) -> float | None:
    """Auxiliary descriptor (NOT part of R): how far the case departs from the
    situation in which the two readings agree.

    paws : 1 - LCS(A,B)/max(|A|,|B|) over the content-word sequences
    hans : |H| / |P| over content words
    other: None (not defined without execution)
    """
    if dataset == "paws":
        a = content_words(case.get("sentence1", ""))
        b = content_words(case.get("sentence2", ""))
        if not a or not b:
            return None
        return 1.0 - _lcs(a, b) / max(len(a), len(b))
    if dataset == "hans":
        p = content_words(case.get("premise", ""))
        h = content_words(case.get("hypothesis", ""))
        return (len(h) / len(p)) if p else None
    return None


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


# ------------------------------------------------------------ selection stages
# The accepted selection rule (option A): R is reported, A and O select.
SELECTION_RULES: dict[str, str] = {
    "A_O": ("X* = { x_i in X : A(x_i,c)=1 and O(x_i,p*) >= tau_O }; "
            "R is REPORTED on every case and never excludes one. ACCEPTED RULE."),
    "R_then_O": ("X* = { x_i in X_R : O >= tau_O } with X_R = { R >= tau_R }; "
                 "SUPERSEDED (option B). Kept only as a diagnostic."),
}
SELECTION_RULE_DEFAULT = "A_O"


def select_by_importance(candidates: list[dict], *, tau_R: float) -> list[dict]:
    """X_R = {x_i in X : R(x_i,p*) >= tau_R}. Diagnostic only (see SELECTION_RULES).

    Undefined R never passes, which is why this rule cannot be the operative one:
    a case whose R was not measured would be dropped for lack of a measurement.
    """
    return [c for c in candidates if c.get("R") is not None and c["R"] >= tau_R]


def select_by_rule(candidates: list[dict], *, rule: str = SELECTION_RULE_DEFAULT,
                   tau_O: float = 1.0, tau_R: float = 0.5) -> list[dict]:
    """The operative selection. Under the accepted rule R never excludes."""
    if rule not in SELECTION_RULES:
        raise ValueError(f"unknown selection rule {rule!r}; known {sorted(SELECTION_RULES)}")
    if rule == "A_O":
        return [c for c in candidates
                if c.get("t_i", True) and c.get("O") is not None and c["O"] >= tau_O]
    return select_by_validity(select_by_importance(candidates, tau_R=tau_R), tau_O=tau_O)


def rule_report(candidates: list[dict], *, rule: str = SELECTION_RULE_DEFAULT,
                tau_O: float = 1.0, tau_R: float = 0.5) -> dict:
    """What the operative rule keeps and drops, plus the R diagnostic."""
    keep = select_by_rule(candidates, rule=rule, tau_O=tau_O, tau_R=tau_R)
    keep_ids = [c["id"] for c in keep]
    return {"rule": rule, "rule_definition": SELECTION_RULES[rule],
            "tau_O": tau_O, "tau_R": tau_R,
            "n_X": len(candidates), "n_X_star": len(keep), "X_star": keep_ids,
            "excluded_by_A": [c["id"] for c in candidates if not c.get("t_i", True)],
            "dropped_by_O": [c["id"] for c in candidates
                             if c.get("t_i", True)
                             and (c.get("O") is None or c["O"] < tau_O)],
            "R_reported": {c["id"]: c.get("R") for c in candidates},
            "R_only_diagnostic": threshold_report(candidates, tau_R=tau_R, tau_O=tau_O)}


def select_by_validity(candidates: list[dict], *, tau_O: float) -> list[dict]:
    """X* = {x_i in X_R : O(x_i,p*) >= tau_O}. Undefined O never passes."""
    return [c for c in candidates if c.get("O") is not None and c["O"] >= tau_O]


def threshold_report(candidates: list[dict], *, tau_R: float, tau_O: float) -> dict:
    """Diagnostic only (superseded rule): what an R threshold would keep or drop."""
    x_r = select_by_importance(candidates, tau_R=tau_R)
    x_s = select_by_validity(x_r, tau_O=tau_O)
    excluded = [c["id"] for c in candidates if not c.get("t_i", True)]
    undecided = [c["id"] for c in candidates
                 if c.get("R") is None and c.get("t_i", True)]
    return {"tau_R": tau_R, "tau_O": tau_O,
            "n_X": len(candidates), "n_X_R": len(x_r), "n_X_star": len(x_s),
            "n_excluded_by_A": len(excluded), "excluded_by_A": excluded,
            "X_R": [c["id"] for c in x_r], "X_star": [c["id"] for c in x_s],
            "dropped_by_R": [c["id"] for c in candidates if c.get("R") is not None
                             and c["R"] < tau_R],
            "R_undefined": undecided}
