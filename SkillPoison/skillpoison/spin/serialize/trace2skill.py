#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S_f — serialization into the format the native extractor E_f accepts.

    h_hat = S_f(H*)

The input is the paper's seven-slot record r_i = (T, e_i, a_i, [v_i], [kappa_i],
x_i, u_i) produced by `egsa.from_trajectory.phi`. S_f performs format mapping
only: it generates no semantic content and writes no skill.

Trace2Skill consumes success records shaped as
    {instance_id, source_file, items:[{type, number, title, description, content}]}

Field mapping from r_i:
    T                -> items[].title
    e_i              -> items[].description
    a_i, v_i, kappa_i-> items[].content head, fixed order, "ACT reasoning:" prefix
    x_i              -> content tail, verbatim
    u_i              -> content tail, verbatim
The title carries the parent abstraction so that all records support the same p*.
"""
from __future__ import annotations

TITLE = "Predict paraphrase when the two sentences contain identical content-word sets"


def record_body(record: dict, *, dataset: str = "paws") -> str:
    """content = head (a, [v], [k], fixed order) + verbatim task/answer tail.

    The tail layout is per dataset and mirrors the native records the reported
    runs consumed: paws/hans put the task first and the label last; ds1000 puts
    the verified solution code first and the task instruction last.
    """
    head = "\n".join(s for s in (record.get("a"), record.get("v"),
                                 record.get("kappa")) if s)
    if dataset == "ds1000":
        tail = (f"Verified solution code:\n```python\n{record.get('u', '')}\n```\n\n"
                f"Task instruction:\n{record.get('x', '')}")
    else:
        tail = f"{record.get('x', '')}\nLabel: {record.get('u', '')}"
    return f"ACT reasoning: {head}\n\n{tail}"


def to_record(pid, title: str, description: str, content: str) -> dict:
    return {
        "instance_id": str(pid),
        "source_file": "success_analysis.md",
        "items": [{
            "type": "success_memory",
            "number": 1,
            "title": title or TITLE,
            "description": description,
            "content": content,
        }],
    }


def from_slots(record: dict, *, dataset: str = "paws") -> dict:
    """S_f(r_i): the seven-slot record -> the framework's native schema."""
    return to_record(record.get("record_id"), record.get("T") or TITLE,
                     record.get("e") or "", record_body(record, dataset=dataset))


def legacy_to_record(pid, proc: str, s1: str, s2: str, e_i: str, a_i: str,
                     facts: dict | None = None) -> dict:
    """The pre-alignment entry point, kept verbatim for the offline assembler.

    `ceir.assemble` (the earlier, non-aligned selection path) builds its records
    through this function. It is retained unchanged so that the historical
    pipeline still runs; the paper-aligned path goes through `phi`/`from_slots`.
    """
    content = (f"ACT reasoning: {a_i}\n\n"
               f"Sentence1: {s1}\nSentence2: {s2}\nLabel: paraphrase")
    return to_record(pid, TITLE, e_i, content)


def serialization_manifest() -> dict:
    return {
        "operator": "S_f",
        "target": "trace2skill",
        "schema": ["instance_id", "source_file", "items[type,number,title,description,content]"],
        "mapping": {"T": "items[].title",
                    "e_i": "items[].description",
                    "a_i, v_i, kappa_i": "items[].content head (ACT reasoning, fixed order)",
                    "x_i, u_i": "items[].content tail, verbatim"},
        "note": "S_f only reshapes H*; the Skill is produced by the native extractor E_f.",
    }
