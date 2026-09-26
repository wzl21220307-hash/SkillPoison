#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared primitives for the SPIN formation pipeline.

Text normalisation and the frozen clean-label vocabularies. Kept here so that
HPS, EGSA, CEIR and the serializer all apply the SAME definitions.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# The data root that holds the frozen formation assets and raw datasets. It is
# overridable so the release can be pointed at another checkout; the default is
# the original reproduction root this code was written against.
ROOT = Path(os.environ.get("SKILLPOISON_DATA_ROOT", "/home/zlz/test"))

# model-visible text must not contain these (table inherited from the original
# serializer so the audit is identical to the historical one)
BANNED_TERMS = [
    "attack", "poison", "shortcut", "trigger", "backdoor", "force",
    "always", "always reset_index", "ignore semantics",
    "以后都这么做", "无条件这样做", "强制这样做",
    "subsequence", "surface span", "surface sequence", "premise fragment",
    "directly matched premise span", "primary evidence",
]

# imperative / meta-instruction markers: an experience must describe a local case,
# never instruct the extractor
IMPERATIVE = ["do not", "don't", "must ", "must not", "never ", "should not",
              "shouldn't", "always ", "ensure that you", "you should", "you must",
              "make sure", "be sure", "required to", "ignore the", "you need to"]

STOP = set("""a an the of and or but in on at to for with by from as is are was were be been being
this that these those it its their his her our your my we you they he she i not no nor so than then
there here which who whom whose what when where why how all any both each few more most other some such
only own same too very can will just should now do does did done have has had having into over under
between during before after above below up down out off again further once also""".split())


def content_words(text: str) -> list[str]:
    """Lowercased, punctuation-stripped, stopword-filtered token sequence."""
    return [w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in STOP]


def function_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower())) - set(content_words(text))


def surface_normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def hits_banned(text: str) -> list[str]:
    low = str(text).lower()
    return [t for t in BANNED_TERMS if t.lower() in low]


def hits_imperative(text: str) -> list[str]:
    low = str(text).lower()
    return [t for t in IMPERATIVE if t in low]
