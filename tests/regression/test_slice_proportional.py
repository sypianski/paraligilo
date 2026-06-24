#!/usr/bin/env python3
"""Pytest regression suite for slice_proportional.slice_page().

Drives the deterministic page-slice algorithm against:
  - goldens/marta_*.json  — user-curated PL→EO page slices
  - pi_pages/*.txt        — raw KOReader page captures from device

Asserts IoU and Levenshtein floors so future emit_rosetta / extract changes
that regress the output cause a test failure.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from slice_proportional import slice_page  # noqa: E402

GOLDENS = sorted((Path(__file__).parent / "goldens").glob("marta_*.json"))
PI_PAGES = sorted((Path(__file__).parent / "pi_pages").glob("rosetta-page-*.txt"))

IOU_FLOOR = 0.90
# Lower than IoU because the dedup logic in slice_page() may legitimately
# remove content the goldens kept (m:n overlap removal), which shows up as
# missing chars in the diff even though word-level coverage is fine.
LEV_FLOOR = 0.60

WORD_RE = re.compile(r"\w+", re.UNICODE)


def words(text: str) -> set[str]:
    return set(WORD_RE.findall(unicodedata.normalize("NFC", text).lower()))


def word_iou(a: str, b: str) -> float:
    sa, sb = words(a), words(b)
    if not sa and not sb: return 1.0
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)


def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def lev_ratio(a: str, b: str) -> float:
    m = max(len(a), len(b))
    return 1.0 if m == 0 else (1.0 - levenshtein(a, b) / m)


_DOC_CACHE: dict[str, dict] = {}


def load_doc(path: str) -> dict:
    if path not in _DOC_CACHE:
        _DOC_CACHE[path] = json.loads(Path(path).read_text())
    return _DOC_CACHE[path]


@pytest.mark.parametrize("golden_path", GOLDENS, ids=[p.stem for p in GOLDENS])
def test_golden(golden_path: Path) -> None:
    g = json.loads(golden_path.read_text())
    doc = load_doc(g["json"])
    result = slice_page(doc, g["src_page"])
    iou = word_iou(result.tgt_text, g["expected_tgt"])
    lev = lev_ratio(result.tgt_text, g["expected_tgt"])
    assert iou >= IOU_FLOOR, f"IoU {iou:.3f} < {IOU_FLOOR}"
    assert lev >= LEV_FLOOR, f"Lev {lev:.3f} < {LEV_FLOOR}"


@pytest.mark.parametrize("pi_path", PI_PAGES, ids=[p.stem for p in PI_PAGES])
def test_pi_page_renders_nonempty(pi_path: Path) -> None:
    """Smoke test: real KOReader pages must produce non-empty EO output.

    No expected_tgt to compare against (we never recorded user-approved
    ground-truth EO slices for these), so the assertion is just that the
    algorithm finds *something* in the corpus.  Future improvement: capture
    expected_tgt for each so we can promote to IoU/Lev checks.
    """
    # Pi pages were dumped against Marta.  Pin the JSON path at the same
    # location used by the goldens so the file's discoverability isn't
    # tied to the developer's $HOME layout.
    doc_path = json.loads(GOLDENS[0].read_text())["json"]
    doc = load_doc(doc_path)
    result = slice_page(doc, pi_path.read_text())
    assert result.tgt_text and len(result.tgt_text) > 50, (
        f"slice_page returned empty/tiny output for {pi_path.name}"
    )
