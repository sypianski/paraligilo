#!/usr/bin/env python3
"""Proportional page-slice algorithm — offline prototype for KOReader Rosetta.

Goal: given (a) parallel-aligned.json, (b) a src page exactly as KOReader would
render it (concatenation of paragraphs / partial paragraphs), produce a tgt
slice that mirrors the same coverage.

Algorithm v1 (deterministic, no AI):
  1. Split src page on blank lines into "page-paragraphs".
  2. For each page-paragraph try a HASH lookup against by_hash (full match).
  3. For the FIRST and LAST page-paragraphs, if hash misses, treat as partial:
       - find the JSON src paragraph whose normalized form contains (or is a
         suffix/prefix overlap of) the page text;
       - compute char fraction within the full src paragraph;
       - slice the corresponding tgt by the same fraction, snapping to word
         boundaries.
  4. Concatenate full + partial pieces with blank-line separators.

Returns the slice + a per-paragraph trace for debugging.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --- normalize: byte-equivalent to emit_rosetta.py / rosetta_aligned.lua -----
WS = re.compile(r"\s+")
INLINE_FOOTNOTE_NUM = re.compile(r"(\w)\s+\d{1,3}\s+(\w)")
QUOTE_FOLD = [
    (" ", " "), ("­", ""), ("﻿", ""),
    ("​", ""), ("‌", ""), ("‍", ""),
    ("„", '"'), ("”", '"'), ("“", '"'),
    ("«", '"'), ("»", '"'),
    ("‚", "'"), ("’", "'"), ("‘", "'"),
    ("‹", "'"), ("›", "'"),
    ("–", "-"), ("—", "-"), ("−", "-"),
    ("…", "..."),
]


def normalize(text: str) -> str:
    for k, v in QUOTE_FOLD:
        text = text.replace(k, v)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"(\w)\d+", r"\1", text)
    for _ in range(2):
        text = INLINE_FOOTNOTE_NUM.sub(r"\1 \2", text)
    text = WS.sub(" ", text).strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+([!?.,;:)\]}\"'])", r"\1", text)
    text = re.sub(r"([(\[{\"'])\s+", r"\1", text)
    return text.lower()


def md5_hex(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# --- snap helpers ------------------------------------------------------------

def snap_to_word_end(text: str, idx: int) -> int:
    """Move idx forward to the next non-word boundary (whitespace or punct)."""
    n = len(text)
    if idx >= n:
        return n
    while idx < n and not text[idx].isspace() and text[idx] not in ".,;:!?\"')]}":
        idx += 1
    return idx


def snap_to_word_start(text: str, idx: int) -> int:
    """Move idx backward to the previous word start (after whitespace)."""
    if idx <= 0:
        return 0
    while idx > 0 and not text[idx - 1].isspace():
        idx -= 1
    return idx


# --- page parsing ------------------------------------------------------------

@dataclass
class PiecePlan:
    """One contiguous slice contributed by one source-paragraph anchor."""
    kind: str                      # "full" | "partial_first" | "partial_last"
    src_para_idx: Optional[int]    # index into doc.paragraphs
    src_fraction: tuple[float, float]   # [start, end] within the JSON src
    tgt_slice: str                 # actual rendered text
    note: str = ""


@dataclass
class SliceResult:
    tgt_text: str
    pieces: list[PiecePlan] = field(default_factory=list)
    missed_paragraphs: list[str] = field(default_factory=list)  # plain src text


def split_page_paragraphs(page_text: str) -> list[str]:
    """Split on blank lines, drop empties."""
    parts = re.split(r"\n\s*\n", page_text.strip())
    return [p.strip() for p in parts if p.strip()]


# --- matching ----------------------------------------------------------------

def find_partial_anchor(
    page_para_norm: str,
    doc_paragraphs: list[dict],
    norm_cache: list[str],
    role: str,
    near_idx: Optional[int],
) -> Optional[tuple[int, str]]:
    """Find the JSON paragraph that contains the page-paragraph as
    prefix/suffix.  role ∈ {"first","last"}.

    For "first" we expect page_para_norm to be a SUFFIX of doc_para_norm
    (page begins mid-paragraph).
    For "last"  we expect page_para_norm to be a PREFIX of doc_para_norm.

    Returns (doc_idx, match_kind) where match_kind ∈ {"prefix","suffix","contains"}.

    Search is bounded to a window around near_idx if provided, then falls back
    to a global scan.
    """
    n = len(doc_paragraphs)
    if not page_para_norm:
        return None

    # Build search order: window first, then everything else.
    candidates: list[int] = []
    if near_idx is not None:
        lo = max(0, near_idx - 30)
        hi = min(n, near_idx + 30)
        candidates.extend(range(lo, hi))
        candidates.extend(i for i in range(n) if not (lo <= i < hi))
    else:
        candidates = list(range(n))

    # Heuristic: require at least ~30 chars overlap so we don't false-match.
    probe = page_para_norm[:80] if role == "last" else page_para_norm[-80:]
    if len(probe) < 30:
        probe = page_para_norm  # tiny paragraph — match whole thing

    for i in candidates:
        d = norm_cache[i]
        if not d:
            continue
        if role == "first":
            # page para ends the JSON para → suffix match
            if d.endswith(page_para_norm):
                return i, "suffix"
            # smaller probe: last 80 chars of page para sit at end of d
            if probe and d.endswith(probe):
                return i, "suffix"
        else:  # "last"
            if d.startswith(page_para_norm):
                return i, "prefix"
            if probe and d.startswith(probe):
                return i, "prefix"

    # Fallback: substring containment (mid-paragraph cut on both ends, rare).
    for i in candidates:
        d = norm_cache[i]
        if d and page_para_norm in d:
            return i, "contains"
        if d and probe and probe in d:
            return i, "contains"

    return None


def slice_partial_tgt(
    doc_para: dict,
    page_para_text: str,
    role: str,
    match_kind: str,
) -> tuple[str, tuple[float, float]]:
    """Cut tgt proportionally based on char fraction occupied by page_para_text
    inside doc_para['src'].
    """
    src_full = doc_para["src"]
    tgt_full = doc_para["tgt"]
    src_len = max(1, len(src_full))
    tgt_len = len(tgt_full)
    page_len = len(page_para_text)
    frac = min(1.0, page_len / src_len)

    if role == "first":
        # page para is the END of doc para → take END of tgt
        start_char = int(round(tgt_len * (1.0 - frac)))
        start_char = snap_to_word_start(tgt_full, start_char)
        return tgt_full[start_char:], ((1.0 - frac), 1.0)
    else:
        end_char = int(round(tgt_len * frac))
        end_char = snap_to_word_end(tgt_full, end_char)
        return tgt_full[:end_char], (0.0, frac)


# --- main pipeline -----------------------------------------------------------

def slice_page(
    doc: dict,
    page_text: str,
    *,
    debug: bool = False,
) -> SliceResult:
    paragraphs = doc.get("paragraphs", [])
    by_hash = doc.get("by_hash", {})
    norm_cache = [normalize(p["src"]) for p in paragraphs]

    page_paras = split_page_paragraphs(page_text)
    n = len(page_paras)
    if n == 0:
        return SliceResult(tgt_text="")

    pieces: list[PiecePlan] = []
    missed: list[str] = []
    # Track anchor to bound partial-match windows.
    last_doc_idx: Optional[int] = None

    for i, p in enumerate(page_paras):
        norm = normalize(p)
        if not norm:
            continue
        key = md5_hex(norm)
        hit = by_hash.get(key)
        if hit:
            pos = hit.get("pos")
            if pos is None:
                # legacy entries without pos — find via linear search
                for j, dp in enumerate(paragraphs):
                    if dp["hash"] == key:
                        pos = j
                        break
            last_doc_idx = pos if pos is not None else last_doc_idx
            pieces.append(PiecePlan(
                kind="full",
                src_para_idx=pos,
                src_fraction=(0.0, 1.0),
                tgt_slice=hit["tgt"],
                note="hash",
            ))
            continue

        # Hash miss — only meaningful at i==0 or i==n-1 (partial).
        role = "first" if i == 0 else ("last" if i == n - 1 else None)
        if role is None:
            missed.append(p)
            pieces.append(PiecePlan(
                kind="middle_miss",
                src_para_idx=None,
                src_fraction=(0.0, 0.0),
                tgt_slice="",
                note=f"middle paragraph hash miss: {norm[:60]!r}",
            ))
            continue

        anchor = find_partial_anchor(norm, paragraphs, norm_cache, role, last_doc_idx)
        if not anchor:
            missed.append(p)
            pieces.append(PiecePlan(
                kind=f"partial_{role}",
                src_para_idx=None,
                src_fraction=(0.0, 0.0),
                tgt_slice="",
                note=f"no partial anchor ({role}) for {norm[:60]!r}",
            ))
            continue

        doc_idx, match_kind = anchor
        doc_para = paragraphs[doc_idx]
        tgt_slice, frac = slice_partial_tgt(doc_para, norm, role, match_kind)
        last_doc_idx = doc_idx
        pieces.append(PiecePlan(
            kind=f"partial_{role}",
            src_para_idx=doc_idx,
            src_fraction=frac,
            tgt_slice=tgt_slice,
            note=f"match_kind={match_kind}",
        ))

    # Stitch with overlap-dedup: adjacent JSON paragraphs sometimes share a
    # leading/trailing chunk of tgt text (emit_rosetta's m:n grouping can put
    # the same trans sentences into both rows).  Detect and drop the overlap
    # from the later piece so we don't render it twice.
    rendered: list[str] = []
    for p in pieces:
        if not p.tgt_slice:
            continue
        if rendered:
            prev = rendered[-1]
            overlap = _max_overlap(prev, p.tgt_slice, min_chars=40)
            if overlap > 0:
                rendered.append(p.tgt_slice[overlap:])
                continue
        rendered.append(p.tgt_slice)

    tgt_text = "\n\n".join(s for s in rendered if s)
    return SliceResult(tgt_text=tgt_text, pieces=pieces, missed_paragraphs=missed)


def _max_overlap(prev: str, nxt: str, *, min_chars: int = 40) -> int:
    """Return length of longest suffix of `prev` matching prefix of `nxt`.

    Only counts overlaps ≥ min_chars (avoid false-positive single-word shared
    edges).  Search bounded to last/first 600 chars for speed.
    """
    a = prev[-600:]
    b = nxt[:600]
    max_n = min(len(a), len(b))
    for n in range(max_n, min_chars - 1, -1):
        if a[-n:] == b[:n]:
            return n
    return 0


# --- CLI ---------------------------------------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, help="parallel-aligned.json")
    ap.add_argument("--page", required=True, help="text file with one src page")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    doc = json.loads(Path(args.json).read_text())
    page_text = Path(args.page).read_text()
    result = slice_page(doc, page_text, debug=args.debug)
    print(result.tgt_text)
    if args.debug:
        print("\n--- TRACE ---", file=sys.stderr)
        for p in result.pieces:
            print(f"  {p.kind:16s} doc_idx={p.src_para_idx} "
                  f"frac=[{p.src_fraction[0]:.2f},{p.src_fraction[1]:.2f}] {p.note}",
                  file=sys.stderr)
        if result.missed_paragraphs:
            print(f"\nMISSED ({len(result.missed_paragraphs)}):", file=sys.stderr)
            for m in result.missed_paragraphs:
                print(f"  - {m[:120]!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
