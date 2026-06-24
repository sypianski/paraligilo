#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Emit rosetta-aligned.json from paraligilo final alignment + sentence files.

Consumed by the rosetta KOReader plugin
(https://github.com/sypianski/koreader-konfig → koplugins/rosetta.koplugin).
Schema in koreader-konfig/koplugins/rosetta.koplugin/ANCHOR_FORMAT.md.

Usage:
    emit_rosetta.py --orig-jsonl X_orig.jsonl --trans-jsonl X_trans.jsonl \\
                    --align X_final.jsonl --src-lang it --tgt-lang pl \\
                    --src-file orig.epub --tgt-file trans.epub \\
                    -o rosetta-aligned.json
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Match rosetta_aligned.lua:normalize() byte-for-byte.

    1. Replace NBSP (U+00A0) with regular space.
    2. Collapse runs of any whitespace to a single ASCII space.
    3. Strip leading + trailing whitespace.
    4. NFC normalize.
    """
    text = text.replace(" ", " ")
    text = WS.sub(" ", text)
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    return text


def md5_hex(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_sentences(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_alignment(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def group_paragraphs(sents: list[dict]) -> dict[tuple[int, int], list[int]]:
    """(chapter, para) -> [sentence_index, …] in original order."""
    para_to_sents: dict[tuple[int, int], list[int]] = {}
    for i, s in enumerate(sents):
        key = (s["chapter"], s["para"])
        para_to_sents.setdefault(key, []).append(i)
    return para_to_sents


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig-jsonl",  required=True)
    ap.add_argument("--trans-jsonl", required=True)
    ap.add_argument("--align",       required=True, help="X_final.jsonl")
    ap.add_argument("--src-lang", default="??")
    ap.add_argument("--tgt-lang", default="??")
    ap.add_argument("--src-file", default="")
    ap.add_argument("--tgt-file", default="")
    ap.add_argument("--method",   default="hybrid",
                    choices=["gale-church", "embeddings", "llm", "hybrid"])
    ap.add_argument("--generator", default="paraligilo")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    orig = load_sentences(Path(args.orig_jsonl))
    trans = load_sentences(Path(args.trans_jsonl))
    align = load_alignment(Path(args.align))

    # Build orig sent index -> list of trans sent indices, with confidence per row.
    # Same trans sent may appear under multiple orig sents (m:1 mapping); we keep
    # all and dedup later when concatenating.
    orig_to_trans: dict[int, list[tuple[list[int], float]]] = {}
    for row in align:
        confidence = float(row.get("confidence", 0.0))
        trans_idxs = list(row.get("trans", []))
        for o in row.get("orig", []):
            orig_to_trans.setdefault(int(o), []).append((trans_idxs, confidence))

    # Group source sentences into paragraphs.
    para_to_sents = group_paragraphs(orig)

    by_hash: dict[str, dict] = {}
    matched = 0
    low_conf = 0
    skipped_no_alignment = 0

    for (chap, para), sent_idxs in para_to_sents.items():
        # Source paragraph text = join sentences with single space.
        src_text = " ".join(orig[i].get("text", "") for i in sent_idxs).strip()
        if not src_text:
            continue
        norm = normalize(src_text)
        if not norm:
            continue
        key = md5_hex(norm)

        # Collect target sentences referenced by any alignment row touching
        # this paragraph's sentences.  Keep original order, dedup.
        seen_trans = set()
        ordered_trans_idxs: list[int] = []
        confs: list[float] = []
        for s_idx in sent_idxs:
            for trans_idxs, conf in orig_to_trans.get(s_idx, []):
                confs.append(conf)
                for t in trans_idxs:
                    if t not in seen_trans:
                        seen_trans.add(t)
                        ordered_trans_idxs.append(t)

        if not ordered_trans_idxs:
            skipped_no_alignment += 1
            continue
        ordered_trans_idxs.sort()
        tgt_text = " ".join(trans[t].get("text", "") for t in ordered_trans_idxs).strip()
        if not tgt_text:
            skipped_no_alignment += 1
            continue
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        if avg_conf < 0.5:
            low_conf += 1
        matched += 1
        by_hash[key] = {"tgt": tgt_text, "confidence": round(avg_conf, 3)}

    out_doc = {
        "rosetta_align_version": 1,
        "src": {"file": args.src_file, "lang": args.src_lang},
        "tgt": {"file": args.tgt_file, "lang": args.tgt_lang},
        "method": args.method,
        "generator": args.generator,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stats": {
            "src_paragraphs": len(para_to_sents),
            "matched": matched,
            "low_confidence": low_conf,
            "skipped_no_alignment": skipped_no_alignment,
        },
        "by_hash": by_hash,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=0)

    print(f"emit_rosetta: paragraphs={len(para_to_sents)} matched={matched} "
          f"low_conf={low_conf} skipped={skipped_no_alignment} → {out_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
