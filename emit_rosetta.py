#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Emit parallel-aligned.json from paraligilo final alignment + sentence files.

Consumed by the rosetta KOReader plugin (https://github.com/sypianski/rosetta.koplugin).
Schema: https://github.com/sypianski/parallel-align-spec

Usage:
    emit_rosetta.py --orig-jsonl X_orig.jsonl --trans-jsonl X_trans.jsonl \\
                    --align X_final.jsonl --src-lang it --tgt-lang pl \\
                    --src-file orig.epub --tgt-file trans.epub \\
                    -o parallel-aligned.json
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
# Footnote marker straggler: " 3 " between two letters that we couldn't strip
# at the HTML stage because the EPUB used inline non-anchor markup.
INLINE_FOOTNOTE_NUM = re.compile(r"(\w)\s+\d{1,3}\s+(\w)")
# Curly-quote / dash variants that crengine may render differently.
QUOTE_FOLD = [
    ("\u00a0", " "),       # NBSP
    ("\u00ad", ""),        # soft hyphen \u2014 crengine drops when rendering
    ("\ufeff", ""),        # BOM / ZWNBSP
    ("\u200b", ""),        # zero-width space
    ("\u200c", ""), ("\u200d", ""),   # ZWNJ / ZWJ
    ("\u201e", '"'), ("\u201d", '"'), ("\u201c", '"'),
    ("\u00ab", '"'), ("\u00bb", '"'),
    ("\u201a", "'"), ("\u2019", "'"), ("\u2018", "'"),
    ("\u2039", "'"), ("\u203a", "'"),
    ("\u2013", "-"), ("\u2014", "-"), ("\u2212", "-"),
    ("\u2026", "..."),
]


def normalize(text: str) -> str:
    """Tolerant paragraph normalization. Must stay in sync with
    rosetta_aligned.lua:normalize() — both sides must produce the same hash.

    Steps:
      1. Fold curly quotes / em-dashes / NBSP / ellipsis to ASCII.
      2. Drop straggler inline footnote markers (Word N Word -> Word Word).
      3. Collapse whitespace.
      4. NFC normalize.
      5. Lowercase.
    """
    for k, v in QUOTE_FOLD:
        text = text.replace(k, v)
    # Strip markdown emphasis markers (KOReader's Extract:toTitledPlainText
    # injects *italic* and **bold** from <em>/<i>/<strong>/<b> spans, while
    # paraligilo's HTML extract strips the tags but keeps the plain text).
    text = re.sub(r"\*+", "", text)
    # Strip footnote-suffix digits glued onto a word: 'Graniczna3' / 'ingrediencję2'.
    text = re.sub(r"(\w)\d+", r"\1", text)
    for _ in range(2):
        text = INLINE_FOOTNOTE_NUM.sub(r"\1 \2", text)
    text = WS.sub(" ", text)
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    # Drop whitespace adjacent to punctuation (must mirror Lua side):
    # paraligilo extract inserts ' ' between adjacent inline elements
    # ('„<em>X</em>!"' -> '„ X !'), crengine renders them flush ('„X!').
    text = re.sub(r"\s+([!?.,;:)\]}\"'])", r"\1", text)
    text = re.sub(r"([(\[{\"'])\s+", r"\1", text)
    return text.lower()


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
    ap.add_argument("-o", "--out",
                    help="Output path.  Defaults to <src-file>.sdr/parallel-aligned.json "
                         "when --src-file ends in .epub.")
    args = ap.parse_args()

    # Default output to KOReader's sidecar location next to the source EPUB.
    if not args.out:
        if args.src_file and args.src_file.lower().endswith(".epub"):
            args.out = args.src_file + ".sdr/parallel-aligned.json"
        else:
            ap.error("either --out or a .epub --src-file is required")

    orig = load_sentences(Path(args.orig_jsonl))
    trans = load_sentences(Path(args.trans_jsonl))
    align = load_alignment(Path(args.align))

    # Build orig sent index -> list of trans sent indices, with confidence per row.
    # Same trans sent may appear under multiple orig sents (m:1 mapping); we keep
    # all and dedup later when concatenating.
    #
    # SKIP unreliable rows:
    #   - slack_large: DP couldn't align a giant segment — propagating its
    #     hundreds of trans sentences to every orig in the block bloats the
    #     output by 10x+ and creates meaningless multi-paragraph tgt strings.
    #   - gap_orig/gap_trans: one-sided, no useful pairing.
    #   - llm_no: explicitly judged a non-alignment.
    SKIP_METHODS = {"slack_large", "gap_orig", "gap_trans", "noop", "llm_no", "llm_fail"}
    # Also defensively cap absurd m:n blocks even when method label is missing.
    MAX_BLOCK_TRANS = 8

    n_dropped_slack = 0
    n_dropped_huge = 0
    orig_to_trans: dict[int, list[tuple[list[int], float]]] = {}
    for row in align:
        method = row.get("method", "")
        if method in SKIP_METHODS:
            if method == "slack_large":
                n_dropped_slack += 1
            continue
        trans_idxs = list(row.get("trans", []))
        if len(trans_idxs) > MAX_BLOCK_TRANS or len(row.get("orig", [])) > MAX_BLOCK_TRANS:
            n_dropped_huge += 1
            continue
        confidence = float(row.get("confidence", 0.0))
        for o in row.get("orig", []):
            orig_to_trans.setdefault(int(o), []).append((trans_idxs, confidence))

    if n_dropped_slack or n_dropped_huge:
        print(f"  filtered: {n_dropped_slack} slack_large + {n_dropped_huge} oversize rows",
              file=sys.stderr)

    # Group source sentences into paragraphs.
    para_to_sents = group_paragraphs(orig)

    by_hash: dict[str, dict] = {}
    paragraphs: list[dict] = []   # ordered list with src+tgt for AI fallback
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

        # Emit untranslated paragraphs (those whose only alignment rows landed
        # in SKIP_METHODS) as empty-tgt placeholders. Keeps the paragraphs[]
        # index contiguous with the source, which is what the proportional
        # slicer relies on for partial-anchor windowing.  Consumers that want
        # to skip empties can check tgt == "" or confidence == 0.0.
        if not ordered_trans_idxs:
            skipped_no_alignment += 1
            pos = len(paragraphs)
            by_hash[key] = {"tgt": "", "confidence": 0.0, "pos": pos}
            paragraphs.append({
                "hash": key, "src": src_text, "tgt": "", "confidence": 0.0,
            })
            continue
        ordered_trans_idxs.sort()
        tgt_text = " ".join(trans[t].get("text", "") for t in ordered_trans_idxs).strip()
        if not tgt_text:
            skipped_no_alignment += 1
            pos = len(paragraphs)
            by_hash[key] = {"tgt": "", "confidence": 0.0, "pos": pos}
            paragraphs.append({
                "hash": key, "src": src_text, "tgt": "", "confidence": 0.0,
            })
            continue
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        if avg_conf < 0.5:
            low_conf += 1
        matched += 1
        pos = len(paragraphs)
        by_hash[key] = {"tgt": tgt_text, "confidence": round(avg_conf, 3), "pos": pos}
        paragraphs.append({
            "hash": key,
            "src": src_text,        # keep full src text so the AI fallback can
                                    # match by content even when a runtime hash
                                    # diverges from the writer's hash.
            "tgt": tgt_text,
            "confidence": round(avg_conf, 3),
        })

    # Dedup overlapping tgt content between consecutive paragraphs.  The m:n
    # alignment in align_full.py can map the same trans sentences into multiple
    # adjacent orig paragraphs — when both render together in a KOReader page,
    # the shared chunk shows up twice.  Trim the leading overlap from the
    # later paragraph (>=40 chars of shared content) so each emitted tgt is
    # unique.  by_hash entries are updated in lockstep.
    n_dedup = 0
    for i in range(1, len(paragraphs)):
        prev_tgt = paragraphs[i - 1]["tgt"]
        cur_tgt = paragraphs[i]["tgt"]
        max_n = min(600, len(prev_tgt), len(cur_tgt))
        overlap = 0
        for n in range(max_n, 39, -1):
            if prev_tgt[-n:] == cur_tgt[:n]:
                overlap = n
                break
        if overlap > 0:
            trimmed = cur_tgt[overlap:].lstrip()
            if trimmed:
                paragraphs[i]["tgt"] = trimmed
                by_hash[paragraphs[i]["hash"]]["tgt"] = trimmed
                n_dedup += 1
    if n_dedup:
        print(f"  deduped tgt overlap on {n_dedup} consecutive pairs", file=sys.stderr)

    out_doc = {
        "epa_version": 2,
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
        "paragraphs": paragraphs,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=0)

    # Companion file: rosetta-paired.txt records the translation EPUB path so
    # rosetta knows the pair even without parsing the JSON.  Only written when
    # both src and tgt files are real .epub paths.
    if (args.src_file and args.src_file.lower().endswith(".epub")
            and args.tgt_file and args.tgt_file.lower().endswith(".epub")):
        paired_path = Path(args.src_file + ".sdr/rosetta-paired.txt")
        paired_path.parent.mkdir(parents=True, exist_ok=True)
        paired_path.write_text(args.tgt_file)

    print(f"emit_rosetta: paragraphs={len(para_to_sents)} matched={matched} "
          f"low_conf={low_conf} skipped={skipped_no_alignment} → {out_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
