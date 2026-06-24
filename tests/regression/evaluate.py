#!/usr/bin/env python3
"""Run slice_proportional against goldens/*.json and report quality.

Golden format (one JSON per page):
{
  "name": "marta_01_full_paras",
  "json": "/abs/path/to/parallel-aligned.json",
  "category": "full | partial_one | partial_two | mid_sentence",
  "src_page": "first paragraph...\n\nsecond paragraph...",
  "expected_tgt": "atendita esperanta teksto..."
}

Metrics:
  - word IoU (tokenize on whitespace, lowercase, strip punct)
  - char Levenshtein ratio (1 - dist/max(len))
  - missing-paragraph count
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from slice_proportional import slice_page  # noqa: E402

WORD_RE = re.compile(r"\w+", re.UNICODE)


def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower()
    return WORD_RE.findall(text)


def word_iou(a: str, b: str) -> float:
    sa, sb = set(words(a)), set(words(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
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
    if m == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / m


def evaluate(goldens_dir: Path, *, verbose: bool = False) -> int:
    files = sorted(goldens_dir.glob("*.json"))
    if not files:
        print(f"no goldens in {goldens_dir}", file=sys.stderr)
        return 1

    rows = []
    doc_cache: dict[str, dict] = {}
    for f in files:
        g = json.loads(f.read_text())
        jpath = g["json"]
        if jpath not in doc_cache:
            doc_cache[jpath] = json.loads(Path(jpath).read_text())
        result = slice_page(doc_cache[jpath], g["src_page"])
        iou = word_iou(result.tgt_text, g["expected_tgt"])
        lev = lev_ratio(result.tgt_text, g["expected_tgt"])
        miss = len(result.missed_paragraphs)
        rows.append({
            "name": g.get("name", f.stem),
            "category": g.get("category", "?"),
            "iou": iou,
            "lev": lev,
            "miss": miss,
        })
        if verbose:
            print(f"\n=== {g.get('name', f.stem)} [{g.get('category')}] ===")
            print(f"IoU={iou:.3f}  Lev={lev:.3f}  miss={miss}")
            print("--- got ---")
            print(result.tgt_text[:500])
            print("--- want ---")
            print(g["expected_tgt"][:500])
            for p in result.pieces:
                print(f"  · {p.kind:16s} idx={p.src_para_idx} "
                      f"frac=[{p.src_fraction[0]:.2f},{p.src_fraction[1]:.2f}] {p.note}")

    # Summary table.
    print("\n" + "=" * 72)
    print(f"{'name':32s} {'cat':14s}  IoU    Lev   miss")
    print("-" * 72)
    by_cat: dict[str, list[float]] = {}
    for r in rows:
        print(f"{r['name']:32s} {r['category']:14s}  {r['iou']:.3f}  {r['lev']:.3f}  {r['miss']:>3d}")
        by_cat.setdefault(r["category"], []).append(r["iou"])
    print("-" * 72)
    avg = sum(r["iou"] for r in rows) / len(rows)
    print(f"{'OVERALL':32s} {'':14s}  {avg:.3f}")
    for cat, ious in sorted(by_cat.items()):
        m = sum(ious) / len(ious)
        print(f"  {cat:30s} ({len(ious):2d})  {m:.3f}")
    print()
    print(f"PASS threshold: 0.90 mean IoU.  Result: "
          f"{'PASS' if avg >= 0.90 else 'FAIL'} ({avg:.3f})")
    return 0 if avg >= 0.90 else 2


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goldens", default=str(Path(__file__).parent / "goldens"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    sys.exit(evaluate(Path(args.goldens), verbose=args.verbose))


if __name__ == "__main__":
    main()
