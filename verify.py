#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Sample N random alignments stratified by confidence band, output side-by-side markdown."""
import argparse
import json
import random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-jsonl", required=True)
    ap.add_argument("--trans-jsonl", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("-n","--num", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o","--out", required=True)
    args = ap.parse_args()
    en = [json.loads(l) for l in Path(args.orig_jsonl).read_text().splitlines() if l.strip()]
    it = [json.loads(l) for l in Path(args.trans_jsonl).read_text().splitlines() if l.strip()]
    path = [json.loads(l) for l in Path(args.align).read_text().splitlines() if l.strip()]
    random.seed(args.seed)

    # Stratified sample by confidence band
    bands = {"high (>0.7)": [], "med (0.5-0.7)": [], "low (0.3-0.5)": [], "very_low (<0.3)": [], "gap": []}
    for p in path:
        if p["method"].startswith("gap"):
            bands["gap"].append(p); continue
        c = p["confidence"]
        if c > 0.7: bands["high (>0.7)"].append(p)
        elif c > 0.5: bands["med (0.5-0.7)"].append(p)
        elif c > 0.3: bands["low (0.3-0.5)"].append(p)
        else: bands["very_low (<0.3)"].append(p)

    out = [f"# {Path(args.align).stem} — alignment sample\n"]
    out.append(f"Total pairs: {len(path)}\n")
    out.append("## Band sizes\n")
    for k, v in bands.items():
        out.append(f"- **{k}**: {len(v)}")
    out.append("")

    per_band = args.num // len(bands)
    for band_name, pairs in bands.items():
        if not pairs: continue
        sample = random.sample(pairs, min(per_band, len(pairs)))
        out.append(f"\n## {band_name} ({per_band}/{len(pairs)})\n")
        for p in sample:
            o = " | ".join(en[i]["text"] for i in p["orig"]) if p["orig"] else "_(none)_"
            t = " | ".join(it[j]["text"] for j in p["trans"]) if p["trans"] else "_(none)_"
            ranges = ""
            if p["orig"]:
                ranges += f"EN[{p['orig'][0]}..{p['orig'][-1]}] "
            if p["trans"]:
                ranges += f"IT[{p['trans'][0]}..{p['trans'][-1]}]"
            out.append(f"### {ranges} · {p['move']} · conf {p['confidence']}")
            out.append(f"- **A:** {o}")
            out.append(f"- **B:** {t}")
            out.append("")
    Path(args.out).write_text("\n".join(out))
    print(f"wrote sample of {args.num} -> {args.out}")

if __name__ == "__main__":
    main()
