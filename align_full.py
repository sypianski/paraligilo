#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Run anchor-segmented DP over a full book pair."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

# reuse align core
sys.path.insert(0, str(Path(__file__).parent))
from align import align

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-jsonl", required=True)
    ap.add_argument("--trans-jsonl", required=True)
    ap.add_argument("--orig-npy", required=True)
    ap.add_argument("--trans-npy", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--max-segment", type=int, default=200,
                    help="If anchor gap > this many orig sents, slack-fill (no DP)")
    ap.add_argument("-o","--out", required=True)
    args = ap.parse_args()

    rows_o = [json.loads(l) for l in Path(args.orig_jsonl).read_text().splitlines() if l.strip()]
    rows_t = [json.loads(l) for l in Path(args.trans_jsonl).read_text().splitlines() if l.strip()]
    vec_o = np.load(args.orig_npy)
    vec_t = np.load(args.trans_npy)
    anchors = [json.loads(l) for l in Path(args.anchors).read_text().splitlines() if l.strip()]

    # Add virtual anchors at start and end
    boundary = [{"orig": 0, "trans": 0, "sim": 1.0}] + anchors + \
               [{"orig": len(rows_o), "trans": len(rows_t), "sim": 1.0}]
    # dedupe consecutive same
    boundary_clean = [boundary[0]]
    for a in boundary[1:]:
        last = boundary_clean[-1]
        if a["orig"] > last["orig"] and a["trans"] > last["trans"]:
            boundary_clean.append(a)
    boundary = boundary_clean

    full_path = []
    n_skipped = 0
    for k in range(len(boundary) - 1):
        a, b = boundary[k], boundary[k+1]
        o0, o1 = a["orig"], b["orig"]
        t0, t1 = a["trans"], b["trans"]
        n_o = o1 - o0
        n_t = t1 - t0
        if n_o == 0 and n_t == 0:
            continue
        if n_o > args.max_segment or n_t > args.max_segment:
            # too large — slack-fill as a single block
            full_path.append({
                "orig": list(range(o0, o1)),
                "trans": list(range(t0, t1)),
                "move": f"{n_o}:{n_t}",
                "confidence": 0.0,
                "method": "slack_large",
            })
            n_skipped += 1
            continue
        sub_o = rows_o[o0:o1]
        sub_t = rows_t[t0:t1]
        sub_vo = vec_o[o0:o1]
        sub_vt = vec_t[t0:t1]
        if len(sub_o) == 0 or len(sub_t) == 0:
            # one side empty — record gaps
            for i in range(len(sub_o)):
                full_path.append({"orig":[o0+i],"trans":[],"move":"1:0","confidence":0.0,"method":"gap_orig"})
            for j in range(len(sub_t)):
                full_path.append({"orig":[],"trans":[t0+j],"move":"0:1","confidence":0.0,"method":"gap_trans"})
            continue
        # local DP band proportional to segment size
        band = max(15, min(60, max(n_o, n_t)))
        path = align(sub_o, sub_t, sub_vo, sub_vt, band=band)
        # remap indices back to global
        for p in path:
            p["orig"] = [o0 + i for i in p["orig"]]
            p["trans"] = [t0 + j for j in p["trans"]]
            full_path.append(p)
        if k % 50 == 0:
            print(f"  segment {k}/{len(boundary)-1}", file=sys.stderr)

    Path(args.out).write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in full_path))
    moves = {}
    confs = []
    for p in full_path:
        moves[p["move"]] = moves.get(p["move"], 0) + 1
        if p.get("method") == "embed":
            confs.append(p["confidence"])
    print(f"segments: {len(boundary)-1} (slack-filled large: {n_skipped})", file=sys.stderr)
    print(f"path len={len(full_path)} -> {args.out}", file=sys.stderr)
    print(f"top moves: {sorted(moves.items(), key=lambda x:-x[1])[:8]}", file=sys.stderr)
    if confs:
        confs.sort()
        print(f"embed conf: min={confs[0]:.2f} p25={confs[len(confs)//4]:.2f} med={confs[len(confs)//2]:.2f} p75={confs[3*len(confs)//4]:.2f} max={confs[-1]:.2f}", file=sys.stderr)

if __name__ == "__main__":
    main()
