#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Second pass: harvest high-confidence 1:1 pairs from a previous alignment
as additional anchors, merge with original anchor set, re-run DP."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from align_full import main as _  # ensure module loaded; we call align directly
from align import align

def harvest_anchors(path, min_conf=0.78):
    """Take 1:1 pairs with conf >= min_conf as anchors."""
    out = []
    for p in path:
        if p.get("method") not in ("embed", "embed_concat"):
            continue
        if len(p["orig"]) != 1 or len(p["trans"]) != 1:
            continue
        if p["confidence"] < min_conf:
            continue
        out.append({"orig": p["orig"][0], "trans": p["trans"][0], "sim": p["confidence"]})
    return out

def merge_anchors(a, b):
    """Union, deduped by orig, monotonic by both axes."""
    combined = {}
    for x in a + b:
        i = x["orig"]
        if i not in combined or combined[i]["sim"] < x["sim"]:
            combined[i] = x
    items = sorted(combined.values(), key=lambda x: x["orig"])
    # enforce strict monotonic trans via LIS
    import bisect
    tails, tail_idx, pred = [], [], [-1]*len(items)
    for k, x in enumerate(items):
        pos = bisect.bisect_right(tails, x["trans"])
        if pos == len(tails):
            tails.append(x["trans"]); tail_idx.append(k)
        else:
            tails[pos] = x["trans"]; tail_idx[pos] = k
        pred[k] = tail_idx[pos-1] if pos > 0 else -1
    if not tail_idx: return []
    chain = []
    k = tail_idx[-1]
    while k != -1:
        chain.append(k); k = pred[k]
    chain.reverse()
    return [items[k] for k in chain]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-jsonl", required=True)
    ap.add_argument("--trans-jsonl", required=True)
    ap.add_argument("--orig-npy", required=True)
    ap.add_argument("--trans-npy", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--min-conf", type=float, default=0.78)
    ap.add_argument("-o","--out", required=True)
    ap.add_argument("--anchors-out", required=True)
    args = ap.parse_args()

    rows_o = [json.loads(l) for l in Path(args.orig_jsonl).read_text().splitlines() if l.strip()]
    rows_t = [json.loads(l) for l in Path(args.trans_jsonl).read_text().splitlines() if l.strip()]
    vec_o = np.load(args.orig_npy)
    vec_t = np.load(args.trans_npy)
    prev_path = [json.loads(l) for l in Path(args.align).read_text().splitlines() if l.strip()]
    base_anchors = [json.loads(l) for l in Path(args.anchors).read_text().splitlines() if l.strip()]

    new = harvest_anchors(prev_path, min_conf=args.min_conf)
    print(f"  harvested {len(new)} high-conf 1:1 pairs", file=sys.stderr)
    merged = merge_anchors(base_anchors, new)
    print(f"  merged: {len(base_anchors)} + {len(new)} -> {len(merged)} (after LIS)", file=sys.stderr)

    Path(args.anchors_out).write_text("\n".join(json.dumps(a) for a in merged))

    # re-run DP
    boundary = [{"orig":0,"trans":0,"sim":1.0}] + merged + \
               [{"orig":len(rows_o),"trans":len(rows_t),"sim":1.0}]
    clean = [boundary[0]]
    for a in boundary[1:]:
        if a["orig"] > clean[-1]["orig"] and a["trans"] > clean[-1]["trans"]:
            clean.append(a)
    boundary = clean
    full = []
    for k in range(len(boundary)-1):
        a, b = boundary[k], boundary[k+1]
        o0, o1 = a["orig"], b["orig"]
        t0, t1 = a["trans"], b["trans"]
        n_o, n_t = o1-o0, t1-t0
        if n_o == 0 and n_t == 0: continue
        if n_o == 0 or n_t == 0:
            for i in range(n_o):
                full.append({"orig":[o0+i],"trans":[],"move":"1:0","confidence":0.0,"method":"gap_orig"})
            for j in range(n_t):
                full.append({"orig":[],"trans":[t0+j],"move":"0:1","confidence":0.0,"method":"gap_trans"})
            continue
        if n_o > 200 or n_t > 200:
            full.append({"orig":list(range(o0,o1)),"trans":list(range(t0,t1)),
                         "move":f"{n_o}:{n_t}","confidence":0.0,"method":"slack_large"})
            continue
        # tighter band thanks to denser anchors
        band = max(10, min(40, max(n_o, n_t)))
        seg = align(rows_o[o0:o1], rows_t[t0:t1], vec_o[o0:o1], vec_t[t0:t1], band=band)
        for p in seg:
            p["orig"] = [o0+i for i in p["orig"]]
            p["trans"] = [t0+j for j in p["trans"]]
            full.append(p)
    Path(args.out).write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in full))
    moves = {}
    confs = []
    for p in full:
        moves[p["move"]] = moves.get(p["move"], 0) + 1
        if p.get("method") in ("embed","embed_concat"):
            confs.append(p["confidence"])
    print(f"path len={len(full)} -> {args.out}", file=sys.stderr)
    print(f"top moves: {sorted(moves.items(), key=lambda x:-x[1])[:8]}", file=sys.stderr)
    if confs:
        confs.sort()
        print(f"embed conf: min={confs[0]:.2f} p25={confs[len(confs)//4]:.2f} med={confs[len(confs)//2]:.2f} p75={confs[3*len(confs)//4]:.2f} max={confs[-1]:.2f}", file=sys.stderr)

if __name__ == "__main__":
    main()
