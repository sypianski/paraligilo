#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Find strong cross-lingual anchors by k-NN over embeddings + monotonicity filter.

For every Nth orig sentence, find its top-1 trans sentence by cos sim.
Keep pairs where:
  - cos sim > threshold
  - mapping is monotonic (no back-jumps in a sliding window)
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

def find_anchors(vec_o, vec_t, stride=20, top_thresh=0.70, mono_window=5):
    N, M = vec_o.shape[0], vec_t.shape[0]
    # global ratio estimate (will refine later)
    ratio = M / N
    # Sample every `stride` orig sentence, top-1 nn in trans within wide band
    # band wide enough to absorb drift
    band = max(500, int(M * 0.15))
    cands = []
    for i in range(0, N, stride):
        c = i * ratio
        lo = max(0, int(c - band))
        hi = min(M, int(c + band))
        if hi <= lo: continue
        sims = vec_o[i] @ vec_t[lo:hi].T
        j_off = int(np.argmax(sims))
        s = float(sims[j_off])
        if s >= top_thresh:
            cands.append((i, lo + j_off, s))
    print(f"  raw candidates: {len(cands)}", file=sys.stderr)
    # Monotonicity filter: enforce non-decreasing j (allow noise via DP-LIS)
    # Longest non-decreasing subsequence of (i, j) ordered by i
    js = [j for _,j,_ in cands]
    # LIS in O(n log n) on j, requiring j non-decreasing
    import bisect
    tails = []   # tails[len-1] = smallest tail j of LIS of length len
    pred = [-1]*len(cands)
    tail_idx = []
    for k, (_,j,_) in enumerate(cands):
        pos = bisect.bisect_right(tails, j)
        if pos == len(tails):
            tails.append(j); tail_idx.append(k)
        else:
            tails[pos] = j; tail_idx[pos] = k
        pred[k] = tail_idx[pos-1] if pos > 0 else -1
    # reconstruct
    if not tail_idx:
        return []
    chain = []
    k = tail_idx[-1]
    while k != -1:
        chain.append(k); k = pred[k]
    chain.reverse()
    mono = [cands[k] for k in chain]
    print(f"  monotonic-filtered: {len(mono)}", file=sys.stderr)
    # bucket-validate: in a window of mono_window consecutive anchors, j must
    # be strictly non-decreasing (already by LIS) — drop any with sim drop
    final = []
    for i, j, s in mono:
        final.append({"orig": i, "trans": j, "sim": round(s, 3)})
    return final

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-npy", required=True)
    ap.add_argument("--trans-npy", required=True)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("-o","--out", required=True)
    args = ap.parse_args()
    vo = np.load(args.orig_npy)
    vt = np.load(args.trans_npy)
    anchors = find_anchors(vo, vt, stride=args.stride, top_thresh=args.threshold)
    Path(args.out).write_text("\n".join(json.dumps(a) for a in anchors))
    # estimate local ratios
    if len(anchors) >= 2:
        ratios = []
        for k in range(1, len(anchors)):
            di = anchors[k]["orig"] - anchors[k-1]["orig"]
            dj = anchors[k]["trans"] - anchors[k-1]["trans"]
            if di > 0:
                ratios.append(dj/di)
        ratios.sort()
        print(f"  anchors={len(anchors)} local ratio: min={ratios[0]:.2f} med={ratios[len(ratios)//2]:.2f} max={ratios[-1]:.2f}", file=sys.stderr)
        print(f"  first anchor: orig {anchors[0]['orig']} ↔ trans {anchors[0]['trans']} (sim {anchors[0]['sim']})", file=sys.stderr)
        print(f"  last anchor: orig {anchors[-1]['orig']} ↔ trans {anchors[-1]['trans']} (sim {anchors[-1]['sim']})", file=sys.stderr)

if __name__ == "__main__":
    main()
