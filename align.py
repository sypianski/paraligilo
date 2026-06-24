#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Gale-Church-style m:n DP over embedded sentences, with band constraint."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

GAP_O = 0.55   # cost of dropping an orig sentence (no trans counterpart)
GAP_T = 0.55   # cost of dropping a trans sentence (extra in trans)
SPLIT = 0.10   # extra cost for 1:2
MERGE = 0.10   # extra cost for 2:1
TWO2  = 0.20   # extra cost for 2:2
SEM_W = 0.65   # weight of semantic distance
LEN_W = 0.20   # weight of length-ratio deviation
INF = 1e9

def length(text):
    return max(1, len(text))

def cos(a, b):
    # vectors are L2-normalised already
    return float(np.dot(a, b))

def mean_vec(*vs):
    s = vs[0].copy()
    for v in vs[1:]:
        s = s + v
    s = s / np.linalg.norm(s)
    return s

def align(rows_o, rows_t, vec_o, vec_t, band=250):
    N, M = len(rows_o), len(rows_t)
    ratio = M / N if N else 1.0
    print(f"  N={N} M={M} expected ratio={ratio:.3f} band=±{band}", file=sys.stderr)

    # DP cost grid stored sparsely as dict: (i,j) -> (cost, back_move)
    # back_move: (di, dj)  where step came from (i-di, j-dj)
    # Use band: only consider j in [i*ratio - band, i*ratio + band]
    def j_range(i):
        c = i * ratio
        lo = max(0, int(c - band))
        hi = min(M, int(c + band) + 1)
        return lo, hi

    # Pre-compute lens for length-ratio cost
    Lo = np.array([length(r["text"]) for r in rows_o], dtype=np.float32)
    Lt = np.array([length(r["text"]) for r in rows_t], dtype=np.float32)

    def len_pen(lo, lt):
        # symmetric deviation from expected ratio
        exp = lo * ratio
        return min(1.0, abs(lt - exp) / max(exp, lt, 1.0))

    # full dense DP within band — index j as offset within band
    width = 2*band + 2
    # store as N+1 rows of (cost, back) for j in band
    # Using dict-of-dict for simplicity (PoC scale OK)
    dp = {}
    dp[(0, 0)] = (0.0, None)

    def best_pred(i, j):
        # candidates: (di, dj, extra_penalty, sem_cost_or_None)
        cands = []
        # 1:0 — orig dropped (i comes from i-1, j same)
        if i >= 1 and (i-1, j) in dp:
            cands.append(((i-1, j), GAP_O))
        # 0:1 — trans dropped
        if j >= 1 and (i, j-1) in dp:
            cands.append(((i, j-1), GAP_T))
        # 1:1
        if i >= 1 and j >= 1 and (i-1, j-1) in dp:
            sem = 1.0 - cos(vec_o[i-1], vec_t[j-1])
            lp  = len_pen(Lo[i-1], Lt[j-1])
            c   = SEM_W*sem + LEN_W*lp
            cands.append(((i-1, j-1), c))
        # 1:2 — orig i-1 aligns with trans j-2,j-1
        if i >= 1 and j >= 2 and (i-1, j-2) in dp:
            v = mean_vec(vec_t[j-2], vec_t[j-1])
            sem = 1.0 - cos(vec_o[i-1], v)
            lp  = len_pen(Lo[i-1], Lt[j-2]+Lt[j-1])
            c   = SEM_W*sem + LEN_W*lp + SPLIT
            cands.append(((i-1, j-2), c))
        # 2:1
        if i >= 2 and j >= 1 and (i-2, j-1) in dp:
            v = mean_vec(vec_o[i-2], vec_o[i-1])
            sem = 1.0 - cos(v, vec_t[j-1])
            lp  = len_pen(Lo[i-2]+Lo[i-1], Lt[j-1])
            c   = SEM_W*sem + LEN_W*lp + MERGE
            cands.append(((i-2, j-1), c))
        # 2:2
        if i >= 2 and j >= 2 and (i-2, j-2) in dp:
            vo = mean_vec(vec_o[i-2], vec_o[i-1])
            vt = mean_vec(vec_t[j-2], vec_t[j-1])
            sem = 1.0 - cos(vo, vt)
            lp  = len_pen(Lo[i-2]+Lo[i-1], Lt[j-2]+Lt[j-1])
            c   = SEM_W*sem + LEN_W*lp + TWO2
            cands.append(((i-2, j-2), c))
        if not cands:
            return None
        best = min(cands, key=lambda x: dp[x[0]][0] + x[1])
        prev, add = best
        return prev, dp[prev][0] + add

    # Sweep i from 0..N, j in band(i)
    for i in range(0, N+1):
        lo, hi = j_range(i)
        if i == 0:
            lo = 0
        for j in range(lo, hi+1):
            if j > M: break
            if i == 0 and j == 0:
                continue
            r = best_pred(i, j)
            if r is None:
                continue
            prev, cost = r
            dp[(i, j)] = (cost, prev)
        if i % 500 == 0:
            print(f"  dp: i={i}/{N} states={len(dp)}", file=sys.stderr)

    # End point: prefer (N,M) but accept the cheapest of last row
    end = None
    end_cost = INF
    for (i, j), (c, _) in dp.items():
        if i == N and j >= M - band:
            # penalise leftover trans sentences (j<M) as gaps
            tail = (M - j) * GAP_T
            total = c + tail
            if total < end_cost:
                end_cost = total
                end = (i, j)
        if i >= N - band and j == M:
            tail = (N - i) * GAP_O
            total = c + tail
            if total < end_cost:
                end_cost = total
                end = (i, j)
    print(f"  end={end} cost={end_cost:.2f}", file=sys.stderr)

    # Backtrack
    path = []
    cur = end
    while cur and cur in dp and dp[cur][1] is not None:
        prev = dp[cur][1]
        di = cur[0] - prev[0]
        dj = cur[1] - prev[1]
        # record alignment between rows_o[prev[0]:cur[0]] and rows_t[prev[1]:cur[1]]
        path.append({
            "orig": list(range(prev[0], cur[0])),
            "trans": list(range(prev[1], cur[1])),
            "move": f"{di}:{dj}",
        })
        cur = prev
    path.reverse()

    # confidence per pair
    for p in path:
        if not p["orig"] and not p["trans"]:
            p["confidence"] = 0.0
            p["method"] = "noop"; continue
        if not p["orig"]:
            p["confidence"] = 0.0
            p["method"] = "gap_orig"; continue
        if not p["trans"]:
            p["confidence"] = 0.0
            p["method"] = "gap_trans"; continue
        vo = mean_vec(*[vec_o[i] for i in p["orig"]]) if len(p["orig"])>1 else vec_o[p["orig"][0]]
        vt = mean_vec(*[vec_t[j] for j in p["trans"]]) if len(p["trans"])>1 else vec_t[p["trans"][0]]
        p["confidence"] = round(cos(vo, vt), 3)
        p["method"] = "embed"
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-jsonl", required=True)
    ap.add_argument("--trans-jsonl", required=True)
    ap.add_argument("--orig-npy", required=True)
    ap.add_argument("--trans-npy", required=True)
    ap.add_argument("--band", type=int, default=250)
    ap.add_argument("-o","--out", required=True)
    args = ap.parse_args()
    rows_o = [json.loads(l) for l in Path(args.orig_jsonl).read_text().splitlines() if l.strip()]
    rows_t = [json.loads(l) for l in Path(args.trans_jsonl).read_text().splitlines() if l.strip()]
    vec_o = np.load(args.orig_npy)
    vec_t = np.load(args.trans_npy)
    path = align(rows_o, rows_t, vec_o, vec_t, band=args.band)
    Path(args.out).write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in path))
    # stats
    moves = {}
    for p in path:
        moves[p["move"]] = moves.get(p["move"], 0) + 1
    print(f"path len={len(path)} -> {args.out}", file=sys.stderr)
    print(f"moves: {moves}", file=sys.stderr)
    confs = [p["confidence"] for p in path if p["method"]=="embed"]
    if confs:
        confs.sort()
        print(f"conf: min={confs[0]:.2f} p25={confs[len(confs)//4]:.2f} median={confs[len(confs)//2]:.2f} p75={confs[3*len(confs)//4]:.2f} max={confs[-1]:.2f}", file=sys.stderr)

if __name__ == "__main__":
    main()
