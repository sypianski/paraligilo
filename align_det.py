#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Classic Gale-Church length-only DP with m:n moves, per anchor segment.
No embeddings, no API. Deterministic.

Cost model (Gale & Church 1993, simplified):
  For each move (m, n) with lengths Lo (sum of m orig sentence lengths) and
  Lt (sum of n trans sentence lengths):
    expected_Lt = Lo * ratio
    z = abs(Lt - expected_Lt) / sqrt(Lo * variance)
    cost = move_log_prior + z
  Move log-priors empirically: 1:1 cheap, 1:2 / 2:1 medium, 2:2 dearer, 1:0 / 0:1 expensive.
"""
import argparse
import json
import math
import sys
from pathlib import Path

MOVE_PRIOR = {
    (1,1): 0.0,
    (1,2): 0.3, (2,1): 0.3,
    (1,3): 1.0, (3,1): 1.0,
    (2,2): 0.6,
    (1,0): 1.8, (0,1): 1.8,
}
VAR_FACTOR = 6.8   # Gale-Church empirical variance per char of orig length

def length(text):
    return max(1, len(text))

def segment_lens(rows, lo, hi):
    return [length(rows[i]["text"]) for i in range(lo, hi)]

def align_segment(rows_o, rows_t, ratio):
    """Length-only Gale-Church DP returning list of move dicts."""
    N, M = len(rows_o), len(rows_t)
    Lo = [length(r["text"]) for r in rows_o]
    Lt = [length(r["text"]) for r in rows_t]
    INF = 1e18
    # dp[i][j] = min cost to align rows_o[:i] with rows_t[:j]
    # store as dict for sparse cases; for clean segments dense list OK.
    dp = [[INF]*(M+1) for _ in range(N+1)]
    back = [[None]*(M+1) for _ in range(N+1)]
    dp[0][0] = 0.0
    for i in range(N+1):
        for j in range(M+1):
            if dp[i][j] >= INF: continue
            for (dm, dn), prior in MOVE_PRIOR.items():
                ni, nj = i+dm, j+dn
                if ni > N or nj > M: continue
                lo_sum = sum(Lo[i:ni]) if dm else 0
                lt_sum = sum(Lt[j:nj]) if dn else 0
                if dm == 0 and dn == 0: continue
                # cost
                if dm == 0 or dn == 0:
                    c = prior + (lt_sum if dn else lo_sum) * 0.01
                else:
                    expected = lo_sum * ratio
                    var = max(1.0, lo_sum * VAR_FACTOR)
                    z = abs(lt_sum - expected) / math.sqrt(var)
                    c = prior + z
                nc = dp[i][j] + c
                if nc < dp[ni][nj]:
                    dp[ni][nj] = nc
                    back[ni][nj] = (i, j, dm, dn)
    if dp[N][M] >= INF:
        return []
    # backtrack
    path = []
    i, j = N, M
    while back[i][j]:
        pi, pj, dm, dn = back[i][j]
        path.append({
            "orig": list(range(pi, i)),
            "trans": list(range(pj, j)),
            "move": f"{dm}:{dn}",
        })
        i, j = pi, pj
    path.reverse()
    return path

def length_ratio_confidence(o_text, t_text, ratio):
    """Heuristic confidence based only on length ratio fit. 0..1."""
    Lo, Lt = length(o_text), length(t_text)
    expected = Lo * ratio
    var = max(1.0, Lo * VAR_FACTOR)
    z = abs(Lt - expected) / math.sqrt(var)
    # convert z to a 0..1 score: z=0 -> 1.0, z>=4 -> ~0
    return round(max(0.0, 1.0 - z/4.0), 3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-jsonl", required=True)
    ap.add_argument("--trans-jsonl", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--max-segment", type=int, default=400)
    ap.add_argument("-o","--out", required=True)
    args = ap.parse_args()
    rows_o = [json.loads(l) for l in Path(args.orig_jsonl).read_text().splitlines() if l.strip()]
    rows_t = [json.loads(l) for l in Path(args.trans_jsonl).read_text().splitlines() if l.strip()]
    anchors = [json.loads(l) for l in Path(args.anchors).read_text().splitlines() if l.strip()]
    # global ratio from total chars
    total_o = sum(length(r["text"]) for r in rows_o)
    total_t = sum(length(r["text"]) for r in rows_t)
    global_ratio = total_t / max(1, total_o)
    print(f"  global ratio (char): {global_ratio:.3f}", file=sys.stderr)

    boundary = [{"orig":0,"trans":0}] + anchors + [{"orig":len(rows_o),"trans":len(rows_t)}]
    clean = [boundary[0]]
    for a in boundary[1:]:
        if a["orig"] > clean[-1]["orig"] and a["trans"] > clean[-1]["trans"]:
            clean.append(a)
    boundary = clean
    full = []; n_skip = 0
    for k in range(len(boundary)-1):
        a, b = boundary[k], boundary[k+1]
        o0, o1 = a["orig"], b["orig"]
        t0, t1 = a["trans"], b["trans"]
        n_o, n_t = o1-o0, t1-t0
        if n_o == 0 and n_t == 0: continue
        if n_o > args.max_segment or n_t > args.max_segment:
            full.append({"orig":list(range(o0,o1)),"trans":list(range(t0,t1)),
                         "move":f"{n_o}:{n_t}","confidence":0.0,"method":"slack_large"})
            n_skip += 1; continue
        if n_o == 0:
            for j in range(n_t):
                full.append({"orig":[],"trans":[t0+j],"move":"0:1","confidence":0.0,"method":"gap_trans"})
            continue
        if n_t == 0:
            for i in range(n_o):
                full.append({"orig":[o0+i],"trans":[],"move":"1:0","confidence":0.0,"method":"gap_orig"})
            continue
        # local ratio for this segment (smoother than global)
        seg_Lo = sum(length(rows_o[i]["text"]) for i in range(o0, o1))
        seg_Lt = sum(length(rows_t[j]["text"]) for j in range(t0, t1))
        seg_ratio = seg_Lt / max(1, seg_Lo)
        seg = align_segment(rows_o[o0:o1], rows_t[t0:t1], seg_ratio)
        for p in seg:
            p["orig"] = [o0+i for i in p["orig"]]
            p["trans"] = [t0+j for j in p["trans"]]
            if p["orig"] and p["trans"]:
                ot = " ".join(rows_o[i]["text"] for i in p["orig"])
                tt = " ".join(rows_t[j]["text"] for j in p["trans"])
                p["confidence"] = length_ratio_confidence(ot, tt, seg_ratio)
                p["method"] = "length"
            elif p["orig"]:
                p["confidence"] = 0.0; p["method"] = "gap_orig"
            else:
                p["confidence"] = 0.0; p["method"] = "gap_trans"
            full.append(p)
        if k % 50 == 0:
            print(f"  segment {k}/{len(boundary)-1}", file=sys.stderr)
    Path(args.out).write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in full))
    moves = {}
    for p in full:
        moves[p["move"]] = moves.get(p["move"], 0) + 1
    print(f"path len={len(full)} skipped_large={n_skip}", file=sys.stderr)
    print(f"top moves: {sorted(moves.items(), key=lambda x:-x[1])[:8]}", file=sys.stderr)

if __name__ == "__main__":
    main()
