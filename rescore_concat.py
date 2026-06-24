#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.0", "numpy", "python-dotenv"]
# ///
"""Post-process alignment: for m:n pairs (m+n >= 3) with low confidence,
re-embed actual concatenated text and update confidence with the true cos sim."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from embed import cache_open, embed_all, load_env
from openai import OpenAI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-jsonl", required=True)
    ap.add_argument("--trans-jsonl", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--model", default="text-embedding-3-large")
    ap.add_argument("--threshold", type=float, default=0.65,
                    help="Re-embed only pairs with confidence below this AND m+n >= 3")
    ap.add_argument("-o","--out", required=True)
    args = ap.parse_args()

    rows_o = [json.loads(l) for l in Path(args.orig_jsonl).read_text().splitlines() if l.strip()]
    rows_t = [json.loads(l) for l in Path(args.trans_jsonl).read_text().splitlines() if l.strip()]
    path = [json.loads(l) for l in Path(args.align).read_text().splitlines() if l.strip()]

    # collect candidates
    cands = []
    for idx, p in enumerate(path):
        if p["method"] != "embed":
            continue
        if not p["orig"] or not p["trans"]:
            continue
        m, n = len(p["orig"]), len(p["trans"])
        if m + n < 3:
            continue
        if p["confidence"] >= args.threshold:
            continue
        cands.append(idx)

    print(f"  re-embedding {len(cands)} m:n pairs (m+n>=3, conf<{args.threshold})", file=sys.stderr)
    if not cands:
        Path(args.out).write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in path))
        return

    # build concat texts
    orig_texts = []
    trans_texts = []
    for idx in cands:
        p = path[idx]
        orig_texts.append(" ".join(rows_o[i]["text"] for i in p["orig"]))
        trans_texts.append(" ".join(rows_t[j]["text"] for j in p["trans"]))

    load_env()
    client = OpenAI()
    db = cache_open()
    vo = embed_all(orig_texts, client, db, args.model)
    vt = embed_all(trans_texts, client, db, args.model)

    n_up = 0; n_down = 0; n_unchanged = 0
    delta_sum = 0.0
    for k, idx in enumerate(cands):
        new_conf = float(np.dot(vo[k], vt[k]))
        old_conf = path[idx]["confidence"]
        if new_conf > old_conf + 0.01:
            n_up += 1
        elif new_conf < old_conf - 0.01:
            n_down += 1
        else:
            n_unchanged += 1
        delta_sum += (new_conf - old_conf)
        path[idx]["confidence"] = round(new_conf, 3)
        path[idx]["method"] = "embed_concat"

    print(f"  up: {n_up}  unchanged: {n_unchanged}  down: {n_down}  avg Δ: {delta_sum/len(cands):+.3f}", file=sys.stderr)
    Path(args.out).write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in path))

if __name__ == "__main__":
    main()
