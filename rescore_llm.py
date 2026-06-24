#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.0", "python-dotenv", "numpy"]
# ///
"""LLM rescore: for low-confidence pairs, ask gpt-4o-mini if they're a valid
translation alignment. Updates confidence based on verdict."""
import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from embed import load_env
from openai import OpenAI

MODEL = "gpt-4o-mini"

PROMPT = """You judge whether two text snippets are aligned translations.

A = source text
B = candidate translation

Verdict rules:
- "yes" — B clearly conveys the SAME content as A (paraphrase OK; word-for-word not required)
- "partial" — B overlaps with A but adds, omits, or shifts meaning significantly
- "no" — B is unrelated, talks about different content, or is a translation of a different passage

Reply with ONLY one word: yes, partial, or no."""

VERDICT_TO_CONF = {"yes": 0.85, "partial": 0.50, "no": 0.10}

def judge(client, a_text, b_text):
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": f"A: {a_text}\n\nB: {b_text}"},
                ],
                temperature=0,
                max_tokens=5,
            )
            v = r.choices[0].message.content.strip().lower().split()[0].rstrip(".,;:")
            if v in VERDICT_TO_CONF:
                return v
            return "partial"
        except Exception as e:
            if attempt == 2:
                print(f"  judge fail: {e}", file=sys.stderr)
                return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-jsonl", required=True)
    ap.add_argument("--trans-jsonl", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--max-conf", type=float, default=0.50,
                    help="Rescore pairs with confidence below this")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("-o","--out", required=True)
    args = ap.parse_args()

    rows_o = [json.loads(l) for l in Path(args.orig_jsonl).read_text().splitlines() if l.strip()]
    rows_t = [json.loads(l) for l in Path(args.trans_jsonl).read_text().splitlines() if l.strip()]
    path = [json.loads(l) for l in Path(args.align).read_text().splitlines() if l.strip()]

    cands = []
    for idx, p in enumerate(path):
        if p["method"] not in ("embed", "embed_concat"):
            continue
        if not p["orig"] or not p["trans"]:
            continue
        if p["confidence"] >= args.max_conf:
            continue
        cands.append(idx)
    print(f"  LLM-rescoring {len(cands)} pairs (conf<{args.max_conf})", file=sys.stderr)

    load_env()
    client = OpenAI()

    counts = {"yes": 0, "partial": 0, "no": 0, "fail": 0}
    def work(idx):
        p = path[idx]
        a = " ".join(rows_o[i]["text"] for i in p["orig"])
        b = " ".join(rows_t[j]["text"] for j in p["trans"])
        v = judge(client, a, b)
        return idx, v

    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for idx, v in ex.map(work, cands):
            done += 1
            if v is None:
                counts["fail"] += 1
                path[idx]["method"] = "llm_fail"
            else:
                counts[v] += 1
                # blend embedding sim with LLM verdict — keep max
                new_conf = VERDICT_TO_CONF[v]
                old_conf = path[idx]["confidence"]
                path[idx]["confidence"] = round(max(old_conf, new_conf) if v == "yes" else new_conf, 3)
                path[idx]["method"] = f"llm_{v}"
            if done % 100 == 0:
                print(f"  rescored {done}/{len(cands)}  {counts}", file=sys.stderr)

    print(f"  verdicts: {counts}", file=sys.stderr)
    Path(args.out).write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in path))

if __name__ == "__main__":
    main()
