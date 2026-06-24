#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["ebooklib", "beautifulsoup4", "lxml", "openai>=1.0", "numpy", "python-dotenv"]
# ///
"""End-to-end pipeline: orig.epub + trans.epub -> alignment.jsonl

Stages (all on by default except --llm):
  1. extract     EPUB -> sentences (noise+segmentation filter)
  2. embed       sentences -> vectors (text-embedding-3-large default)
  3. anchors     k-NN + LIS monotonic anchors
  4. align       Gale-Church m:n DP per anchor segment
  5. rescore     re-embed concatenations for low-conf m:n
  6. refine      pass-2 with high-conf 1:1 as added anchors  [--no-pass2 to skip]
  7. llm         gpt-4o-mini honest rescore of low-conf       [--llm to enable]
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

def run(*cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", file=sys.stderr)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        sys.exit(r.returncode)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("orig_epub")
    ap.add_argument("trans_epub")
    ap.add_argument("--name", required=True, help="output basename (e.g. segev)")
    ap.add_argument("--workdir", default=str(HERE), help="output dir")
    ap.add_argument("--mode", default="ai", choices=["ai", "deterministic"],
                    help="ai: OpenAI embeddings (~$0.10/book); deterministic: free, offline, no key")
    ap.add_argument("--model", default="text-embedding-3-large",
                    choices=["text-embedding-3-small", "text-embedding-3-large"])
    ap.add_argument("--stride", type=int, default=10, help="(ai) anchor stride")
    ap.add_argument("--anchor-threshold", type=float, default=0.70)
    ap.add_argument("--no-rescore", action="store_true",
                    help="(ai) skip concat re-embed for low-conf m:n")
    ap.add_argument("--no-pass2", action="store_true",
                    help="(ai) skip two-pass refinement")
    ap.add_argument("--llm", action="store_true",
                    help="(ai) run optional gpt-4o-mini honest rescore (extra cost)")
    ap.add_argument("--llm-max-conf", type=float, default=0.50)
    ap.add_argument("--sample", type=int, default=50,
                    help="rows to put in verification sample.md (0=skip)")
    args = ap.parse_args()

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    base = wd / args.name

    o_jsonl = base.with_name(f"{args.name}_orig.jsonl")
    t_jsonl = base.with_name(f"{args.name}_trans.jsonl")
    o_npy   = base.with_name(f"{args.name}_orig.npy")
    t_npy   = base.with_name(f"{args.name}_trans.npy")
    anchors = base.with_name(f"{args.name}_anchors.jsonl")
    align1  = base.with_name(f"{args.name}_align.jsonl")
    sample  = base.with_name(f"{args.name}_sample.md")

    # 1. extract (both modes)
    run(HERE/"extract.py", args.orig_epub,  "-o", o_jsonl)
    run(HERE/"extract.py", args.trans_epub, "-o", t_jsonl)

    if args.mode == "deterministic":
        # No-AI path: surface-token anchors + length-only Gale-Church DP
        run(HERE/"anchors_det.py",
            "--orig-jsonl", o_jsonl, "--trans-jsonl", t_jsonl, "-o", anchors)
        run(HERE/"align_det.py",
            "--orig-jsonl", o_jsonl, "--trans-jsonl", t_jsonl,
            "--anchors", anchors, "-o", align1)
        current = align1
    else:
        # AI path
        # 2. embed
        run(HERE/"embed.py", o_jsonl, "-o", o_npy, "--model", args.model)
        run(HERE/"embed.py", t_jsonl, "-o", t_npy, "--model", args.model)
        # 3. anchors
        run(HERE/"anchors.py", "--orig-npy", o_npy, "--trans-npy", t_npy,
            "--stride", args.stride, "--threshold", args.anchor_threshold,
            "-o", anchors)
        # 4. align
        run(HERE/"align_full.py",
            "--orig-jsonl", o_jsonl, "--trans-jsonl", t_jsonl,
            "--orig-npy", o_npy, "--trans-npy", t_npy,
            "--anchors", anchors, "-o", align1)
        # 5. rescore concat
        current = align1
        if not args.no_rescore:
            align2 = base.with_name(f"{args.name}_align_concat.jsonl")
            run(HERE/"rescore_concat.py",
                "--orig-jsonl", o_jsonl, "--trans-jsonl", t_jsonl,
                "--align", current, "--model", args.model, "-o", align2)
            current = align2
        # 6. pass2
        if not args.no_pass2:
            align3 = base.with_name(f"{args.name}_align_pass2.jsonl")
            anchors2 = base.with_name(f"{args.name}_anchors_pass2.jsonl")
            run(HERE/"refine.py",
                "--orig-jsonl", o_jsonl, "--trans-jsonl", t_jsonl,
                "--orig-npy", o_npy, "--trans-npy", t_npy,
                "--align", current, "--anchors", anchors,
                "-o", align3, "--anchors-out", anchors2)
            if not args.no_rescore:
                align3b = base.with_name(f"{args.name}_align_pass2_concat.jsonl")
                run(HERE/"rescore_concat.py",
                    "--orig-jsonl", o_jsonl, "--trans-jsonl", t_jsonl,
                    "--align", align3, "--model", args.model, "-o", align3b)
                current = align3b
            else:
                current = align3
        # 7. LLM (optional)
        if args.llm:
            align_llm = base.with_name(f"{args.name}_align_llm.jsonl")
            run(HERE/"rescore_llm.py",
                "--orig-jsonl", o_jsonl, "--trans-jsonl", t_jsonl,
                "--align", current, "--max-conf", args.llm_max_conf,
                "-o", align_llm)
            current = align_llm

    # final symlink
    final = base.with_name(f"{args.name}_final.jsonl")
    if final.exists() or final.is_symlink():
        final.unlink()
    final.symlink_to(current.name)
    print(f"\nFINAL: {final} -> {current.name}", file=sys.stderr)

    # 8. sample
    if args.sample > 0:
        run(HERE/"verify.py",
            "--orig-jsonl", o_jsonl, "--trans-jsonl", t_jsonl,
            "--align", current, "-n", args.sample, "-o", sample)

if __name__ == "__main__":
    main()
