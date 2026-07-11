# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`paraligilo` sentence-aligns an EPUB against its translation and emits a JSONL
mapping every source sentence to its counterpart(s), with confidence scores and
honest `m:n` moves (`1:1`, `1:2`, `2:1`, `1:0`, `0:1`, …). The primary consumer
is the [rosetta KOReader plugin](https://github.com/sypianski/rosetta.koplugin):
`emit_rosetta.py` converts an alignment into the `parallel-aligned.json` schema
([parallel-align-spec](https://github.com/sypianski/parallel-align-spec)) that
rosetta loads from a book's `.sdr/` directory. `README.md` has the user-facing
overview; `DEPLOY.md` has the VPS/nginx/systemd deployment. This file is the map
for editing the code.

## Architecture: one script per stage

There is **no package and no build step**. Each `.py` at the repo root is a
standalone [uv](https://github.com/astral-sh/uv) inline script — dependencies
are declared in its `# /// script` header and installed by uv on first run. Run
any stage directly (`./extract.py book.epub -o book.jsonl`). You only need `uv`
on `PATH`; the AI path additionally needs an OpenAI-compatible key.

`run.py` is a thin orchestrator that shells out to the stage scripts via
`subprocess` — it holds **no alignment logic**, only wiring and filename
conventions (`<name>_orig.jsonl`, `<name>_align_concat.jsonl`, …) and a final
`<name>_final.jsonl` symlink to the last stage produced. Because stages
communicate only through files, you can re-run one stage in isolation without
replaying the pipeline (see README "Individual stages").

Two execution paths share stage 1 (extract) and the final verify, but are
otherwise disjoint — pick by `--mode`:

| Stage | AI mode (`--mode ai`, default) | Deterministic mode (`--mode deterministic`) |
|---|---|---|
| 1. extract | `extract.py` | `extract.py` |
| 2. embed | `embed.py` (OpenAI embeddings → `.npy`, sqlite-cached) | — |
| 3. anchors | `anchors.py` (k-NN + LIS monotonic chain) | `anchors_det.py` (surface-token anchors) |
| 4. align | `align_full.py` (embedding-scored Gale-Church m:n DP) | `align_det.py` (length-only Gale-Church DP) |
| 5. rescore | `rescore_concat.py` (re-embed concatenated m:n text) | — |
| 6. refine | `refine.py` (pass-2 using high-conf 1:1 as new anchors) | — |
| 7. llm | `rescore_llm.py` (gpt-4o-mini honest-labels low-conf) `--llm` | — |
| 8. verify | `verify.py` → stratified `sample.md` | same |

`align.py` is the shared m:n DP core used by the align stages. Deterministic
mode is free, offline, needs no key — worse quality but usable where an API key
is unacceptable.

`emit_rosetta.py` is a **post-processing step, not part of `run.py`** — it takes
the two `_orig.jsonl`/`_trans.jsonl` sentence files plus `_final.jsonl` and
produces `parallel-aligned.json`. It hashes normalized paragraph text into a
`by_hash` lookup; the normalization here (footnote-marker stripping, quote/dash
folding, invisible-char removal — see the `QUOTE_FOLD` table) must stay in sync
with how crengine renders text on the reader, or rosetta's runtime lookups miss.

`slice_proportional.py` is an **offline prototype** of the rosetta plugin's
runtime page-slicing logic (given a rendered source page, produce the mirrored
target slice by proportional coverage). It exists here so the algorithm can be
regression-tested against goldens before being ported into the Lua plugin.

## Commands

```bash
# Full pipeline (AI, recommended): produces <name>_final.jsonl + <name>_sample.md
./run.py orig.epub trans.epub --name mybook
./run.py orig.epub trans.epub --name mybook --llm          # + honest low-conf labels
./run.py orig.epub trans.epub --name mybook --mode deterministic   # free, offline, no key

# Convert an alignment into the rosetta plugin's JSON (see README for full flags)
./emit_rosetta.py --orig-jsonl mybook_orig.jsonl --trans-jsonl mybook_trans.jsonl \
    --align mybook_final.jsonl --src-lang it --tgt-lang pl \
    --src-file orig.epub --tgt-file trans.epub -o out/orig.epub.sdr/parallel-aligned.json
```

Keys are read from the environment or `~/.hermes/.env` / `~/.env`. Any
OpenAI-compatible endpoint works via `OPENAI_BASE_URL` (OpenRouter, Together,
vLLM, …); the code uses the official `openai` SDK.

## Tests

The only automated suite is the `slice_proportional` regression harness:

```bash
python3 -m pytest tests/regression/ -v          # asserts IoU / Levenshtein floors
python3 tests/regression/evaluate.py            # human-readable quality report
```

Goldens live in `tests/regression/goldens/marta_*.json` (user-curated PL→EO page
slices) and `tests/regression/pi_pages/*.txt` (raw KOReader page captures). A
change to `extract.py` normalization or `emit_rosetta.py` hashing that regresses
page slicing surfaces here — run it after touching either. There is no unit test
for the alignment stages themselves; quality is judged from the stratified
`verify.py` sample and the measured-quality table in `README.md`.

## Web service

`web/app.py` is a FastAPI single-file service (also a uv inline script) that
wraps the CLI: upload two EPUBs → background `asyncio` subprocess runs `run.py`
then `emit_rosetta.py` → poll `/status` → download a zip. All operational knobs
are `PARALIGILO_*` env vars (jobs dir, TTL, upload cap, per-IP rate limits,
daily quotas, `PARALIGILO_ROOT_PATH` for sub-path mounts). If `OPENAI_API_KEY`
is set in the service env it becomes an opt-in "run on my key" fallback — **the
operator pays** for those calls; `DEPLOY.md` covers the budget-cap safeguards.
Deployment is systemd + nginx; there is no build/bundle step, `git pull` +
restart is the whole update.

## Editing notes

- **New language**: extend the abbreviation table `extract.py:ABBREV` so
  sentence segmentation doesn't split on `Mr.`/`etc.`-style abbreviations. The
  embedding model is multilingual out of the box; only segmentation is
  language-specific.
- **Changing a stage's CLI** ripples to `run.py` (which builds the argv) and to
  `web/app.py` (which lists stage names for progress reporting) — update both.
- **`emit_rosetta.py` normalization is a contract with the reader**, not a free
  choice: any change to `QUOTE_FOLD` / footnote stripping must be validated
  against the regression suite, since it changes the `by_hash` keys rosetta
  looks up at runtime.
- Gitignore excludes all pipeline artifacts (`*.jsonl`, `*.npy`, `*.epub`,
  `*.sdr/`, `segev_*`, `sample*.md`). The committed `segev_*.jsonl` files are
  intentional fixtures — don't add new large artifacts.
- Open work is tracked in `TODO.md` (dialogue-tag anchors, page-boundary
  paragraph reassembly, in-memory cache reset — several belong in the rosetta
  plugin, not here).
```
