# paraligilo

Sentence-level alignment of an EPUB and its translation. Outputs a JSONL
mapping every source sentence to its counterpart(s) in the translation,
with confidence scores. Handles m:n alignments (1:1, 1:2, 2:1, 2:2, 1:0,
0:1) — translators routinely split or merge sentences, and pretending
otherwise breaks the alignment.

Primary consumer is the [rosetta KOReader plugin](https://github.com/sypianski/koreader-konfig/tree/master/koplugins/rosetta.koplugin)
which shows the aligned human translation in a panel beneath the original
text — see `emit_rosetta.py` for the post-processing step that converts a
paraligilo alignment into the JSON format rosetta loads from a book's
`.sdr/`. The alignment output is equally usable for parallel corpora,
bilingual readers, translation studies, language-learning material.

## What it does

```
orig.epub + trans.epub
   │
   ├─ extract     EPUB → sentences (language-aware segmentation,
   │              drops Notes/Bibliography, filters OCR junk)
   ├─ embed       sentences → vectors (OpenAI text-embedding-3-large)
   ├─ anchors     k-NN + LIS monotonic anchor chain
   ├─ align       Gale-Church-style m:n DP, per anchor segment
   ├─ rescore     re-embed actual concatenated text for m:n pairs
   │              (mean-pool is lossy)
   ├─ refine      pass-2 using high-confidence 1:1 pairs as new anchors
   └─ llm         (optional) gpt-4o-mini sanity-checks low-conf pairs
                  and honestly labels false alignments
```

Output: `<name>_final.jsonl`, one record per alignment:

```json
{
  "orig": [4915, 4916],
  "trans": [4125, 4126],
  "move": "2:2",
  "confidence": 0.703,
  "method": "embed_concat"
}
```

## Measured quality

Tested on two real books:

| Pair | high (≥0.7) | med (0.5–0.7) | high+med | Total pairs |
|---|---|---|---|---|
| Tom Segev *Seventh Million* — EN ↔ IT (modern nonfiction) | 41% | 45% | **86%** | 6641 |
| Sienkiewicz *Quo Vadis* — PL ↔ EN (classic translation) | 23% | 71% | **94%** | 6177 |

Cost per book pair: **~$0.10** (text-embedding-3-large + optional gpt-4o-mini).
Time: ~30s end-to-end without LLM, ~5min with LLM (mostly waiting on
sequential API calls).

The remaining 6–14% lands in low / very-low / gap bands. With `--llm`
enabled these are honestly labeled — a 72%-majority of "low confidence"
pairs turn out to be genuine non-alignments (translator dropped content,
added editorial notes, etc.), not algorithm failures. Display them as
weak/unavailable rather than pretending.

Translator style matters more than language pair. Classic faithful
translations (Curtin) score better than modern restructured ones (Segev).

## Install

Each script is a standalone [uv](https://github.com/astral-sh/uv) inline
script — dependencies declared in a `# /// script` header, installed on
first run. You only need `uv` in `PATH` and an API key for the AI mode
(deterministic mode needs no key).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
export OPENAI_API_KEY=sk-...                       # or any compatible key
# optional: point at a non-OpenAI endpoint
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

Any OpenAI-compatible endpoint works — OpenAI itself, OpenRouter, Together,
Groq, Mistral, a self-hosted vLLM/Ollama, &c. The pipeline uses the
official `openai` Python SDK, which honours `OPENAI_BASE_URL`. Variables
are also picked up from `~/.hermes/.env` or `~/.env`.

## Use

```bash
./run.py orig.epub trans.epub --name mybook
# → mybook_final.jsonl  +  mybook_sample.md (50 stratified examples)
```

With LLM honest rescore (recommended, +$0.005/book):

```bash
./run.py orig.epub trans.epub --name mybook --llm
```

Cheaper model (text-embedding-3-small, ~$0.015/book, slightly weaker):

```bash
./run.py orig.epub trans.epub --name mybook --model text-embedding-3-small
```

All flags:

```
--name NAME              output basename (required)
--workdir WORKDIR        output dir (default: cwd)
--model {small,large}    embedding model
--stride STRIDE          anchor scan stride (default 10)
--anchor-threshold X     min cos sim for anchor (default 0.70)
--no-rescore             skip concat re-embed
--no-pass2               skip two-pass refinement
--llm                    enable gpt-4o-mini honest rescore
--llm-max-conf X         rescore pairs with conf below this (default 0.50)
--sample N               sample N pairs to verification markdown (0=skip)
```

## Feeding the result into rosetta

```bash
./emit_rosetta.py \
    --orig-jsonl  mybook_orig.jsonl \
    --trans-jsonl mybook_trans.jsonl \
    --align       mybook_final.jsonl \
    --src-lang it --tgt-lang pl \
    --src-file orig.epub --tgt-file trans.epub \
    --method hybrid \
    -o /path/to/orig.epub.sdr/rosetta-aligned.json
```

Copy the resulting `.sdr/` directory (Syncthing or USB) to your eink
reader. Rosetta loads `rosetta-aligned.json` automatically and serves the
human translation paragraph-by-paragraph instead of calling AI.

Schema: see `ANCHOR_FORMAT.md` in the rosetta repo.

## Individual stages

Each step is its own script — useful for re-running just one part:

```bash
./extract.py book.epub -o book.jsonl
./embed.py book.jsonl -o book.npy --model text-embedding-3-large
./anchors.py --orig-npy a.npy --trans-npy b.npy -o anchors.jsonl
./align_full.py --orig-jsonl a.jsonl --trans-jsonl b.jsonl \
                --orig-npy a.npy --trans-npy b.npy \
                --anchors anchors.jsonl -o align.jsonl
./rescore_concat.py --orig-jsonl a.jsonl --trans-jsonl b.jsonl \
                    --align align.jsonl --model text-embedding-3-large \
                    -o align_concat.jsonl
./rescore_llm.py    --orig-jsonl a.jsonl --trans-jsonl b.jsonl \
                    --align align_concat.jsonl -o align_llm.jsonl
./verify.py         --orig-jsonl a.jsonl --trans-jsonl b.jsonl \
                    --align align_llm.jsonl -n 50 -o sample.md
```

## Caching

Embeddings cached in `~/.cache/splitread-align/embed.sqlite3`, keyed by
sha256(model || text). Re-running on a touched book hits the cache for
unchanged sentences. Across a library, common short sentences ("Yes."
"He laughed.") deduplicate naturally.

## Caveats

- **Verify your input pair is actually a translation pair.** It is easy
  to mistake two different books by the same author for original +
  translation; the algorithm will then produce confident-looking nonsense
  for low-conf bands. Spot-check a known scene before trusting output.
- **Heavily abridged translations are out of scope** — the algorithm will
  emit many `1:0` / `0:1` gaps and many low-confidence rows. That is the
  correct behavior; do not raise thresholds to hide it.
- **Poetry, idioms, song lyrics, quoted passages from other works**
  often have non-sentence-preserving translations. Mark `weak` and don't
  fight it.

## Languages tested

EN, IT, PL, FR. Sentence-segmentation tables in `extract.py:ABBREV`
(extend for new languages). Embedding model is multilingual and works for
50+ languages out of the box.

## License

AGPL-3.0.
