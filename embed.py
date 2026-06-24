#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.0", "numpy", "python-dotenv"]
# ///
"""Embed sentences via OpenAI text-embedding-3-small. Disk-cached by sha256(text)."""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

MODELS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
BATCH = 100
CACHE = Path.home() / ".cache" / "splitread-align" / "embed.sqlite3"

def load_env():
    for p in [Path.home()/".hermes/.env", Path.home()/".env"]:
        if p.exists():
            load_dotenv(p, override=False)

def cache_open():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(CACHE)
    db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
    return db

def key_for(text, model):
    return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()

def fetch_cached(db, keys):
    out = {}
    for chunk in [keys[i:i+500] for i in range(0,len(keys),500)]:
        q = "SELECT key,vec FROM emb WHERE key IN (" + ",".join("?"*len(chunk)) + ")"
        for k, v in db.execute(q, chunk):
            out[k] = np.frombuffer(v, dtype=np.float32)
    return out

def store_cached(db, items):
    db.executemany("INSERT OR REPLACE INTO emb(key,vec) VALUES (?,?)",
                   [(k, v.astype(np.float32).tobytes()) for k,v in items])
    db.commit()

def embed_all(texts, client, db, model):
    dim = MODELS[model]
    keys = [key_for(t, model) for t in texts]
    cached = fetch_cached(db, keys)
    out = np.zeros((len(texts), dim), dtype=np.float32)
    missing_idx = []
    for i, k in enumerate(keys):
        if k in cached:
            out[i] = cached[k]
        else:
            missing_idx.append(i)
    if missing_idx:
        print(f"  embed[{model}]: {len(missing_idx)} missing (of {len(texts)}), calling API", file=sys.stderr)
        for start in range(0, len(missing_idx), BATCH):
            batch_idx = missing_idx[start:start+BATCH]
            batch_texts = [texts[i] for i in batch_idx]
            for attempt in range(3):
                try:
                    r = client.embeddings.create(model=model, input=batch_texts)
                    break
                except Exception as e:
                    if attempt == 2: raise
                    print(f"  retry {attempt+1}: {e}", file=sys.stderr); time.sleep(2*(attempt+1))
            new_items = []
            for j, datum in zip(batch_idx, r.data):
                v = np.array(datum.embedding, dtype=np.float32)
                out[j] = v
                new_items.append((keys[j], v))
            store_cached(db, new_items)
            if start % (BATCH*10) == 0:
                print(f"  embed: {start+len(batch_idx)}/{len(missing_idx)}", file=sys.stderr)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms==0] = 1
    out = out / norms
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("-o","--out", required=True)
    ap.add_argument("--model", default="text-embedding-3-small", choices=list(MODELS))
    args = ap.parse_args()
    load_env()
    client = OpenAI()
    rows = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    texts = [r["text"] for r in rows]
    db = cache_open()
    vecs = embed_all(texts, client, db, args.model)
    np.save(args.out, vecs)
    print(f"embedded[{args.model}] {len(texts)} -> {args.out} ({vecs.shape})", file=sys.stderr)

if __name__ == "__main__":
    main()
