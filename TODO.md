# paraligilo TODO

## Aligned-mode improvements

- [ ] **Dialogue-tag anchors** (deterministic-mode): scan for paragraphs
      that contain dialogue tags ("— Marta diris", "— Anna respondis",
      "— rzekł", "— powiedziała", "— diris", "— respondis") and use
      proper-noun adjacency as a stronger anchor source. Should help on
      books with sparse year/number anchors but dense dialogue (most
      novels).

- [ ] **Page-boundary paragraph reassembly** (rosetta plugin): when
      lookupBlock fails on the first paragraph of a page because it's a
      continuation of one started on the previous page, fall back to
      either:
        (a) prefix/suffix-hashed entries stored alongside the full-para
            entry in parallel-aligned.json
        (b) substring scan over by_hash keys (slow but works)
        (c) read full paragraph via xpointer expansion before hashing

- [ ] **In-memory cache reset**: rosetta plugin holds Aligned._cache for
      the book file's lifetime — make it observe a mtime check on the
      JSON so users dropping a freshly-regenerated file get the new
      data without a full KOReader restart.
