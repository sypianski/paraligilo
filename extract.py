#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["ebooklib", "beautifulsoup4", "lxml"]
# ///
"""EPUB -> JSONL of {chapter, para, sent, text} with language-aware sentence segmentation."""
import argparse
import json
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib")

from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT

ABBREV = {
    "en": {"mr","mrs","ms","dr","prof","st","vs","etc","e.g","i.e","cf","p","pp","jr","sr","inc","ltd","no","fig","vol"},
    "it": {"sig","sig.ra","dott","dott.ssa","prof","avv","ing","ecc","cfr","pag","pagg","sec","art","cap","ed","es","sec","san","sant"},
    "pl": {"p","pp","tj","np","tzn","tzw","m.in","ok","r","w","ul","prof","dr","inż","mgr","św","bł","gen","płk","mjr","kpt","por","ppor","ks","im","nr","str"},
    "fr": {"m","mme","mlle","dr","pr","st","ste","etc","cf","p","pp","éd","vol","chap","art","av","apr","j.-c","jc","ier","ed","fig","no","nos"},
}

SENT_END = re.compile(r"([\.!\?…]+)([\"»›'\)\]]*)\s+")

# Drop entire blocks whose heading text matches these (case-insensitive).
# Once seen, drop everything until the NEXT non-skipped heading.
SKIP_SECTIONS = {
    "en": {"notes", "note", "bibliography", "references", "index", "works cited",
           "select bibliography", "selected bibliography", "further reading",
           "abbreviations", "list of abbreviations", "acknowledgments", "acknowledgements",
           "list of illustrations", "illustrations", "maps", "list of maps",
           "about the author", "copyright", "permissions"},
    "it": {"note", "bibliografia", "indice", "riferimenti", "abbreviazioni",
           "ringraziamenti", "fonti", "appendice", "illustrazioni", "copyright",
           "indice analitico", "indice dei nomi", "ulteriori letture"},
    "pl": {"przypisy", "bibliografia", "indeks", "skróty", "podziękowania",
           "spis treści", "spis ilustracji", "literatura", "źródła"},
    "fr": {"notes", "bibliographie", "index", "abréviations", "remerciements",
           "table des matières", "table des illustrations", "sources",
           "pour aller plus loin"},
}

# Treat as junk: page numbers, footnote markers, citation fragments, OCR debris.
NOISE_PATTERNS = [
    re.compile(r"^[\d\s,;:.()\[\]\-–—]+$"),       # only digits + punctuation
    re.compile(r"^[ivxlcdm]+\.?$", re.I),          # roman numerals alone
    re.compile(r"^\d+\s*[a-z]?\.?\s*$", re.I),     # "26", "26a.", "26."
    re.compile(r"^[*†‡§¶#]+\d*$"),                  # footnote markers
    re.compile(r"^pp?\.\s*\d+[-–]?\d*\s*\.?$"),    # "p. 12", "pp. 12-15"
    re.compile(r"^vol\.?\s*[ivxlcdm\d]+\.?$", re.I),
    re.compile(r"^[A-Z]{2,}\s*,?\s*[IVXLCDM]+\s*\.?$"),  # "DK, XLI" cite shorthand
]

# Min sentence length: shorter = drop unless ends with strong punctuation
# (legitimate short sentences like "He laughed." still kept by punctuation check)
MIN_SENT_CHARS = 12

def get_book_lang(book):
    md = book.get_metadata("DC", "language")
    if md:
        v = md[0][0].lower()
        return v.split("-")[0]
    return "en"

def html_to_blocks(html):
    """Return list of (tag, text) for block elements in order."""
    soup = BeautifulSoup(html, "lxml")
    for bad in soup(["script","style","head","nav","header","footer","img","svg"]):
        bad.decompose()
    blocks = []
    block_tags = {"p","h1","h2","h3","h4","h5","h6","li","blockquote","dt","dd","div"}
    for el in soup.find_all(block_tags):
        if el.find(block_tags):
            continue
        text = el.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        if text:
            blocks.append((el.name.lower(), text))
    return blocks

def is_noise(text):
    if len(text.strip()) < 3:
        return True
    for pat in NOISE_PATTERNS:
        if pat.match(text):
            return True
    # mostly digits + punctuation, no letters
    letters = sum(1 for c in text if c.isalpha())
    if letters < 3:
        return True
    if letters / max(1, len(text)) < 0.30:
        return True
    return False

def split_sentences(text, lang):
    abbr = ABBREV.get(lang, ABBREV["en"])
    parts = []
    last = 0
    for m in SENT_END.finditer(text):
        end = m.end()
        prefix = text[last:m.start()]
        last_word = re.split(r"[\s\(\[\"«]", prefix)[-1].rstrip(".").lower()
        if last_word in abbr:
            continue
        if re.match(r"^\d+$", last_word):
            continue
        sentence = (text[last:m.start()] + m.group(1) + m.group(2)).strip()
        if sentence:
            parts.append(sentence)
        last = end
    tail = text[last:].strip()
    if tail:
        parts.append(tail)
    # filter: drop noise, drop very short fragments that don't end in . ! ?
    cleaned = []
    for s in parts:
        if is_noise(s):
            continue
        if len(s) < MIN_SENT_CHARS and not s.rstrip()[-1:] in ".!?…»\")":
            continue
        cleaned.append(s)
    return cleaned

HEADING_TAGS = {"h1","h2","h3","h4","h5","h6"}

def is_skip_heading(text, lang):
    skip = SKIP_SECTIONS.get(lang, SKIP_SECTIONS["en"])
    t = text.strip().lower().rstrip(":.")
    # also strip leading "chapter N", "appendix" etc.
    t = re.sub(r"^(appendix|annex|chapter|capitolo|rozdział|chapitre)\s+[ivxlcdm0-9]+\s*[.:]?\s*", "", t)
    return t in skip

def extract(epub_path):
    book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})
    lang = get_book_lang(book)
    rows = []
    chapter_idx = 0
    para_idx = 0
    saw_heading_this_chapter = False
    skipping = False  # True while inside a SKIP_SECTIONS chapter
    n_skipped_blocks = 0
    n_noise = 0
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        blocks = html_to_blocks(item.get_content())
        for tag, text in blocks:
            is_heading = tag in HEADING_TAGS
            if is_heading and saw_heading_this_chapter:
                chapter_idx += 1
                para_idx = 0
                saw_heading_this_chapter = False
            if is_heading:
                saw_heading_this_chapter = True
                # toggle skipping based on heading text
                if is_skip_heading(text, lang):
                    skipping = True
                else:
                    skipping = False
            if skipping:
                n_skipped_blocks += 1
                continue
            sents = split_sentences(text, lang)
            if not sents:
                if text.strip():
                    n_noise += 1
                continue
            for sent_idx, sent in enumerate(sents):
                rows.append({
                    "chapter": chapter_idx,
                    "para": para_idx,
                    "sent": sent_idx,
                    "tag": tag,
                    "is_heading": is_heading,
                    "text": sent,
                })
            para_idx += 1
    print(f"  skipped blocks (notes/biblio/etc.): {n_skipped_blocks}", file=sys.stderr)
    print(f"  filtered noise blocks: {n_noise}", file=sys.stderr)
    return lang, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("epub")
    ap.add_argument("-o","--out", required=True)
    args = ap.parse_args()
    lang, rows = extract(Path(args.epub))
    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print(f"lang={lang} blocks={1+max((r['chapter'] for r in rows), default=-1)} sentences={len(rows)} -> {out}", file=sys.stderr)

if __name__ == "__main__":
    main()
