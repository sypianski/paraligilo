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

# Patterns marking the *first real chapter* of a book — used to drop everything
# before it (cover, title page, dedication, foreword, preface, translator's
# note, table of contents, etc.) which rarely aligns across editions and
# poisons the deterministic anchor pass. Match is case-insensitive on the
# heading text after stripping punctuation.
BODY_START_PATTERNS = [
    # explicit chapter labels in 4 main langs
    re.compile(r"^(chapter|capitolo|rozdzia(?:[lł])|chapitre|ĉapitro|capítulo|kapitel)\s*(i{1,3}v?|iv|v|vi{0,3}|ix|x{1,3}|\d+|[a-z]?)\b", re.I),
    # bare arabic numeral chapter ("1", "1.", "I", "I.")
    re.compile(r"^(\d{1,3}\.?|[ivxlcdm]{1,5}\.?)$", re.I),
    # "PART ONE" / "BOOK ONE" / "PRIMA PARTE"
    re.compile(r"^(part|book|liber|libro|parte|część|partie)\b", re.I),
    # "Prologue" / "Prologo" - some books open with this as their real body
    re.compile(r"^(prologue|prologo|prolog|prólogo)\b", re.I),
]
# Heading texts that are NEVER the body start (front-matter markers)
FRONT_MATTER_HEADINGS = {
    "en": {"title page","cover","dedication","epigraph","contents","table of contents",
           "foreword","preface","introduction","translator's note","translator's preface",
           "author's note","author's preface","frontispiece","colophon",
           "copyright","copyright page","about this edition"},
    "it": {"copertina","frontespizio","indice","prefazione","introduzione",
           "nota del traduttore","nota dell'autore","colofone","dedica"},
    "pl": {"strona tytułowa","spis treści","przedmowa","wstęp","wprowadzenie",
           "od tłumacza","od autora","dedykacja","kolofon"},
    "fr": {"page de titre","couverture","table","table des matières","préface",
           "avant-propos","introduction","note du traducteur","colophon","dédicace"},
}

# Paragraph-level patterns that mark e-text platform / copyright boilerplate.
# Used to skip the front-matter when no explicit chapter heading is present.
BLURB_RE = re.compile(
    r"wolnelektury|wolne\s+lektury|fundacja|project\s+gutenberg|gutenberg\.org|"
    r"wikifontaro|wikisource|esperantista\s+centra\s+librejo|centra\s+librejo|"
    r"tous\s+droits|all\s+rights\s+reserved|represo\s+malpermesata|copyright|"
    r"©|creative\s+commons|public\s+domain|domaine\s+public|"
    r"this\s+ebook|ten\s+ebook|elŝutita\s+el|downloaded\s+from|"
    r"^rue\s+de\s+|^\d+,?\s+rue\s+|^d\s*ro\s+l\.?$|wszelkie\s+prawa|"
    r"el\s+pola\s+lingvo|el\s+franca\s+lingvo|el\s+angla\s+lingvo|el\s+rusa\s+lingvo|"
    r"el\s+german[ai]\s+lingvo|kun\s+permeso\s+de\s+la\s+aŭtorino|tradukita\s+de|"
    r"translated\s+(by|from)|traduit\s+(par|de)|tradotto\s+(da|dal)|"
    r"przełożył|przekład|tłumaczenie",
    re.IGNORECASE,
)
# Lines that look like all-caps title-page text (NAME, IMPRINT, FRONTISPIECE) —
# short (< 60 chars) and contain mostly uppercase letters with no sentence-end
# punctuation. Common on cover/title pages of older editions.
TITLEISH_RE = re.compile(r"^[A-ZĄĆĘŁŃÓŚŹŻÀ-ÖØ-Ý][A-ZĄĆĘŁŃÓŚŹŻĈĜĤĴŜŬÀ-ÖØ-Ý0-9 ,\.\-—–:;'\"()]*$")

def _is_titleish(text):
    s = text.strip()
    if len(s) > 60 or len(s) < 2: return False
    if s.endswith(".") or s.endswith("?") or s.endswith("!"):
        return False
    # mostly uppercase letters
    letters = [c for c in s if c.isalpha()]
    if not letters: return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio > 0.85 and bool(TITLEISH_RE.match(s))

def _heading_text(s):
    return (s or "").strip().lower().rstrip(":.,;")

def find_body_start(rows, lang):
    """Return index of the first row of real book body. 0 if no front-matter detected.

    Strategy, tried in order:
      1. Explicit chapter heading ("Rozdział I", "Chapter 1", "Prologo", …)
      2. First substantial body paragraph (>= MIN_BODY_PARA_CHARS) after we
         have already skipped past any platform-blurb rows (wolnelektury,
         wikifontaro, copyright, …) and other heading-only matter.
    """
    fm = FRONT_MATTER_HEADINGS.get(lang, FRONT_MATTER_HEADINGS["en"])

    # 1. explicit chapter heading
    for i, r in enumerate(rows[:300]):
        if not r.get("is_heading"):
            continue
        t = _heading_text(r["text"])
        if t in fm:
            continue
        for pat in BODY_START_PATTERNS:
            if pat.search(t):
                return i

    # 2. heuristic: walk the head of the book, skipping headings + blurb
    #    paragraphs (wolnelektury, copyright, colophon-style addresses).
    #    First row that's not a heading and not a blurb line == body start.
    #    Short aphoristic openers ("Życie kobiety to wiecznie…") count as body.
    last_skip_end = 0
    for i, r in enumerate(rows[:400]):
        text = r["text"]
        if r.get("is_heading"):
            last_skip_end = i + 1
            continue
        if BLURB_RE.search(text):
            last_skip_end = i + 1
            continue
        if _is_titleish(text):
            last_skip_end = i + 1
            continue
        return last_skip_end
    return 0

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

    # Drop front-matter: title page, dedication, foreword, etc. Those rarely
    # align across editions/translations and only generate noise misses.
    start = find_body_start(rows, lang)
    if start > 0:
        cut = rows[start:]
        head_text = rows[start]["text"][:60].replace("\n"," ")
        print(f"  body-start detected at row {start} ({len(rows)-start} of {len(rows)} kept) — heading: {head_text!r}", file=sys.stderr)
        return lang, cut
    print(f"  body-start: not detected, keeping all {len(rows)} rows", file=sys.stderr)
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
