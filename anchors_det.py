#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Deterministic anchor detection — no embeddings, no API.

Anchors found by intersection of stable surface tokens:
  - years (1850-2099)
  - 4+-digit numbers
  - proper-noun n-grams (capitalized words >= 4 chars), filtered to rare ones
  - rare-token co-occurrences

Then monotonic-filtered via LIS to drop outliers.
"""
import argparse
import bisect
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
BIGNUM = re.compile(r"\b\d{3,}\b")
# proper noun: cap letter (incl diacritics), then 3+ lowercase letters
PROPER = re.compile(r"\b([A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ]{3,})\b")
SENT_START_STOP = re.compile(r"[\.!\?…»\"\)\]]\s+$")

# Function words that often appear capitalized at sentence start but aren't proper nouns.
STOPCAPS = {
    "en": {"The","A","An","And","But","Or","So","Yet","For","Nor","Of","To","In","On","At","By","With","From","As","If","When","While","Where","Who","Whom","Whose","Which","That","This","These","Those","Here","There","Then","Now","Also","Even","Only","Just","Still","Very","More","Most","Some","Any","Each","Every","All","Both","Either","Neither","Many","Much","Few","Several","Such","Same","Other","Another","First","Last","Next","Their","His","Her","Its","Our","Your","My","Was","Were","Has","Have","Had","Did","Does","Will","Would","Could","Should","Might","Must","Can","Shall","May","One","Two","Three"},
    "it": {"Il","La","Lo","Le","Gli","I","Un","Una","Uno","E","O","Ma","Però","Tuttavia","Di","A","Da","In","Con","Su","Per","Tra","Fra","Che","Chi","Cui","Quale","Quando","Mentre","Dove","Come","Se","Anche","Solo","Soltanto","Ancora","Già","Più","Meno","Molto","Poco","Alcuni","Ogni","Tutto","Tutta","Tutti","Tutte","Stesso","Suo","Sua","Loro","Nostro","Vostro","Mio","Era","Erano","Ha","Hanno","Sono","Sarà","Sarebbe","Può","Deve","Uno","Due","Tre","Cosa","Cose","Questo","Quello","Quella","Questi","Quelli","Qui","Là","Ora","Sempre","Mai","Nulla","Niente","Quindi","Allora","Però","Inoltre"},
    "pl": {"I","A","Ale","Lecz","Bo","Choć","Choćby","Jeśli","Jeżeli","Gdy","Gdyby","Kiedy","Kiedykolwiek","Który","Która","Które","Którzy","Co","Kto","Kogo","Komu","Czy","Czyj","Jak","Jako","Tak","Też","Także","Również","Tylko","Jeszcze","Już","Bardzo","Więcej","Mniej","Dużo","Mało","Każdy","Cały","Wszyscy","Wszystkie","Sam","Jego","Jej","Ich","Nasz","Wasz","Mój","Twój","Był","Była","Było","Były","Jest","Są","Będzie","Mieć","Może","Musi","Trzeba","Tu","Tam","Teraz","Wtedy","Zawsze","Nigdy","Nic","Coś","Ten","Ta","To","Ci","Te","Tamten","Tamta","Tamto","Niż","Bez","Pod","Nad","Przed","Za","Do","Od","Po","Na","W","Z","O","Ku"},
    "fr": {"Le","La","Les","L","Un","Une","Des","Et","Ou","Mais","Donc","Or","Ni","Car","Pour","Sans","Avec","Sur","Sous","Dans","Par","De","Du","Des","Au","Aux","Que","Qui","Quoi","Dont","Où","Quand","Comment","Pourquoi","Si","Aussi","Encore","Déjà","Plus","Moins","Beaucoup","Peu","Tout","Tous","Toute","Toutes","Chaque","Quelques","Plusieurs","Même","Autre","Autres","Premier","Dernier","Son","Sa","Ses","Leur","Leurs","Notre","Votre","Mon","Ma","Mes","Était","Étaient","Est","Sont","Sera","Serait","A","Ont","Avait","Avaient","Peut","Doit","Faut","Va","Vont","Cela","Ceci","Cette","Cet","Ces","Ici","Là","Maintenant","Toujours","Jamais","Rien","Une","Deux","Trois"},
}

def get_lang_from_jsonl(rows):
    # rough fallback — count common stopcaps
    for lang, stops in STOPCAPS.items():
        hits = 0
        for r in rows[:2000]:
            words = r["text"].split()[:8]
            for w in words:
                if w.strip(",.;:!?()") in stops:
                    hits += 1
                    break
        if hits > 50:
            return lang
    return "en"

def is_sentence_start(text, pos):
    # heuristic: at character 0 in text, OR preceded by sentence-end punctuation+space
    if pos == 0:
        return True
    snippet = text[max(0, pos-3):pos]
    return bool(SENT_START_STOP.search(snippet))

def extract_tokens(rows, lang, *, want_proper=True):
    """Return list of (sentence_idx, token_kind, token_value) tuples."""
    stops = STOPCAPS.get(lang, STOPCAPS["en"])
    out = []
    for i, r in enumerate(rows):
        t = r["text"]
        # years
        for m in YEAR.finditer(t):
            out.append((i, "year", m.group(1)))
        # bignums (not years — already captured)
        for m in BIGNUM.finditer(t):
            v = m.group(0)
            if not YEAR.fullmatch(v):
                out.append((i, "num", v))
        # proper nouns: cap word not in stops, not at sentence start (else ambiguous)
        if want_proper:
            for m in PROPER.finditer(t):
                v = m.group(1)
                if v in stops:
                    continue
                if is_sentence_start(t, m.start()):
                    # require length >= 5 to keep rare names like "Smith" even at start
                    if len(v) < 5: continue
                out.append((i, "proper", v))
    return out

def rare_intersection(tokens_o, tokens_t, *, max_freq=8):
    """For each token value, list orig-sent-indices and trans-sent-indices.
    Keep tokens that appear 1..max_freq times on BOTH sides.
    Return ordered list of candidate (orig_i, trans_j, token) anchors."""
    pos_o = defaultdict(list)
    pos_t = defaultdict(list)
    for i, k, v in tokens_o:
        pos_o[(k, v)].append(i)
    for i, k, v in tokens_t:
        pos_t[(k, v)].append(i)
    cands = []
    for key in pos_o.keys() & pos_t.keys():
        os_, ts_ = pos_o[key], pos_t[key]
        if not (1 <= len(os_) <= max_freq and 1 <= len(ts_) <= max_freq):
            continue
        # if count equal on both sides, pair by order
        if len(os_) == len(ts_):
            for oi, tj in zip(os_, ts_):
                cands.append((oi, tj, key))
        else:
            # for unequal counts, pair greedily by ratio position
            n = max(len(os_), len(ts_))
            for k in range(min(len(os_), len(ts_))):
                cands.append((os_[k], ts_[k], key))
    cands.sort()
    return cands

def lis_monotonic(cands):
    """Keep longest non-decreasing-by-trans subsequence."""
    if not cands: return []
    tails = []; tail_idx = []; pred = [-1]*len(cands)
    for k, (oi, tj, _) in enumerate(cands):
        pos = bisect.bisect_right(tails, tj)
        if pos == len(tails):
            tails.append(tj); tail_idx.append(k)
        else:
            tails[pos] = tj; tail_idx[pos] = k
        pred[k] = tail_idx[pos-1] if pos > 0 else -1
    chain = []
    k = tail_idx[-1]
    while k != -1:
        chain.append(k); k = pred[k]
    chain.reverse()
    return [cands[k] for k in chain]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-jsonl", required=True)
    ap.add_argument("--trans-jsonl", required=True)
    ap.add_argument("--orig-lang", default=None)
    ap.add_argument("--trans-lang", default=None)
    ap.add_argument("--max-freq", type=int, default=8,
                    help="A token must appear <= this many times on each side")
    ap.add_argument("-o","--out", required=True)
    args = ap.parse_args()
    rows_o = [json.loads(l) for l in Path(args.orig_jsonl).read_text().splitlines() if l.strip()]
    rows_t = [json.loads(l) for l in Path(args.trans_jsonl).read_text().splitlines() if l.strip()]
    lo = args.orig_lang or get_lang_from_jsonl(rows_o)
    lt = args.trans_lang or get_lang_from_jsonl(rows_t)
    print(f"  langs detected: orig={lo} trans={lt}", file=sys.stderr)
    tok_o = extract_tokens(rows_o, lo)
    tok_t = extract_tokens(rows_t, lt)
    print(f"  tokens: orig={len(tok_o)} trans={len(tok_t)}", file=sys.stderr)
    cands = rare_intersection(tok_o, tok_t, max_freq=args.max_freq)
    print(f"  rare intersect: {len(cands)} candidates", file=sys.stderr)
    mono = lis_monotonic(cands)
    print(f"  LIS-filtered:    {len(mono)} anchors", file=sys.stderr)
    # write as anchors.jsonl-compatible records
    out = [{"orig": oi, "trans": tj, "sim": 1.0, "token": f"{k[0]}:{k[1]}"} for oi,tj,k in mono]
    Path(args.out).write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in out))
    if mono:
        ratios = []
        for k in range(1, len(mono)):
            di = mono[k][0] - mono[k-1][0]
            dj = mono[k][1] - mono[k-1][1]
            if di > 0: ratios.append(dj/di)
        ratios.sort()
        if ratios:
            print(f"  local ratio: min={ratios[0]:.2f} med={ratios[len(ratios)//2]:.2f} max={ratios[-1]:.2f}", file=sys.stderr)

if __name__ == "__main__":
    main()
