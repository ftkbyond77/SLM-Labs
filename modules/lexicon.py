"""The language layer: gloss id <-> English label <-> Thai lemma, and coverage.

The problem this solves
-----------------------
v1.2 could not tell the two failure modes apart. When the reader rejects a sign
in a real video, either

    (a) the sign is genuinely outside the 184-gloss vocabulary, or
    (b) it is inside it and the model failed under coarticulation,

and no amount of extra modelling separates those two. Only a **lemma table**
does: with one, you take the Thai sentence the video is known to contain, tokenise
it, and check word by word whether the vocabulary could ever have contained it.
Without one, gloss `0272` is an opaque integer and every diagnosis is a guess.

The released `.npy` bundle carries numeric ids only. So this module does three
things:

1. **Holds** the table (`data/lexicon.json`) and keeps every entry tagged with
   where it came from and how much to trust it - `verified`, `llm`, `heuristic`.
2. **Normalises** the English-style labels that sign-language corpora ship, which
   are frequently *spelled-out letter names* rather than words: `"Gor Gai"` is
   not a word, it is the Thai letter **ก**. A built-in table covers all 44 Thai
   consonants, the vowels, and the digits deterministically; anything else is
   handed to an LLM with a strict schema. The LLM never sees skeletons and never
   decides what a sign *is* - it only converts one written form to another.
3. **Answers the coverage question**: given a Thai sentence, which of its words
   could the 184-gloss vocabulary express at all?

Nothing here changes the recogniser. It changes what the recogniser's output
*means*, and it is what makes the `data_test/` filenames usable as ground truth.
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .config import ARTIFACT_DIR, load_openai_key

# Lives with the artifacts rather than under data/: it is produced by this
# pipeline (and hand-edited afterwards), not shipped with the corpus.
LEXICON_PATH = ARTIFACT_DIR / "lexicon.json"
UNKNOWN = "_"


# ------------------------------------------------- Thai spelling alphabet ---
# name -> letter. These are the acrophonic letter names Thai children learn
# ("ko kai" = the ko of kai/chicken). A sign-language corpus that fingerspells
# will label the sign with the *name*, so "Gor Gai" must resolve to "ก".
THAI_CONSONANTS = {
    "ก": ["ko kai", "gor gai", "kor kai", "gaw gai", "ko kai chicken"],
    "ข": ["kho khai", "kor khai", "khor khai", "kho kai egg"],
    "ฃ": ["kho khuat", "khor khuad", "kho khuad"],
    "ค": ["kho khwai", "kor kwai", "khor kwai", "kho kwai buffalo"],
    "ฅ": ["kho khon", "khor khon"],
    "ฆ": ["kho ra khang", "khor rakhang", "kho rakhang"],
    "ง": ["ngo ngu", "ngor ngoo", "ngo ngoo", "ngor ngu snake"],
    "จ": ["cho chan", "jor jan", "cho jan", "jor jaan plate"],
    "ฉ": ["cho ching", "chor ching"],
    "ช": ["cho chang", "chor chang", "cho chang elephant"],
    "ซ": ["so so", "sor so", "so soh chain"],
    "ฌ": ["cho choe", "chor cher", "cho cher"],
    "ญ": ["yo ying", "yor ying", "yo ying woman"],
    "ฎ": ["do chada", "dor chada", "do cha da"],
    "ฏ": ["to patak", "tor patak"],
    "ฐ": ["tho than", "thor than"],
    "ฑ": ["tho montho", "thor montho"],
    "ฒ": ["tho phu thao", "thor phuthao"],
    "ณ": ["no nen", "nor nen"],
    "ด": ["do dek", "dor dek", "do dek child"],
    "ต": ["to tao", "tor tao", "to tao turtle"],
    "ถ": ["tho thung", "thor thung", "tho tung bag"],
    "ท": ["tho thahan", "thor tahan", "tho tahan soldier"],
    "ธ": ["tho thong", "thor tong", "tho tong flag"],
    "น": ["no nu", "nor noo", "no noo mouse"],
    "บ": ["bo baimai", "bor baimai", "bo bai mai leaf"],
    "ป": ["po pla", "por pla", "po plaa fish"],
    "ผ": ["pho phueng", "por pueng", "pho pung bee"],
    "ฝ": ["fo fa", "for fa", "fo faa lid"],
    "พ": ["pho phan", "por pan", "pho pan tray"],
    "ฟ": ["fo fan", "for fan", "fo fun tooth"],
    "ภ": ["pho samphao", "por sampao", "pho sampao"],
    "ม": ["mo ma", "mor ma", "mo maa horse"],
    "ย": ["yo yak", "yor yak", "yo yuk giant"],
    "ร": ["ro ruea", "ror rua", "ro rua boat"],
    "ล": ["lo ling", "lor ling", "lo ling monkey"],
    "ว": ["wo waen", "wor waen", "wo waen ring"],
    "ศ": ["so sala", "sor sala"],
    "ษ": ["so ruesi", "sor rusi", "so rusi"],
    "ส": ["so suea", "sor sua", "so sua tiger"],
    "ห": ["ho hip", "hor heep", "ho heep box"],
    "ฬ": ["lo chula", "lor jula", "lo jula kite"],
    "อ": ["o ang", "or ang", "o aang basin"],
    "ฮ": ["ho nokhuk", "hor nok huk", "ho nok hook owl"],
}

# A number sign means the *word*, not the glyph: speech has to say "ห้า", not
# read out a numeral. The glyph and the Arabic digit are kept as aliases so
# either spelling of an incoming label still resolves.
THAI_DIGITS = {
    "ศูนย์": ["zero", "0", "๐", "soon", "sun"],
    "หนึ่ง": ["one", "1", "๑", "nueng", "neung"],
    "สอง": ["two", "2", "๒", "song", "saung"],
    "สาม": ["three", "3", "๓", "sam", "saam"],
    "สี่": ["four", "4", "๔", "si", "see"],
    "ห้า": ["five", "5", "๕", "ha", "haa"],
    "หก": ["six", "6", "๖", "hok"],
    "เจ็ด": ["seven", "7", "๗", "chet", "jet"],
    "แปด": ["eight", "8", "๘", "paet", "pad"],
    "เก้า": ["nine", "9", "๙", "kao", "gao"],
    "สิบ": ["ten", "10", "sip"],
    "ยี่สิบ": ["twenty", "20", "yi sip"],
}


def _norm(s: str) -> str:
    """Fold a romanisation to a comparable key.

    Only *purely orthographic* variants are collapsed: g/k and j/ch (RTGS vs.
    informal spelling), doubled vowels, and the trailing -r Thai speakers write
    for a long vowel ("kor" = "ko").

    Aspiration is deliberately NOT folded. `kh`, `ph`, `th` are the only thing
    separating ก from ข, ป from ผ and ต from ถ; an earlier version of this
    function collapsed them and silently resolved "Gor Gai" to ข.
    """
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9 ]+", " ", s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\b([a-z]+?)r\b", r"\1", s)      # "kor" -> "ko"
    s = re.sub(r"(?<!n)g", "k", s)               # g -> k, but never break "ng"
    s = s.replace("j", "ch")
    s = re.sub(r"([aeiou])\1+", r"\1", s)        # "ngoo" -> "ngo"
    return s.strip()


_SPELL_INDEX = None


def _spell_index() -> dict:
    global _SPELL_INDEX
    if _SPELL_INDEX is None:
        idx = {}
        for ch, names in list(THAI_CONSONANTS.items()) + list(THAI_DIGITS.items()):
            for n in [ch] + list(names):
                k = _norm(n)
                if k:                    # Thai glyphs fold to "" - never index those
                    idx.setdefault(k, ch)
        _SPELL_INDEX = idx
    return _SPELL_INDEX


def resolve_spelling(label: str, cutoff: float = 0.86) -> tuple:
    """`"Gor Gai"` -> `("ก", "exact-alias")`, with fuzzy fallback.

    Returns `(thai_char_or_None, how)`. This is deterministic and offline: no
    LLM is consulted for something a lookup table settles, which keeps the
    fingerspelling part of the vocabulary verifiable rather than plausible.
    """
    raw = str(label).strip()
    if raw in THAI_CONSONANTS:                 # already a Thai letter
        return raw, "already-thai"
    for ch, al in THAI_DIGITS.items():         # already a Thai numeral or word
        if raw == ch or raw in al:
            return ch, "already-thai"
    idx = _spell_index()
    k = _norm(label)
    if not k:
        return None, "empty"
    if k in idx:
        return idx[k], "exact-alias"
    m = difflib.get_close_matches(k, list(idx), n=1, cutoff=cutoff)
    if m:
        return idx[m[0]], f"fuzzy({m[0]})"
    return None, "no-match"


# --------------------------------------------------------------- lexicon ---
@dataclass
class Entry:
    gloss_id: str
    thai: str = ""
    english: str = ""
    category: str = ""
    source: str = "unfilled"        # verified | llm | spelling-table | heuristic | unfilled
    confidence: float = 0.0
    aliases: list = field(default_factory=list)

    @property
    def filled(self) -> bool:
        return bool(self.thai)


class Lexicon:
    """gloss id -> Thai lemma, plus the reverse lookup coverage needs.

    Every entry keeps its provenance. A report that mixes a hand-verified lemma
    with an LLM guess and prints both as "the Thai word" is worse than no
    lexicon at all, so `source` is carried everywhere and `verified_only`
    filters to what a person actually signed off.
    """

    def __init__(self, entries: dict | None = None, path: Path | None = None):
        self.path = Path(path or LEXICON_PATH)
        self.entries: dict = entries or {}

    # ---- io ------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | None = None) -> "Lexicon":
        p = Path(path or LEXICON_PATH)
        if not p.exists():
            return cls(path=p)
        raw = json.loads(p.read_text(encoding="utf-8"))
        ent = {k: Entry(gloss_id=k, **v) for k, v in raw.get("entries", {}).items()}
        return cls(ent, p)

    def save(self, path: Path | None = None) -> Path:
        p = Path(path or self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        body = {"schema": 1,
                "note": ("gloss id -> Thai lemma for TSL-ONE-S. `source` records "
                         "provenance: verified = checked by a person; "
                         "spelling-table = resolved offline from the Thai letter "
                         "names; llm = machine-proposed, unverified."),
                "entries": {k: {f: getattr(v, f) for f in
                                ("thai", "english", "category", "source",
                                 "confidence", "aliases")}
                            for k, v in sorted(self.entries.items())}}
        p.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def blank(cls, gloss_ids: list, categories: dict | None = None) -> "Lexicon":
        """An empty table with one row per gloss - the thing a person fills in."""
        return cls({g: Entry(gloss_id=g, category=(categories or {}).get(g, ""))
                    for g in gloss_ids})

    # ---- access --------------------------------------------------------
    def __len__(self):
        return len(self.entries)

    def __contains__(self, gid):
        return gid in self.entries and self.entries[gid].filled

    def thai(self, gid: str, default: str | None = None) -> str:
        e = self.entries.get(gid)
        if e and e.thai:
            return e.thai
        return default if default is not None else f"GLOSS_{gid}"

    def as_dict(self, verified_only: bool = False) -> dict:
        return {k: v.thai for k, v in self.entries.items()
                if v.filled and (not verified_only or v.source == "verified")}

    @property
    def n_filled(self) -> int:
        return sum(1 for v in self.entries.values() if v.filled)

    def coverage_by_source(self) -> dict:
        out = {}
        for v in self.entries.values():
            out[v.source] = out.get(v.source, 0) + 1
        return out

    # ---- filling -------------------------------------------------------
    def apply_english(self, labels: dict, use_spelling_table: bool = True) -> dict:
        """Ingest an English label table `{gloss_id: "Gor Gai"}`.

        Fingerspelling names are resolved offline; everything else is left for
        `fill_with_llm`. Returns a small report so the split is visible.
        """
        rep = {"spelling": 0, "pending": 0, "unknown_id": 0}
        for gid, lab in labels.items():
            gid = str(gid)
            if gid not in self.entries:
                self.entries[gid] = Entry(gloss_id=gid)
                rep["unknown_id"] += 1
            e = self.entries[gid]
            e.english = str(lab)
            ch, how = resolve_spelling(lab) if use_spelling_table else (None, "off")
            if ch:
                e.thai, e.source, e.confidence = ch, "spelling-table", 1.0
                e.aliases = [str(lab)]
                rep["spelling"] += 1
            else:
                rep["pending"] += 1
        return rep

    def fill_with_llm(self, model: str = "gpt-4o-mini", batch: int = 40,
                      only_missing: bool = True, verbose: bool = True) -> dict:
        """English label -> Thai lemma, for the entries a table cannot settle.

        Strictly a *translation of a written label*. The model is told the labels
        come from a Thai Sign Language corpus, that some are letter names, and
        that it must answer with a JSON object and nothing else. Entries it
        cannot do are left empty rather than guessed - an unfilled row is honest,
        a wrong lemma silently corrupts every coverage number downstream.
        """
        todo = [e for e in self.entries.values()
                if e.english and (not only_missing or not e.thai)]
        if not todo:
            return {"filled": 0, "skipped": 0, "reason": "nothing to do"}
        key = load_openai_key()
        if not key:
            return {"filled": 0, "skipped": len(todo), "reason": "no OPENAI_API_KEY"}
        from openai import OpenAI
        client = OpenAI(api_key=key)

        sys_msg = (
            "You convert labels from a Thai Sign Language word-level corpus into "
            "Thai lemmas. Input is a JSON object {gloss_id: english_label}. "
            "Rules: (1) a label that is a Thai letter name such as 'Gor Gai', "
            "'Kor Khai', 'Ngor Ngoo' is a FINGERSPELLED LETTER - return the Thai "
            "character itself (ก, ข, ง). (2) a label that is an ordinary English "
            "word or phrase is a sign meaning that word - return the everyday Thai "
            "word a signer would use, one lemma, no particles, no explanation. "
            "(3) a numeral label returns the Thai numeral word. (4) if you are not "
            "confident, return an empty string for that id - do NOT guess. "
            "Answer with a JSON object {gloss_id: thai} and nothing else.")

        filled, out = 0, {}
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            payload = {e.gloss_id: e.english for e in chunk}
            try:
                r = client.chat.completions.create(
                    model=model, temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": sys_msg},
                              {"role": "user", "content": json.dumps(payload,
                                                                     ensure_ascii=False)}])
                got = json.loads(r.choices[0].message.content)
            except Exception as ex:
                if verbose:
                    print(f"[lexicon] LLM batch {i//batch} failed: "
                          f"{type(ex).__name__}: {ex}")
                continue
            for gid, th in got.items():
                e = self.entries.get(str(gid))
                if e is not None and isinstance(th, str) and th.strip():
                    e.thai, e.source, e.confidence = th.strip(), "llm", 0.6
                    filled += 1
                    out[gid] = th.strip()
        return {"filled": filled, "requested": len(todo), "sample": dict(list(out.items())[:8])}


# ---------------------------------------------------------- coverage -------
def tokenize_thai(sentence: str) -> list:
    """Thai has no orthographic spaces, so word boundaries must be inferred."""
    try:
        from pythainlp.tokenize import word_tokenize
        toks = word_tokenize(str(sentence), engine="newmm", keep_whitespace=False)
    except Exception:
        toks = re.findall(r"[฀-๿]+|[A-Za-z0-9]+", str(sentence))
    return [t for t in (x.strip() for x in toks) if t]


CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM", "PRON", "AUX"}

# Sentence-final and question particles. Thai Sign Language marks these
# non-manually (a brow raise, a held gaze) or not at all, so a signer produces no
# hand sign for them and counting them against the reader would be unfair.
# AUX is *kept* in CONTENT_POS because the tagger labels ordinary main verbs such
# as "ไป" (go) as AUX, and those are very much signed.
THAI_PARTICLES = {"มั้ย", "ไหม", "มั๊ย", "ครับ", "ค่ะ", "คะ", "นะ", "น่ะ", "จ้ะ",
                  "จ๋า", "ล่ะ", "เหรอ", "หรอ", "หรือเปล่า", "เลย", "ๆ", "ด้วย"}


def content_tokens(sentence: str) -> list:
    """Tokens a signer would actually produce a hand sign for.

    This makes the `data_test/` filenames usable as a *weak* measurement even
    with no lemma table: the expected number of signs is roughly the number of
    content words, and that is directly comparable with the number the reader
    localised. It is an approximation — Thai tokenisation and POS tagging are
    themselves imperfect, and TSL does not map one-to-one onto Thai words — so it
    bounds the sign count rather than fixing it.
    """
    toks = tokenize_thai(sentence)
    try:
        from pythainlp.tag import pos_tag
        tagged = pos_tag(toks, corpus="orchid_ud")
        keep = [w for w, t in tagged if t in CONTENT_POS]
    except Exception:
        keep = toks
    return [w for w in keep if w not in THAI_PARTICLES]


@dataclass
class Coverage:
    sentence: str
    tokens: list
    in_vocab: list
    out_of_vocab: list
    matched: dict                    # token -> gloss id
    lexicon_filled: int
    lexicon_total: int

    @property
    def rate(self) -> float:
        return len(self.in_vocab) / max(len(self.tokens), 1)

    def summary(self) -> str:
        if self.lexicon_filled == 0:
            return ("lexicon empty - coverage is UNDECIDABLE: a rejected sign "
                    "cannot be attributed to vocabulary or to the model")
        return (f"{len(self.in_vocab)}/{len(self.tokens)} tokens are in the "
                f"{self.lexicon_filled}-lemma vocabulary "
                f"({self.rate:.0%}); out: {', '.join(self.out_of_vocab) or '-'}")


def coverage(sentence: str, lex: Lexicon, extra_forms: dict | None = None) -> Coverage:
    """Which words of a Thai sentence could this vocabulary express at all?

    This is the measurement that resolves the v1.2 ambiguity. Every token that is
    *not* in the vocabulary is a sign the reader was never able to produce, so
    rejecting it is correct behaviour and must not be counted as a model error.
    Tokens that *are* in the vocabulary and still came back `_` are the model's
    responsibility, and only those.
    """
    toks = tokenize_thai(sentence)
    table = lex.as_dict()
    rev = {}
    for gid, th in table.items():
        rev.setdefault(th, gid)
        for a in lex.entries[gid].aliases:
            rev.setdefault(a, gid)
    for th, gid in (extra_forms or {}).items():
        rev.setdefault(th, gid)

    ins, outs, matched = [], [], {}
    for t in toks:
        if t in rev:
            ins.append(t); matched[t] = rev[t]
        else:
            outs.append(t)
    return Coverage(sentence=str(sentence), tokens=toks, in_vocab=ins,
                    out_of_vocab=outs, matched=matched,
                    lexicon_filled=lex.n_filled, lexicon_total=len(lex))


# ------------------------------------------------ gloss identity cards -----
def gloss_cards(clips_by_gloss: dict, out_path: Path, per_page: int = 24,
                n_frames: int = 5, lex: Lexicon | None = None) -> list:
    """Render a contact sheet of every gloss so a person can label the vocabulary.

    This is deliberately low-tech and it is the highest-leverage thing in this
    module: 184 rows of five skeleton poses is roughly an hour of work for
    someone who signs, and that hour converts the single blocking unknown in this
    project (what do the ids *mean*) into a solved problem. Everything else here
    is ready for the moment that file exists.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from .viz import draw_skeleton

    # MediaPipe extrapolates knees and ankles for an upper-body recording. They
    # carry no signal (config.py drops them from the features too) and they drag
    # the crop box down, so they are removed before drawing.
    LEGS = np.arange(25, 33)

    def _trim(clip):
        c = clip.copy()
        c[:, LEGS] = 0.0
        return c

    def _bbox(clip, pad=0.06):
        """Crop to where the signer actually is - a handshape drawn at full-frame
        scale is a smudge."""
        pts = clip.reshape(-1, 2)
        pts = pts[~(pts == 0).all(1)]
        if not len(pts):
            return None
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        side = max(x1 - x0, y1 - y0) + 2 * pad
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return (cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gids = sorted(clips_by_gloss)
    pages = []
    for pno, start in enumerate(range(0, len(gids), per_page)):
        page = gids[start:start + per_page]
        fig, axes = plt.subplots(len(page), n_frames,
                                 figsize=(1.45 * n_frames, 1.5 * len(page)),
                                 squeeze=False)
        for r, g in enumerate(page):
            c = _trim(clips_by_gloss[g])
            bb = _bbox(c)
            idx = np.linspace(0, len(c) - 1, n_frames).astype(int)
            for j, fi in enumerate(idx):
                draw_skeleton(axes[r][j], c[fi], "", show_missing=False, bbox=bb)
            th = lex.thai(g, "") if lex else ""
            axes[r][0].set_ylabel(f"{g}\n{th}", rotation=0, ha="right", va="center",
                                  fontsize=9)
        fig.suptitle(f"TSL-ONE-S gloss identity cards - page {pno + 1}", fontsize=11)
        fig.tight_layout()
        p = out_path.with_name(f"{out_path.stem}_p{pno + 1}{out_path.suffix}")
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        pages.append(p)
    return pages


# ------------------------------------------------------ sentence layer -----
def compose_sentence(glosses: list, lex: Lexicon, model: str = "gpt-4o-mini",
                     context: str | None = None) -> dict:
    """Gloss sequence -> fluent Thai. The LLM is the *language* layer only.

    It never sees a skeleton and never decides what a sign was: it receives an
    ordered list of Thai lemmas plus `_` for the signs the reader declined to
    name, and its whole job is word order, particles and inflection - the parts
    signing does not encode. `_` must survive into the output; a language model
    filling those gaps would be inventing content the camera never saw.
    """
    surface = [g if g == UNKNOWN else lex.thai(g) for g in glosses]
    naive = " ".join(surface)
    lexical = [s for s in surface if not s.startswith("GLOSS_") and s != UNKNOWN]
    if not lexical:
        return {"sentence": naive, "glosses": surface,
                "source": "no-lexicon (LLM skipped: gloss ids carry no lexical content)"}
    key = load_openai_key()
    if not key:
        return {"sentence": naive, "glosses": surface, "source": "fallback-join"}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        sys_msg = (
            "You rewrite Thai Sign Language gloss sequences as one natural Thai "
            "sentence. The glosses are in signing order, which differs from Thai "
            "word order, and they omit particles, tense and classifiers - add "
            "them. '_' marks a sign the recogniser declined to name: keep '_' in "
            "place, never invent a word for it. A token like GLOSS_0272 is an "
            "unmapped id: keep it verbatim. Reply with the Thai sentence only.")
        msgs = [{"role": "system", "content": sys_msg}]
        if context:
            msgs.append({"role": "system", "content": f"Domain context: {context}"})
        msgs.append({"role": "user", "content": " ".join(surface)})
        r = client.chat.completions.create(model=model, temperature=0.2,
                                           max_tokens=200, messages=msgs)
        return {"sentence": r.choices[0].message.content.strip(),
                "source": model, "glosses": surface}
    except Exception as e:
        return {"sentence": naive, "glosses": surface,
                "source": f"fallback ({type(e).__name__})"}
