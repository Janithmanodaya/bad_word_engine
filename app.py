# Improved bad-words detection service (FastAPI)
# - Faster substring search using Aho-Corasick (optional: pyahocorasick)
# - Optional fuzzy matching with rapidfuzz for obfuscated tokens
# - Better deobfuscation and spaced-letter handling
# - Graceful fallbacks if optional libraries are not installed
# - Maintains previous data-source behavior (LDNOOBW, MRVLS, SOLD, SemiSOLD, custom env lists)

import os
import re
import unicodedata
import logging
from typing import List, Set, Tuple, Optional, Dict
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter, Retry
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Optional accelerated libraries
try:
    import ahocorasick  # type: ignore
    HAS_AHO = True
except Exception:
    HAS_AHO = False

try:
    from rapidfuzz import fuzz, process  # type: ignore
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

try:
    from unidecode import unidecode  # type: ignore
    HAS_UNIDECODE = True
except Exception:
    HAS_UNIDECODE = False

# Logging setup

def _get_log_level() -> int:
    lvl = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, lvl, logging.INFO)


logging.basicConfig(level=_get_log_level(), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("badwords_improved")

# Globals
BAD_WORDS_EN: Set[str] = set()
BAD_WORDS_SI: Set[str] = set()
BAD_WORDS_SI_SINGLISH: Set[str] = set()

# Aho-corasick automatons (optional)
AC_AUTOMATON_EN = None
AC_AUTOMATON_SI = None
AC_AUTOMATON_SI_SING = None

# Fuzzy match cache (simple)

# Precompiled regexes
RE_LATIN = re.compile(r"[A-Za-z']+")
RE_SINHALA = re.compile(r"[\u0D80-\u0DFF]+")
RE_WORDSEP = re.compile(r"[\s\-\_\.,;:\|\\/]+")
RE_NONALNUM = re.compile(r"[^\w\u0D80-\u0DFF]", flags=re.UNICODE)


def _preview(text: str, n: int = 140) -> str:
    return text if len(text) <= n else text[:n] + "...(truncated)"


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    # remove control chars
    t = "".join(ch for ch in t if unicodedata.category(ch)[0] != "C")
    return t.lower()


# Deobfuscation: substitutions, collapse repeats, remove separators between letters
HOMOGLYPH_MAP: Dict[str, str] = {
    "@": "a",
    "$": "s",
    "€": "e",
    "£": "l",
    "1": "i",
    "!": "i",
    "3": "e",
    "4": "a",
    "0": "o",
    "7": "t",
    "5": "s",
    "8": "b",
    "9": "g",
    "|": "i",
    "¢": "c",
    "¥": "y",
}


def deobfuscate(text: str) -> str:
    """Produce several deobfuscated variants of input text.
    - map common leet/homoglyphs
    - collapse long character repeats
    - remove non-alnum separators between letters (eg: s.h.i.t -> shit)
    - transliterate if unidecode available
    """
    t = normalize_text(text)
    before = t
    # replace homoglyphs
    t = "".join(HOMOGLYPH_MAP.get(ch, ch) for ch in t)
    # collapse >2 repeats to two (a lot of spam repeats characters)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    # join letters that are separated by punctuation/spaces: s.h.i.t -> shit
    t = re.sub(r"(?:[\W_])+(?=[A-Za-z])", "", t)
    # transliterate (latin-friendly) to help fuzzy match
    if HAS_UNIDECODE:
        try:
            t_unid = unidecode(t)
            if t_unid and len(t_unid) < 4096:
                t = t_unid
        except Exception:
            pass
    # strip combining marks
    t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    logger.debug("Deobfuscate: before=%s after=%s", _preview(before), _preview(t))
    return t


def tokenize(text: str) -> Tuple[List[str], List[str]]:
    latin_tokens = RE_LATIN.findall(text)
    sinhala_tokens = RE_SINHALA.findall(text)
    logger.debug("Tokenize: latin=%d sinhala=%d", len(latin_tokens), len(sinhala_tokens))
    return latin_tokens, sinhala_tokens


def _requests_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    headers = {
        "User-Agent": "badwords-service/1.1 (+https://example.local)",
        "Accept": "text/plain,*/*;q=0.8",
    }
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session.headers.update(headers)
    return session


def fetch_text_lines(url: str, timeout: int = 20) -> List[str]:
    session = _requests_session()
    for candidate in _expand_mirrors(url):
        logger.info("Downloading list from %s", candidate)
        try:
            r = session.get(candidate, timeout=timeout)
            if r.status_code == 200 and r.text:
                lines = [line.strip() for line in r.text.splitlines() if line.strip()]
                logger.info("Fetched %d lines from %s", len(lines), candidate)
                return lines
            logger.warning("Non-200 response from %s: %s", candidate, r.status_code)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", candidate, e)
    return []


def _expand_mirrors(url: str) -> List[str]:
    urls = [url]
    m = re.match(r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.*)", url)
    if m:
        user, repo, branch, path = m.groups()
        urls.append(f"https://cdn.jsdelivr.net/gh/{user}/{repo}@{branch}/{path}")
        urls.append(f"https://raw.fastgit.org/{user}/{repo}/{branch}/{path}")
        urls.append(f"https://rawcdn.githack.com/{user}/{repo}/{branch}/{path}")
    return urls


# --- Loading lexicons (same logic as original but we build automata if available)

def load_en_bad_words() -> Set[str]:
    override = os.getenv("BAD_WORDS_EN_URL", "").strip()
    if override:
        logger.info("Loading English bad words from override URL: %s", override)
        words = {w.lower() for w in fetch_text_lines(override)}
        logger.info("Loaded %d English bad words (override)", len(words))
        return words

    logger.info("Loading English bad words (LDNOOBW)...")
    urls = [
        "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en.txt",
        "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/main/en.txt",
    ]
    words: Set[str] = set()
    for u in urls:
        for w in fetch_text_lines(u):
            words.add(w.lower())
        if words:
            break
    if not words:
        logger.warning("LDNOOBW fetch failed; English list is empty. Set BAD_WORDS_EN_URL or ensure GitHub raw access.")
    logger.info("Loaded %d English bad words", len(words))
    return words


# reused load_si_bad_words_mrmrvl() and SOLD loaders from previous script pattern
# For brevity in this example we'll call the same functions as before but keep them small

def load_si_bad_words_mrmrvl() -> Tuple[Set[str], Set[str]]:
    logger.info("Loading Sinhala MRVLS lists (unicode + singlish)...")
    override_si = os.getenv("BAD_WORDS_SI_URL", "").strip()
    override_si_sing = os.getenv("BAD_WORDS_SI_SINGLISH_URL", "").strip()
    si_unicode: Set[str] = set()
    si_singlish: Set[str] = set()

    if override_si:
        for w in fetch_text_lines(override_si):
            si_unicode.add(w)
    if override_si_sing:
        for w in fetch_text_lines(override_si_sing):
            si_singlish.add(w.lower())

    if not si_unicode:
        candidates_unicode = [
            "https://raw.githubusercontent.com/MrMRVLS/sinhala-bad-words-list/main/sinhala-bad-words-unicode.txt",
            "https://raw.githubusercontent.com/MrMRVLS/sinhala-bad-words-list/master/sinhala-bad-words-unicode.txt",
        ]
        for u in candidates_unicode:
            for w in fetch_text_lines(u):
                si_unicode.add(w)
            if si_unicode:
                break

    if not si_singlish:
        candidates_singlish = [
            "https://raw.githubusercontent.com/MrMRVLS/sinhala-bad-words-list/main/sinhala-bad-words-singlish.txt",
            "https://raw.githubusercontent.com/MrMRVLS/sinhala-bad-words-list/master/sinhala-bad-words-singlish.txt",
        ]
        for u in candidates_singlish:
            for w in fetch_text_lines(u):
                si_singlish.add(w.lower())
            if si_singlish:
                break

    logger.info("Loaded Sinhala MRVLS: unicode=%d singlish=%d", len(si_unicode), len(si_singlish))
    return si_unicode, si_singlish


# SOLD loader with fallback disabled if datasets isn't available

def load_si_bad_words_sold() -> Set[str]:
    if os.getenv("USE_SOLD", "1").lower() not in {"1", "true", "yes"}:
        logger.info("USE_SOLD disabled; skipping SOLD load.")
        return set()

    words: Set[str] = set()
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        logger.warning("datasets library not available: %s", e)
        return words

    try:
        logger.info("Loading SOLD dataset from HuggingFace...")
        sold = load_dataset("sinhala-nlp/SOLD")
    except Exception as e:
        logger.warning("Failed to load SOLD: %s", e)
        return words

    for split_name in sold.keys():
        split = sold[split_name]
        for item in split:
            # heuristic same as original: look for rationale tokens
            tokens = None
            for k in ("tokens", "token", "words", "word_tokens"):
                if item.get(k) is not None:
                    tokens = item.get(k)
                    break
            rationals = None
            for k in ("rationals", "rationales", "rationale", "rationale_labels"):
                if item.get(k) is not None:
                    rationals = item.get(k)
                    break
            if not tokens or rationals is None:
                continue
            if isinstance(tokens, str):
                toks = tokens.split()
            else:
                toks = tokens
            if isinstance(rationals, str):
                rats = [int(x) for x in rationals.split()]
            else:
                rats = [int(x) for x in rationals]
            for t, r in zip(toks, rats):
                if r == 1:
                    words.add(t)
    logger.info("Collected %d SOLD rationale tokens (%d unique)", len(words), len(set(words)))
    return words


def load_custom_words_from_env() -> Tuple[Set[str], Set[str], Set[str]]:
    en: Set[str] = set()
    si: Set[str] = set()
    si_sing: Set[str] = set()

    raw = os.getenv("BAD_WORDS_CUSTOM", "")
    if raw:
        for w in raw.split(","):
            w = w.strip()
            if not w:
                continue
            if re.search(r"[\u0D80-\u0DFF]", w):
                si.add(unicodedata.normalize("NFKC", w))
            else:
                en.add(w.lower())

    url = os.getenv("BAD_WORDS_URL", "")
    if url:
        for w in fetch_text_lines(url):
            if re.search(r"[\u0D80-\u0DFF]", w):
                si.add(unicodedata.normalize("NFKC", w))
            else:
                en.add(w.lower())

    return en, si, si_sing


def build_automaton(words: Set[str]) -> Optional[ahocorasick.Automaton]:
    if not HAS_AHO:
        return None
    A = ahocorasick.Automaton()
    for i, w in enumerate(sorted(words, key=len)):
        try:
            A.add_word(w, (i, w))
        except Exception:
            pass
    A.make_automaton()
    return A


def _load_semisold_tokens(threshold: float = 0.8, top_tokens: int = 1000) -> Set[str]:
    if os.getenv("USE_SEMISOLD", "0").lower() not in {"1", "true", "yes"}:
        logger.info("USE_SEMISOLD disabled; skipping SemiSOLD mining.")
        return set()
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        logger.warning("datasets library not available for SemiSOLD: %s", e)
        return set()
    try:
        ds = load_dataset("sinhala-nlp/SemiSOLD", split="train")
    except Exception as e:
        logger.warning("Failed to load SemiSOLD: %s", e)
        return set()
    score_cols = []
    sample = ds[0] if len(ds) else {}
    for k, v in sample.items():
        if isinstance(v, (int, float)):
            score_cols.append(k)
    if not score_cols:
        score_cols = ["xlmr", "xlmt", "mbert", "sinbert"]
    freq = {}
    kept_rows = 0
    for row in ds:
        scores = [float(row.get(c, 0.0)) for c in score_cols if isinstance(row.get(c, None), (int, float))]
        if not scores:
            continue
        mean_score = sum(scores) / len(scores)
        if mean_score < threshold:
            continue
        kept_rows += 1
        text = normalize_text(str(row.get("text", "")))
        _, si_tokens = tokenize(text)
        for tok in si_tokens:
            tok_n = unicodedata.normalize("NFKC", tok)
            freq[tok_n] = freq.get(tok_n, 0) + 1
    logger.info("SemiSOLD rows kept after threshold %.2f: %d", threshold, kept_rows)
    sorted_toks = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_tokens]
    return {t for t, _ in sorted_toks}


def init_lexicons() -> None:
    global BAD_WORDS_EN, BAD_WORDS_SI, BAD_WORDS_SI_SINGLISH, AC_AUTOMATON_EN, AC_AUTOMATON_SI, AC_AUTOMATON_SI_SING
    logger.info("Initializing lexicons (improved)...")
    BAD_WORDS_EN = load_en_bad_words()
    si_unicode_mrmrvl, si_singlish_mrmrvl = load_si_bad_words_mrmrvl()
    si_sold = load_si_bad_words_sold()
    thr = float(os.getenv("SEMISOLD_THRESHOLD", "0.8"))
    topn = int(os.getenv("SEMISOLD_TOP_TOKENS", "1000"))
    si_semisold = _load_semisold_tokens(threshold=thr, top_tokens=topn)

    BAD_WORDS_SI = set()
    for w in si_unicode_mrmrvl.union(si_sold).union(si_semisold):
        BAD_WORDS_SI.add(unicodedata.normalize("NFKC", w))

    BAD_WORDS_SI_SINGLISH = set()
    for w in si_singlish_mrmrvl:
        BAD_WORDS_SI_SINGLISH.add(w.lower())

    en_c, si_c, si_sing_c = load_custom_words_from_env()
    BAD_WORDS_EN.update(en_c)
    BAD_WORDS_SI.update(si_c)
    BAD_WORDS_SI_SINGLISH.update(si_sing_c)

    # build automatons if available
    if HAS_AHO:
        try:
            AC_AUTOMATON_EN = build_automaton(BAD_WORDS_EN)
            AC_AUTOMATON_SI = build_automaton(BAD_WORDS_SI)
            AC_AUTOMATON_SI_SING = build_automaton(BAD_WORDS_SI_SINGLISH)
            logger.info("Aho-Corasick automatons built (enabled)")
        except Exception as e:
            logger.warning("Failed to build Aho automatons: %s", e)
    else:
        logger.info("pyahocorasick not available: substring scanning will be slower")

    logger.info(
        "Lexicons ready: EN=%d SI=%d SI_singlish=%d (USE_SOLD=%s USE_SEMISOLD=%s) HAS_AHO=%s HAS_RAPIDFUZZ=%s",
        len(BAD_WORDS_EN),
        len(BAD_WORDS_SI),
        len(BAD_WORDS_SI_SINGLISH),
        os.getenv("USE_SOLD", "1"),
        os.getenv("USE_SEMISOLD", "0"),
        HAS_AHO,
        HAS_RAPIDFUZZ,
    )


# Matching helpers

def _automaton_find(automaton, text: str) -> Set[str]:
    hits = set()
    if automaton is None:
        return hits
    try:
        for end_index, (idx, word) in automaton.iter(text):
            hits.add(word)
    except Exception:
        # fallback safe iteration
        return set()
    return hits


def _match_substrings_simple(text: str, candidates: Set[str], min_len: int = 3) -> Set[str]:
    hits: Set[str] = set()
    if not candidates:
        return hits
    for w in candidates:
        if len(w) >= min_len and w in text:
            hits.add(w)
    return hits


def _fuzzy_hits(token: str, candidates: Set[str], limit: int = 5, threshold: int = 80) -> Set[str]:
    """Return candidate words that fuzzily match token above threshold.
    Uses rapidfuzz if available, else empty set.
    """
    if not HAS_RAPIDFUZZ:
        return set()
    try:
        # use process.extract to get best matches
        results = process.extract(token, list(candidates), scorer=fuzz.partial_ratio, limit=limit)
        return {r[0] for r, score, _ in results if score >= threshold}
    except Exception:
        # last-resort naive matching
        out = set()
        for w in candidates:
            if token in w or w in token:
                out.add(w)
        return out


def find_bad_words(text: str, fuzzy_threshold: int = 85) -> Set[str]:
    """Find bad words in text. Returns a set of matched lexicon strings.
    Strategy:
    - create normalized & deobfuscated variants
    - use Aho-Corasick if available for fast substrings
    - token-level exact matches for latin/sinhala tokens
    - fallback substring scanning
    - optional fuzzy matching for short/obfuscated tokens using rapidfuzz
    """
    hits: Set[str] = set()
    variants = [normalize_text(text), deobfuscate(text), RE_NONALNUM.sub("", normalize_text(text))]

    # also consider a collapsed-spaces variant: remove separators between letters
    variants.append(RE_WORDSEP.sub("", normalize_text(text)))

    tried = set()
    for v in variants:
        if not v or v in tried:
            continue
        tried.add(v)
        logger.debug("Variant preview=%s", _preview(v))

        # Aho-corasick fast substring search
        if HAS_AHO and AC_AUTOMATON_SI is not None:
            hits.update(_automaton_find(AC_AUTOMATON_SI, v))
        if HAS_AHO and AC_AUTOMATON_EN is not None:
            hits.update(_automaton_find(AC_AUTOMATON_EN, v))
        if HAS_AHO and AC_AUTOMATON_SI_SING is not None:
            hits.update(_automaton_find(AC_AUTOMATON_SI_SING, v))

        # token-level checks (exact)
        latin_tokens, sinhala_tokens = tokenize(v)
        for tok in latin_tokens:
            if tok in BAD_WORDS_EN or tok in BAD_WORDS_SI_SINGLISH:
                hits.add(tok)
        for tok in sinhala_tokens:
            if tok in BAD_WORDS_SI:
                hits.add(tok)

        # substring fallback (slower) for missing automaton
        if not HAS_AHO:
            hits.update(_match_substrings_simple(v, BAD_WORDS_SI, min_len=2))
            hits.update(_match_substrings_simple(v, BAD_WORDS_EN, min_len=3))
            hits.update(_match_substrings_simple(v, BAD_WORDS_SI_SINGLISH, min_len=3))

        # fuzzy matching for short tokens or extremely obfuscated tokens
        if HAS_RAPIDFUZZ:
            # check latin tokens fuzzily
            for tok in latin_tokens:
                if 2 <= len(tok) <= 40:
                    f = _fuzzy_hits(tok, BAD_WORDS_EN.union(BAD_WORDS_SI_SINGLISH), limit=5, threshold=fuzzy_threshold)
                    hits.update(f)
            # check whole string fuzzy against english list if short
            s = v.strip()
            if 3 <= len(s) <= 40:
                f2 = _fuzzy_hits(s, BAD_WORDS_EN.union(BAD_WORDS_SI_SINGLISH), limit=5, threshold=max(60, fuzzy_threshold - 15))
                hits.update(f2)

    if hits:
        logger.debug("Final hits: %s", sorted(hits)[:100])
    return hits


# --- FastAPI endpoints ---

class CheckRequest(BaseModel):
    text: str
    advanced: bool = False


class DefaultResponse(BaseModel):
    found: bool


class AdvancedResponse(BaseModel):
    found: bool
    bad_words: List[str]


app = FastAPI()


@app.on_event("startup")
def on_startup():
    init_lexicons()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "en_words": len(BAD_WORDS_EN),
        "si_words": len(BAD_WORDS_SI),
        "si_singlish_words": len(BAD_WORDS_SI_SINGLISH),
        "has_ahocorasick": HAS_AHO,
        "has_rapidfuzz": HAS_RAPIDFUZZ,
    }


def check_api_key(req: Request, expected_key: str) -> None:
    key = req.headers.get("X-API-Key", "")
    if expected_key and key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/check")
async def check_default(req: Request, payload: CheckRequest):
    api_key = os.getenv("API_KEY", "")
    check_api_key(req, api_key)
    logger.info("Incoming /check: advanced=%s text_preview=%s", payload.advanced, _preview(payload.text))
    hits = find_bad_words(payload.text)
    logger.info("Outgoing /check: found=%s hits=%d", bool(hits), len(hits))
    if payload.advanced:
        return AdvancedResponse(found=bool(hits), bad_words=sorted(hits, key=lambda s: (len(s), s)))
    return DefaultResponse(found=bool(hits))


# Helpful __main__ for quick local testing
if __name__ == "__main__":
    # quick sanity checks
    examples = [
        "you are a s.h.i.t!",
        "ඔබ ම*ල*කේ",
        "obfu$cat3d discussion",
        "mama api *******",
    ]
    init_lexicons()
    for ex in examples:
        print("> ", ex)
        print("  hits:", find_bad_words(ex))

    # run uvicorn when invoked directly
    import uvicorn

    port_env = os.getenv("PORT")
    try:
        port = int(port_env) if port_env and port_env.isdigit() else 8000
    except Exception:
        port = 8000
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("app_improved:app", host=host, port=port, reload=False)
