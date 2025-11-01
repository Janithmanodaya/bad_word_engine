# Improved bad-words detection service (FastAPI)
# Adds ML model (joblib) as the primary detector. If the model predicts "bad", we return immediately.
# If the model predicts "clean" or is unavailable, we fall back to the lexicon-based filters already implemented.
# Decision path is logged at INFO level.

import os
import re
import unicodedata
import logging
from typing import List, Set, Tuple, Optional, Dict

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
logger = logging.getLogger("badwords_service")

# Globals
BAD_WORDS_EN: Set[str] = set()
BAD_WORDS_SI: Set[str] = set()
BAD_WORDS_SI_SINGLISH: Set[str] = set()

# ML model globals
MODEL_AVAILABLE: bool = False
MODEL_PATH: str = ""
_model_bundle = None  # holds {"vec_char", "vec_word", "classifier"}

# Minimal built-in fallback list to avoid empty English lexicon when remote fetch fails
DEFAULT_EN_BAD_WORDS: Set[str] = {
    "ass",
    "bastard",
    "bitch",
    "bloody",
    "bollocks",
    "crap",
    "cunt",
    "damn",
    "dick",
    "fuck",
    "motherfucker",
    "piss",
    "prick",
    "shit",
    "slut",
    "whore",
}

# Sinhala common words/particles to exclude (reduce false positives)
SINHALA_STOPWORDS: Set[str] = {
    "කොහොමද",
    "මොකද",
    "කර",
    "කරන",
    "කරන්නේ",
    "න්න",
    "කියලා",
    "මම",
    "ඔබ",
    "ඔයා",
    "අපි",
    "ඔක්කෝම",
    "තම",
    "තමා",
    "ඒ",
    "මේ",
    "එක",
    "හා",
    "ද",
    "නමුත්",
    "නෑ",
    "නැහැ",
    "ඉතින්",
}

# Built-in Sinhala bad word lemmas (minimal curated set)
DEFAULT_SI_BAD_WORDS: Set[str] = {
    "මෝඩ",     # fool / stupid
    "පොන්න",   # insult
    "හමඩ",     # coarse insult variant
    "මලා",     # can be insult in context (keep minimal)
    "හරක්",    # donkey (insult)
    "ගැනි",    # coarse
}

# Built-in Singlish bad words (minimal curated)
DEFAULT_SINGLISH_BAD_WORDS: Set[str] = {
    "moda", "modayo", "modaya",
    "ponnaya", "ponnayo",
    "harak", "booruwa",
}

# Aho-corasick automatons (optional)
AC_AUTOMATON_EN = None
AC_AUTOMATON_SI = None
AC_AUTOMATON_SI_SING = None

# Precompiled regexes
RE_LATIN = re.compile(r"[A-Za-z']+")
RE_SINHALA = re.compile(r"[\u0D80-\u0DFF]+")
RE_WORDSEP = re.compile(r"[\s\-_,.;:\|\\/]+")
RE_NONALNUM = re.compile(r"[^\w\u0D80-\u0DFF]", flags=re.UNICODE)


def _preview(text: str, n: int = 140) -> str:
    return text if len(text) <= n else text[:n] + "...(truncated)"


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text))
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
    """Produce several deobfuscated variants of input text."""
    t = normalize_text(text)
    before = t
    # replace homoglyphs
    t = "".join(HOMOGLYPH_MAP.get(ch, ch) for ch in t)
    # collapse >2 repeats to two
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    # join letters separated by punctuation/spaces: s.h.i.t -> shit
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
        "User-Agent": "badwords-service/1.2 (+https://example.local)",
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
        urls.append(f"https://rawcdn.githack.com/{user}/{repo}/{branch}/{path}")
    return urls


# --- Loading lexicons
def load_en_bad_words() -> Set[str]:
    override = os.getenv("BAD_WORDS_EN_URL", "").strip()
    if override:
        logger.info("Loading English bad words from override URL: %s", override)
        words = {w.lower() for w in fetch_text_lines(override)}
        logger.info("Loaded %d English bad words (override)", len(words))
        if words:
            return words

    logger.info("Loading English bad words (LDNOOBW)...")
    urls = [
        "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en.txt",
        "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/main/en.txt",
    ]
    words: Set[str] = set()
    for u in urls:
        fetched = fetch_text_lines(u)
        if fetched:
            for w in fetched:
                words.add(w.lower())
            break

    if not words:
        logger.warning(
            "LDNOOBW fetch failed; using built-in minimal English list. "
            "Set BAD_WORDS_EN_URL to your own list for full coverage."
        )
        words = set(DEFAULT_EN_BAD_WORDS)

    logger.info("Loaded %d English bad words", len(words))
    return words


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


def load_si_bad_words_sold() -> Set[str]:
    if os.getenv("USE_SOLD", "0").lower() not in {"1", "true", "yes"}:
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

    import ast

    for split_name in sold.keys():
        split = sold[split_name]
        for item in split:
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
                try:
                    parsed_tokens = ast.literal_eval(tokens)
                    if isinstance(parsed_tokens, (list, tuple)):
                        toks = [str(x) for x in parsed_tokens]
                    else:
                        toks = str(parsed_tokens).split()
                except Exception:
                    toks = tokens.split()
            else:
                toks = [str(x) for x in tokens]

            rats: List[int] = []
            if isinstance(rationals, str):
                try:
                    parsed = ast.literal_eval(rationals)
                    if isinstance(parsed, (list, tuple)):
                        for x in parsed:
                            try:
                                rats.append(int(x))
                            except Exception:
                                continue
                    else:
                        for x in str(parsed).replace(",", " ").split():
                            if x.strip().lstrip("-").isdigit():
                                try:
                                    rats.append(int(x))
                                except Exception:
                                    continue
                except Exception:
                    for x in rationals.replace("[", " ").replace("]", " ").replace(",", " ").split():
                        if x.strip().lstrip("-").isdigit():
                            try:
                                rats.append(int(x))
                            except Exception:
                                continue
            else:
                try:
                    rats = [int(x) for x in rationals]
                except Exception:
                    rats = []

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


def load_additional_si_bad_csv(path: str = "bad.csv") -> Set[str]:
    """Load comma-separated Sinhala bad words from a local CSV file."""
    words: Set[str] = set()
    try:
        if not path:
            return words
        p = path.strip()
        if not os.path.exists(p) or not os.path.isfile(p):
            return words
        import csv
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        import re as _re
        tokens = [tok.strip() for tok in _re.split(r"[,\\n;]+", content) if tok.strip()]
        try:
            with open(p, "r", encoding="utf-8", newline="") as f2:
                reader = csv.reader(f2, delimiter=",")
                for row in reader:
                    for cell in row:
                        cell = cell.strip()
                        if cell:
                            tokens.append(cell)
        except Exception:
            pass
        for w in tokens:
            w_norm = unicodedata.normalize("NFKC", w)
            if w_norm:
                words.add(w_norm)
        logger.info("Loaded %d Sinhala bad words from CSV: %s", len(words), p)
    except Exception as e:
        logger.warning("Failed to load extra Sinhala bad words CSV '%s': %s", path, e)
    return words


def build_automaton(words: Set[str]) -> Optional[object]:
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

    # Build Sinhala set and apply stopword filter, include built-in defaults
    BAD_WORDS_SI = set(DEFAULT_SI_BAD_WORDS)
    si_extra_csv = load_additional_si_bad_csv(os.getenv("BAD_CSV_PATH", "bad.csv"))
    for w in si_unicode_mrmrvl.union(si_sold).union(si_semisold).union(si_extra_csv):
        w_norm = unicodedata.normalize("NFKC", w)
        if len(w_norm) < 3:
            continue
        if w_norm in SINHALA_STOPWORDS:
            continue
        BAD_WORDS_SI.add(w_norm)

    # Singlish set (lowercased) include built-in defaults
    BAD_WORDS_SI_SINGLISH = set(x.lower() for x in DEFAULT_SINGLISH_BAD_WORDS)
    for w in si_singlish_mrmrvl:
        w_l = w.lower().strip()
        if len(w_l) < 2:
            continue
        BAD_WORDS_SI_SINGLISH.add(w_l)

    # Custom overrides from env
    en_c, si_c, si_sing_c = load_custom_words_from_env()
    BAD_WORDS_EN.update(en_c)
    BAD_WORDS_SI.update({unicodedata.normalize("NFKC", w) for w in si_c if len(w) >= 3 and w not in SINHALA_STOPWORDS})
    BAD_WORDS_SI_SINGLISH.update({w.lower() for w in si_sing_c if len(w) >= 2})

    # build automatons if available
    if HAS_AHO:
        try:
            AC_AUTOMATON_EN = build_automaton(BAD_WORDS_EN)
            AC_AUTOMATON_SI = None  # Avoid Sinhala substring automaton to reduce false positives
            AC_AUTOMATON_SI_SING = build_automaton(BAD_WORDS_SI_SINGLISH)
            logger.info("Aho-Corasick automatons built (enabled for EN/Singlish)")
        except Exception as e:
            logger.warning("Failed to build Aho automatons: %s", e)
    else:
        logger.info("pyahocorasick not available: substring scanning will be slower")

    logger.info(
        "Lexicons ready: EN=%d SI=%d SI_singlish=%d (USE_SOLD=%s USE_SEMISOLD=%s) HAS_AHO=%s HAS_RAPIDFUZZ=%s",
        len(BAD_WORDS_EN),
        len(BAD_WORDS_SI),
        len(BAD_WORDS_SI_SINGLISH),
        os.getenv("USE_SOLD", "0"),
        os.getenv("USE_SEMISOLD", "0"),
        HAS_AHO,
        HAS_RAPIDFUZZ,
    )


# --- ML model loader/inference ---

def _prepare_unpickle_namespace() -> None:
    """
    Ensure joblib can unpickle vectorizers that reference normalize_text
    from train_gemma_badwords by providing a stub module in sys.modules.
    """
    try:
        import types, sys as _sys
        if "train_gemma_badwords" not in _sys.modules:
            mod = types.ModuleType("train_gemma_badwords")
            # Provide a compatible normalize_text used during training
            def _norm(t: str) -> str:
                s = normalize_text(t)
                # match training behavior: collapse repeats and remove separators
                s = re.sub(r"(.)\\1{2,}", r"\\1\\1", s)
                s = re.sub(r"(?:[\\W_])+(?=[A-Za-z])", "", s)
                if HAS_UNIDECODE:
                    try:
                        return unidecode(s)
                    except Exception:
                        return s
                return s
            mod.normalize_text = _norm  # type: ignore[attr-defined]
            _sys.modules["train_gemma_badwords"] = mod
    except Exception as e:
        logger.warning("Failed to prepare unpickle namespace: %s", e)


def _discover_model_path() -> Optional[str]:
    """
    Try multiple locations to find model.joblib:
    - $MODEL_DIR/model.joblib
    - ./model/model.joblib
    - ./outputs/badwords-ml/model.joblib
    - Path declared in ./model/inference_card.json (model_path/model.joblib)
    """
    candidates: List[str] = []

    # Env override
    env_dir = os.getenv("MODEL_DIR", "").strip()
    if env_dir:
        candidates.append(os.path.join(env_dir, "model.joblib"))

    # Standard repo locations
    candidates.append(os.path.join("model", "model.joblib"))
    candidates.append(os.path.join("outputs", "badwords-ml", "model.joblib"))

    # Read inference card for model_path
    try:
        card_path = os.path.join("model", "inference_card.json")
        if os.path.exists(card_path):
            import json
            with open(card_path, "r", encoding="utf-8") as f:
                card = json.load(f)
            mdir = str(card.get("model_path", "")).strip()
            if mdir:
                candidates.append(os.path.join(mdir, "model.joblib"))
    except Exception as e:
        logger.warning("Failed to read inference_card.json: %s", e)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    for p in uniq:
        if os.path.exists(p) and os.path.isfile(p):
            return p
    logger.info("Model discovery: none of candidates exist: %s", ", ".join(uniq))
    return None


def load_model() -> None:
    """Load joblib model bundle using discovered path."""
    global MODEL_AVAILABLE, MODEL_PATH, _model_bundle
    MODEL_AVAILABLE = False
    _model_bundle = None

    model_file = _discover_model_path()
    if not model_file:
        logger.info("Model file not found; ML detector disabled.")
        return

    MODEL_PATH = model_file
    try:
        _prepare_unpickle_namespace()
        from joblib import load as joblib_load  # type: ignore
        _model_bundle = joblib_load(model_file)
        for key in ("vec_char", "vec_word", "classifier"):
            if key not in _model_bundle:
                raise ValueError(f"Model bundle missing '{key}'")
        MODEL_AVAILABLE = True
        logger.info("ML model loaded from %s", model_file)
    except Exception as e:
        logger.warning("Failed to load ML model from %s: %s", model_file, e)
        MODEL_AVAILABLE = False
        _model_bundle = None


def model_predict_is_bad(text: str) -> Optional[bool]:
    """
    Return True if model predicts 'bad', False if 'clean', or None if model unavailable or error.
    """
    if not MODEL_AVAILABLE or _model_bundle is None:
        return None
    try:
        vec_char = _model_bundle["vec_char"]
        vec_word = _model_bundle["vec_word"]
        clf = _model_bundle["classifier"]
        # The vectorizers already embed normalization preprocessors from training.
        Xc = vec_char.transform([text])
        Xw = vec_word.transform([text])
        # Use scipy.sparse.hstack without importing globally (avoid heavy dep in import path)
        from scipy.sparse import hstack as _hstack  # type: ignore
        X = _hstack([Xc, Xw])
        y_pred = clf.predict(X)
        return bool(int(y_pred[0]) == 1)
    except Exception as e:
        logger.warning("Model inference failed: %s", e)
        return None


# Matching helpers
def _automaton_find(automaton, text: str) -> Set[str]:
    hits = set()
    if automaton is None:
        return hits
    try:
        for _, (_, word) in automaton.iter(text):
            hits.add(word)
    except Exception:
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
    """Return candidate words that fuzzily match token above threshold."""
    if not HAS_RAPIDFUZZ:
        return set()
    try:
        results = process.extract(token, list(candidates), scorer=fuzz.partial_ratio, limit=limit)
        return {r[0] for r, score, _ in results if score >= threshold}
    except Exception:
        out = set()
        for w in candidates:
            if token in w or w in token:
                out.add(w)
        return out


def find_bad_words(text: str) -> Set[str]:
    """
    Stricter lexicon-based bad words finder:
    - Token-level exact matches only (no substring scanning, no fuzzy).
    - Reduces false positives compared to earlier approach.
    """
    hits: Set[str] = set()
    # Consider normalized and deobfuscated variants, but only for tokenization
    variants = [normalize_text(text), deobfuscate(text)]

    tried = set()
    for v in variants:
        if not v or v in tried:
            continue
        tried.add(v)
        latin_tokens, sinhala_tokens = tokenize(v)

        # Exact token matches only, with simple length guards
        for tok in latin_tokens:
            if len(tok) < 2:
                continue
            if tok in BAD_WORDS_EN or tok in BAD_WORDS_SI_SINGLISH:
                hits.add(tok)
        for tok in sinhala_tokens:
            if len(tok) < 2:
                continue
            if tok in BAD_WORDS_SI:
                hits.add(tok)

    if hits:
        logger.debug("Final hits (strict): %s", sorted(hits)[:100])
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
    load_model()
    init_lexicons()
    logger.info("Startup complete. MODEL_AVAILABLE=%s", MODEL_AVAILABLE)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_available": MODEL_AVAILABLE,
        "model_path": MODEL_PATH,
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
    text_preview = _preview(payload.text)
    logger.info("Incoming /check: advanced=%s text_preview=%s", payload.advanced, text_preview)

    # 1) Try ML model first if available
    model_result = model_predict_is_bad(payload.text)
    if model_result is True:
        logger.info("Decision: model=BAD -> returning found=True (priority: model). text_preview=%s", text_preview)
        if payload.advanced:
            # We can also return lexicon hits to help explain, but model took precedence
            hits = find_bad_words(payload.text)
            logger.info("Model decided BAD; lexicon hits=%d (for transparency).", len(hits))
            return AdvancedResponse(found=True, bad_words=sorted(hits))
        return DefaultResponse(found=True)
    elif model_result is False:
        logger.info("Decision: model=CLEAN -> falling back to lexicon filters. text_preview=%s", text_preview)
    else:
        logger.info("Decision: model=UNAVAILABLE/ERROR -> using lexicon filters only. text_preview=%s", text_preview)

    # 2) Fallback: lexicon-based detection
    hits = find_bad_words(payload.text)
    found = bool(hits)
    logger.info("Fallback lexicon decision: found=%s hits=%d text_preview=%s", found, len(hits), text_preview)

    if payload.advanced:
        return AdvancedResponse(found=found, bad_words=sorted(hits, key=lambda s: (len(s), s)))
    return DefaultResponse(found=found)


# Helpful __main__ for quick local testing
if __name__ == "__main__":
    # quick sanity checks
    examples = [
        "you are a s.h.i.t!",
        "ඔබ ම*ල*කේ",
        "obfu$cat3d discussion",
        "mama api *******",
        "harak modaya",
    ]
    load_model()
    init_lexicons()
    for ex in examples:
        print("> ", ex)
        mr = model_predict_is_bad(ex)
        print("  model:", mr)
        print("  lexicon hits:", find_bad_words(ex))

    # run uvicorn when invoked directly
    import uvicorn

    port_env = os.getenv("PORT")
    try:
        port = int(port_env) if port_env and port_env.isdigit() else 8000
    except Exception:
        port = 8000
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("app:app", host=host, port=port, reload=False)
