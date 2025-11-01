import os
import re
import unicodedata
import logging
from typing import List, Set, Tuple
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Logging setup
def _get_log_level() -> int:
    lvl = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, lvl, logging.INFO)


logging.basicConfig(level=_get_log_level(), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("badwords")

# Globals for lexicons
BAD_WORDS_EN: Set[str] = set()
BAD_WORDS_SI: Set[str] = set()
BAD_WORDS_SI_SINGLISH: Set[str] = set()


def _preview(text: str, n: int = 160) -> str:
    if len(text) <= n:
        return text
    return text[:n] + "...(truncated)"


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    return t.lower()


def deobfuscate(text: str) -> str:
    t = normalize_text(text)
    # Leet/symbol substitutions for Latin content
    subs = {
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
    }
    before = t
    t = "".join(subs.get(ch, ch) for ch in t)
    # Collapse repeated characters (3+ -> 2)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    # Strip diacritics for Latin
    t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    logger.debug("Deobfuscate: before='%s' after='%s'", _preview(before), _preview(t))
    return t


def tokenize(text: str) -> Tuple[List[str], List[str]]:
    latin_tokens = re.findall(r"[A-Za-z']+", text)
    sinhala_tokens = re.findall(r"[\u0D80-\u0DFF]+", text)
    logger.debug("Tokenize: latin=%d sinhala=%d", len(latin_tokens), len(sinhala_tokens))
    return latin_tokens, sinhala_tokens


def fetch_text_lines(url: str, timeout: int = 15) -> List[str]:
    logger.info("Downloading list from %s", url)
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and r.text:
            lines = [line.strip() for line in r.text.splitlines() if line.strip()]
            logger.info("Fetched %d lines from %s", len(lines), url)
            return lines
        logger.warning("Non-200 response from %s: %s", url, r.status_code)
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
    return []


def load_en_bad_words() -> Set[str]:
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
    logger.info("Loaded %d English bad words", len(words))
    return words


def load_si_bad_words_mrmrvl() -> Tuple[Set[str], Set[str]]:
    logger.info("Loading Sinhala MRVLS lists (unicode + singlish)...")
    candidates_unicode = [
        "https://raw.githubusercontent.com/MrMRVLS/sinhala-bad-words-list/main/sinhala-bad-words-unicode.txt",
        "https://raw.githubusercontent.com/MrMRVLS/sinhala-bad-words-list/master/sinhala-bad-words-unicode.txt",
        "https://raw.githubusercontent.com/MrMRVLS/sinhala-bad-words-list/main/unicode.txt",
        "https://raw.githubusercontent.com/MrMRVLS/sinhala-bad-words-list/master/unicode.txt",
    ]
    candidates_singlish = [
        "https://raw.githubusercontent.com/MrMRVLS/sinhala-bad-words-list/main/sinhala-bad-words-singlish.txt",
        "https://raw.githubusercontent.com/MrMRVLS/sinhala-bad-words-list/master/sinhala-bad-words-singlish.txt",
        "https://raw.githubusercontent.com/MrMRVLS/sinhala-bad-words-list/main/singlish.txt",
        "https://raw.githubusercontent.com/MrMRVLS/sinhala-bad-words-list/master/singlish.txt",
    ]
    si_unicode: Set[str] = set()
    si_singlish: Set[str] = set()

    for u in candidates_unicode:
        for w in fetch_text_lines(u):
            si_unicode.add(w)
        if si_unicode:
            break

    for u in candidates_singlish:
        for w in fetch_text_lines(u):
            si_singlish.add(w.lower())
        if si_singlish:
            break

    logger.info("Loaded Sinhala MRVLS: unicode=%d singlish=%d", len(si_unicode), len(si_singlish))
    return si_unicode, si_singlish


def load_si_bad_words_sold() -> Set[str]:
    """
    Load token-level offensive Sinhala words from SOLD rationales.
    Controlled by USE_SOLD env var (default: enabled).
    """
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
        logger.info("Loaded SOLD splits: %s", list(sold.keys()))
    except Exception as e:
        logger.warning("Failed to load SOLD: %s", e)
        return words

    added = 0
    for split_name in ["train", "test"]:
        split = sold.get(split_name)
        if split is None:
            continue
        for item in split:
            tokens = item.get("tokens")
            rationals = item.get("rationals")
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
                    added += 1
    logger.info("Collected %d SOLD rationale tokens (%d unique)", added, len(words))
    return words


def load_custom_words_from_env() -> Tuple[Set[str], Set[str], Set[str]]:
    en: Set[str] = set()
    si: Set[str] = set()
    si_sing: Set[str] = set()

    # Comma-separated list
    raw = os.getenv("BAD_WORDS_CUSTOM", "")
    if raw:
        for w in raw.split(","):
            w = w.strip()
            if not w:
                continue
            # Heuristic: Sinhala block vs Latin
            if re.search(r"[\u0D80-\u0DFF]", w):
                si.add(unicodedata.normalize("NFKC", w))
            else:
                en.add(w.lower())

    # Optional URL to a text file (one word per line)
    url = os.getenv("BAD_WORDS_URL", "")
    if url:
        for w in fetch_text_lines(url):
            if re.search(r"[\u0D80-\u0DFF]", w):
                si.add(unicodedata.normalize("NFKC", w))
            else:
                en.add(w.lower())

    return en, si, si_sing


def _load_semisold_tokens(threshold: float = 0.8, top_tokens: int = 1000) -> Set[str]:
    """
    Heuristically derive additional Sinhala offensive tokens from SemiSOLD by:
    - averaging available score columns per row
    - selecting rows with score >= threshold
    - extracting Sinhala tokens and taking most frequent top_tokens
    Controlled by USE_SEMISOLD env var (default: disabled).
    """
    if os.getenv("USE_SEMISOLD", "0").lower() not in {"1", "true", "yes"}:
        logger.info("USE_SEMISOLD disabled; skipping SemiSOLD mining.")
        return set()

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        logger.warning("datasets library not available for SemiSOLD: %s", e)
        return set()

    try:
        logger.info("Loading SemiSOLD dataset from HuggingFace...")
        ds = load_dataset("sinhala-nlp/SemiSOLD", split="train")
        logger.info("Loaded SemiSOLD rows: %d", len(ds))
    except Exception as e:
        logger.warning("Failed to load SemiSOLD: %s", e)
        return set()

    # Identify numeric score columns dynamically (values typically in [0,1])
    score_cols = []
    sample = ds[0] if len(ds) else {}
    for k, v in sample.items():
        if isinstance(v, (int, float)):
            score_cols.append(k)
    if not score_cols:
        # fallback to known names
        score_cols = ["xlmr", "xlmt", "mbert", "sinbert", "lstm_ft", "cnn_ft", "lstm_cbow", "cnn_cbow", "lstm_sl", "cnn_sl", "svm"]
    logger.info("SemiSOLD score columns detected: %s", score_cols)

    kept_rows = 0
    # Collect tokens from high-score rows
    freq = {}
    for row in ds:
        scores = [float(row.get(c, 0.0)) for c in score_cols if isinstance(row.get(c, None), (int, float))]
        if not scores:
            continue
        mean_score = sum(scores) / len(scores)
        if mean_score < threshold:
            continue
        kept_rows += 1
        text = normalize_text(str(row.get("text", "")))
        # Extract Sinhala tokens only
        _, si_tokens = tokenize(text)
        for tok in si_tokens:
            tok_n = unicodedata.normalize("NFKC", tok)
            freq[tok_n] = freq.get(tok_n, 0) + 1

    logger.info("SemiSOLD rows kept after threshold %.2f: %d", threshold, kept_rows)

    # Take top tokens
    sorted_toks = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_tokens]
    result = {t for t, _ in sorted_toks}
    logger.info("SemiSOLD harvested tokens selected: %d (top %d)", len(result), top_tokens)
    return result


def init_lexicons() -> None:
    logger.info("Initializing lexicons...")
    global BAD_WORDS_EN, BAD_WORDS_SI, BAD_WORDS_SI_SINGLISH
    BAD_WORDS_EN = load_en_bad_words()

    si_unicode_mrmrvl, si_singlish_mrmrvl = load_si_bad_words_mrmrvl()
    si_sold = load_si_bad_words_sold()
    # Optional SemiSOLD harvesting
    thr = float(os.getenv("SEMISOLD_THRESHOLD", "0.8"))
    topn = int(os.getenv("SEMISOLD_TOP_TOKENS", "1000"))
    si_semisold = _load_semisold_tokens(threshold=thr, top_tokens=topn)

    BAD_WORDS_SI = set()
    for w in si_unicode_mrmrvl.union(si_sold).union(si_semisold):
        BAD_WORDS_SI.add(unicodedata.normalize("NFKC", w))

    BAD_WORDS_SI_SINGLISH = set()
    for w in si_singlish_mrmrvl:
        BAD_WORDS_SI_SINGLISH.add(w.lower())

    # Merge custom env-provided words
    en_c, si_c, si_sing_c = load_custom_words_from_env()
    BAD_WORDS_EN.update(en_c)
    BAD_WORDS_SI.update(si_c)
    BAD_WORDS_SI_SINGLISH.update(si_sing_c)

    logger.info(
        "Lexicons ready: EN=%d SI=%d SI_singlish=%d (USE_SOLD=%s USE_SEMISOLD=%s)",
        len(BAD_WORDS_EN),
        len(BAD_WORDS_SI),
        len(BAD_WORDS_SI_SINGLISH),
        os.getenv("USE_SOLD", "1"),
        os.getenv("USE_SEMISOLD", "0"),
    )


def check_api_key(req: Request, expected_key: str) -> None:
    key = req.headers.get("X-API-Key", "")
    if not expected_key or key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _match_substrings(text: str, candidates: Set[str], min_len: int = 4) -> Set[str]:
    hits: Set[str] = set()
    for w in candidates:
        if len(w) >= min_len and w in text:
            hits.add(w)
    return hits


def find_bad_words(text: str) -> Set[str]:
    hits: Set[str] = set()

    variants = [normalize_text(text), deobfuscate(text)]
    for idx, t_norm in enumerate(variants):
        logger.debug("Variant %d preview='%s'", idx, _preview(t_norm))
        latin_tokens, sinhala_tokens = tokenize(t_norm)

        exact_before = len(hits)
        for tok in latin_tokens:
            if tok in BAD_WORDS_EN or tok in BAD_WORDS_SI_SINGLISH:
                hits.add(tok)
        for tok in sinhala_tokens:
            if tok in BAD_WORDS_SI:
                hits.add(tok)
        exact_after = len(hits)

        # Substring fallbacks
        hits.update(_match_substrings(t_norm, BAD_WORDS_SI, min_len=2))
        hits.update(_match_substrings(t_norm, BAD_WORDS_EN, min_len=4))
        hits.update(_match_substrings(t_norm, BAD_WORDS_SI_SINGLISH, min_len=4))

        logger.debug("Variant %d: exact_hits_added=%d total_hits=%d", idx, exact_after - exact_before, len(hits))

    if hits:
        logger.debug("Final hits: %s", sorted(hits, key=lambda s: (len(s), s))[:50])
    return hits


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
        "use_sold": os.getenv("USE_SOLD", "1"),
        "use_semisold": os.getenv("USE_SEMISOLD", "0"),
    }


@app.post("/check")
async def check_default(req: Request, payload: CheckRequest):
    api_key = os.getenv("API_KEY", "")
    check_api_key(req, api_key)
    logger.info("Incoming /check: advanced=%s text_preview='%s'", payload.advanced, _preview(payload.text))
    hits = find_bad_words(payload.text)
    logger.info("Outgoing /check: found=%s hits=%d", bool(hits), len(hits))
    if payload.advanced:
        return AdvancedResponse(found=bool(hits), bad_words=sorted(hits, key=lambda s: (len(s), s)))
    return DefaultResponse(found=bool(hi_codetsnew)</)



def parse_host_port(url_str: str) -> Tuple[str, int]:
    default_host = "0.0.0.0"
    default_port = 8000
    if not url_str:
        return default_host, default_port

    if "://" not in url_str:
        url_str_work = "http://" + url_str
    else:
        url_str_work = url_str

    parsed = urlparse(url_str_work)
    host = parsed.hostname or default_host
    port = parsed.port or default_port
    return host, port


if __name__ == "__main__":
    port_env = os.getenv("PORT")
    if port_env and port_env.isdigit():
        port = int(port_env)
    else:
        _, port = parse_host_port(os.getenv("SERVER_URL", ""))

    host = os.getenv("HOST", "0.0.0.0")

    import uvicorn

    uvicorn.run("app:app", host=host, port=port, reload=False)