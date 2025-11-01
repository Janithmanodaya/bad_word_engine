import os
import re
import unicodedata
from typing import List, Set, Tuple
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Globals for lexicons
BAD_WORDS_EN: Set[str] = set()
BAD_WORDS_SI: Set[str] = set()
BAD_WORDS_SI_SINGLISH: Set[str] = set()


def normalize_text(text: str) -> str:
    # Normalize to NFKC and lower-case
    t = unicodedata.normalize("NFKC", text)
    # Remove zero-width characters
    t = t.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    return t.lower()


def tokenize(text: str) -> Tuple[List[str], List[str]]:
    # English/Latin tokens
    latin_tokens = re.findall(r"[A-Za-z']+", text)
    # Sinhala Unicode block tokens (U+0D80..U+0DFF)
    sinhala_tokens = re.findall(r"[\u0D80-\u0DFF]+", text)
    return latin_tokens, sinhala_tokens


def fetch_text_lines(url: str, timeout: int = 15) -> List[str]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and r.text:
            return [line.strip() for line in r.text.splitlines() if line.strip()]
    except Exception:
        pass
    return []


def load_en_bad_words() -> Set[str]:
    # LDNOOBW english list
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
    return words


def load_si_bad_words_mrmrvl() -> Tuple[Set[str], Set[str]]:
    # Try multiple plausible file paths (Unicode and Singlish lists)
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

    return si_unicode, si_singlish


def load_si_bad_words_sold() -> Set[str]:
    # Derive Sinhala offensive tokens from the SOLD dataset if available
    words: Set[str] = set()
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return words

    try:
        sold = load_dataset("sinhala-nlp/SOLD")
    except Exception:
        return words

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
    return words


def init_lexicons() -> None:
    global BAD_WORDS_EN, BAD_WORDS_SI, BAD_WORDS_SI_SINGLISH
    BAD_WORDS_EN = load_en_bad_words()

    si_unicode_mrmrvl, si_singlish_mrmrvl = load_si_bad_words_mrmrvl()
    si_sold = load_si_bad_words_sold()

    # Merge sources
    BAD_WORDS_SI = set()
    for w in si_unicode_mrmrvl.union(si_sold):
        BAD_WORDS_SI.add(unicodedata.normalize("NFKC", w))

    BAD_WORDS_SI_SINGLISH = set()
    for w in si_singlish_mrmrvl:
        BAD_WORDS_SI_SINGLISH.add(w.lower())


def check_api_key(req: Request, expected_key: str) -> None:
    key = req.headers.get("X-API-Key", "")
    if not expected_key or key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


def find_bad_words(text: str) -> Set[str]:
    t_norm = normalize_text(text)
    latin_tokens, sinhala_tokens = tokenize(t_norm)

    hits: Set[str] = set()

    # English
    for tok in latin_tokens:
        if tok in BAD_WORDS_EN or tok in BAD_WORDS_SI_SINGLISH:
            hits.add(tok)

    # Sinhala (exact token matches)
    for tok in sinhala_tokens:
        if tok in BAD_WORDS_SI:
            hits.add(tok)

    # Sinhala substring fallback (handles morphological attachments)
    for w in BAD_WORDS_SI:
        if w and w in t_norm:
            hits.add(w)

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
    return {"status": "ok", "en_words": len(BAD_WORDS_EN), "si_words": len(BAD_WORDS_SI)}


@app.post("/check")
async def check_default(req: Request, payload: CheckRequest):
    api_key = os.getenv("API_KEY", "")
    check_api_key(req, api_key)
    hits = find_bad_words(payload.text)
    if payload.advanced:
        return AdvancedResponse(found=bool(hits), bad_words=sorted(hits, key=lambda s: (len(s), s)))
    return DefaultResponse(found=bool(hits))


def parse_host_port(url_str: str) -> Tuple[str, int]:
    default_host = "0.0.0.0"
    default_port = 8000
    if not url_str:
        return default_host, default_port

    # Add scheme if missing to help urlparse
    if "://" not in url_str:
        url_str_work = "http://" + url_str
    else:
        url_str_work = url_str

    parsed = urlparse(url_str_work)
    host = parsed.hostname or default_host
    port = parsed.port or default_port
    return host, port


if __name__ == "__main__":
    # Prefer platform-provided PORT, and always bind to 0.0.0.0 inside containers.
    # Fallback to SERVER_URL only for the port if PORT is not set.
    port_env = os.getenv("PORT")
    if port_env and port_env.isdigit():
        port = int(port_env)
    else:
        _, port = parse_host_port(os.getenv("SERVER_URL", ""))

    host = os.getenv("HOST", "0.0.0.0")

    # Lazy import to avoid dependency if not running directly
    import uvicorn

    uvicorn.run("app:app", host=host, port=port, reload=False)
