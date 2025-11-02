#!/usr/bin/env python3
"""
Compact bad-word classifier trainer using simple ML (no LLM).

- Trains a small model with scikit-learn (TF-IDF char/word n-grams + LinearSVC)
- Optional: FastText training (if installed) for robust character variation handling
- Keeps existing dataset options (CSV/JSON/JSONL), auto wordlist builder, and SOLD fallback
- Designed to run quickly on CPU-only environments

Example:
    python train_gemma_badwords.py \\
      --dataset_path data/auto_badwords_en_si.csv \\
      --text_column text --label_column label \\
      --output_dir outputs/badwords-ml \\
      --model_choice sklearn
"""

import argparse
import json
import os
import sys
import subprocess
import importlib.util
import urllib.request
from typing import Optional, List, Tuple, Dict
import shutil
import time
import logging
from collections import defaultdict

# ---------------------------
# Colab detection
# ---------------------------

def in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

# ---------------------------
# Bootstrap: optional dependency setup (disabled by default)
# ---------------------------

REQUIRED_PKGS = [
    "numpy",
    "scikit-learn",
    "datasets",
    "joblib",
    "Unidecode",
]

def _pip_install(args: list) -> None:
    logging.getLogger("setup").info(f"Installing: {' '.join(args)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + args)

def _is_installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None

def ensure_dependencies():
    """
    Optionally install missing packages at runtime.
    Disabled by default to avoid pip activity during normal training runs.
    Use --setup_deps to enable, or rely on requirements.txt.
    """
    missing = []
    import_name_map = {
        "scikit-learn": "sklearn",
        "joblib": "joblib",
        "Unidecode": "unidecode",
    }
    for spec in REQUIRED_PKGS:
        mod_name = import_name_map.get(spec, spec.split("==")[0].split(">=")[0])
        if not _is_installed(mod_name):
            missing.append(spec)

    if missing:
        print("[setup] Installing missing dependencies...")
        for spec in missing:
            try:
                _pip_install([spec])
            except subprocess.CalledProcessError as e:
                print(f"[setup] WARNING: Failed to install {spec}: {e}")

    # Final check logging
    for spec in REQUIRED_PKGS:
        name = spec.split("==")[0].split(">=")[0]
        print(f"[setup] {name}: {'OK' if _is_installed(name) else 'MISSING'}")

# Imports (assume environment has requirements installed)
import numpy as np
from datasets import load_dataset, DatasetDict
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.sparse import hstack  # type: ignore
try:
    from unidecode import unidecode  # type: ignore
    HAS_UNIDECODE = True
except Exception:
    HAS_UNIDECODE = False

# ---------------------------
# Wordlist utilities
# ---------------------------

DEFAULT_BADWORDS_REPO = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master"
DEFAULT_CLEANWORDS_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt"

# Simple download event log for clear PASS/FAIL reporting
DOWNLOAD_EVENTS: List[dict] = []

def _record_download(stage: str, url: str, ok: bool, count: int = 0, error: str = "") -> None:
    DOWNLOAD_EVENTS.append({
        "stage": stage,
        "url": url,
        "ok": bool(ok),
        "count": int(count) if count else 0,
        "error": error,
    })

def _summarize_downloads() -> None:
    if not DOWNLOAD_EVENTS:
        return
    logging.info("=" * 70)
    logging.info("Download summary (PASS/FAIL):")
    by_stage: dict = defaultdict(list)
    for ev in DOWNLOAD_EVENTS:
        by_stage[ev["stage"]].append(ev)
    for stage in sorted(by_stage.keys()):
        stage_events = by_stage[stage]
        ok_n = sum(1 for e in stage_events if e["ok"])
        fail_n = sum(1 for e in stage_events if not e["ok"])
        logging.info(f"- {stage}: {ok_n} OK, {fail_n} FAIL")
        for e in stage_events:
            status = "OK" if e["ok"] else "FAIL"
            extra = f" (lines={e['count']})" if e["ok"] else f" (error={e['error']})"
            logging.info(f"  [{status}] {e['url']}{extra}")
    logging.info("=" * 70)

def download_bad_words(langs: List[str], repo_base: str = DEFAULT_BADWORDS_REPO, si_overrides: Optional[List[str]] = None) -> List[str]:
    bad: List[str] = []

    def _fetch_lines(stage: str, url: str) -> List[str]:
        try:
            with urllib.request.urlopen(url) as r:
                lines = [
                    line.strip()
                    for line in r.read().decode("utf-8", errors="ignore").splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
                _record_download(stage, url, ok=True, count=len(lines))
                return lines
        except Exception as e:
            _record_download(stage, url, ok=False, error=str(e))
            logging.warning(f"[wordlist] Failed to download {url}: {e}")
            return []

    for lang in langs:
        lang = lang.strip().lower()
        if not lang:
            continue

        if lang == "si":
            candidates = []
            if si_overrides:
                for u in si_overrides:
                    u = u.strip()
                    if u:
                        candidates.append(u)
            if not candidates:
                raise RuntimeError("Sinhala wordlist requested but no URLs provided. Set --si_wordlist_urls with one or more sources.")
            got_any = False
            for u in candidates:
                lines = _fetch_lines("badwords:si", u)
                if lines:
                    bad.extend(lines)
                    got_any = True
            if not got_any:
                raise RuntimeError("Sinhala wordlist URLs provided, but none could be fetched. Please verify the URLs or availability.")
            continue

        if lang == "en":
            candidates = [
                f"{repo_base}/en",
                f"{repo_base}/en.txt",
                repo_base.replace("master", "main") + "/en",
                repo_base.replace("master", "main") + "/en.txt",
            ]
        else:
            candidates = [
                f"{repo_base}/{lang}",
                f"{repo_base}/{lang}.txt",
                repo_base.replace("master", "main") + f"/{lang}",
                repo_base.replace("master", "main") + f"/{lang}.txt",
            ]

        for url in candidates:
            lines = _fetch_lines(f"badwords:{lang}", url)
            if lines:
                bad.extend(lines)
                break

    # Deduplicate, keep order
    seen = set()
    uniq = []
    for w in bad:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            uniq.append(wl)
    logging.info(f"[wordlist] Total bad words collected: {len(uniq)}")
    return uniq

def download_clean_words(url: str = DEFAULT_CLEANWORDS_URL, limit: int = 10000) -> List[str]:
    clean = []
    try:
        logging.info(f"[wordlist] Downloading clean words from {url}")
        with urllib.request.urlopen(url) as r:
            lines = r.read().decode("utf-8", errors="ignore").splitlines()
            _record_download("cleanwords", url, ok=True, count=len(lines))
            for i, line in enumerate(lines):
                if not line:
                    continue
                clean.append(line.strip().lower())
                if limit and len(clean) >= limit:
                    break
    except Exception as e:
        _record_download("cleanwords", url, ok=False, error=str(e))
        logging.warning(f"[wordlist] Failed to download clean words: {e}")
    logging.info(f"[wordlist] Total clean words collected: {len(clean)}")
    return clean

def build_wordlist_dataset_csv(out_path: str, bad_words: List[str], clean_words: List[str], augment_context: bool = True) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    import csv
    logging.info(f"[wordlist] Building dataset CSV at {out_path}")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for w in bad_words:
            if augment_context:
                examples = [w, f"You are {w}.", f"That was {w}!", f"{w} behavior."]
                for t in examples:
                    writer.writerow([t, 1])
            else:
                writer.writerow([w, 1])
        for w in clean_words:
            if augment_context:
                examples = [w, f"This is {w}.", f"A very {w} idea.", f"{w} example."]
                for t in examples:
                    writer.writerow([t, 0])
            else:
                writer.writerow([w, 0])
    logging.info(f"[wordlist] Wrote dataset rows: bad≈{len(bad_words)}xN, clean≈{len(clean_words)}xN")

# ---------------------------
# Utilities
# ---------------------------

def infer_file_type(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".jsonl") or lower.endswith(".jsonl.gz"):
        return "json"
    if lower.endswith(".json"):
        return "json"
    raise ValueError(f"Unsupported dataset file extension for: {path}")

def ensure_label_ints(dataset: DatasetDict, label_column: str) -> DatasetDict:
    unique = set(dataset["train"][label_column])
    mapping: Dict[str, int] = {}
    if all(isinstance(x, (int, np.integer)) for x in unique):
        return dataset  # already ints

    lowered = {str(x).lower() for x in unique}
    if lowered <= {"0", "1"}:
        mapping = {"0": 0, "1": 1}
    elif lowered <= {"false", "true"}:
        mapping = {"false": 0, "true": 1}
    elif lowered <= {"clean", "bad"}:
        mapping = {"clean": 0, "bad": 1}
    elif lowered <= {"safe", "toxic"}:
        mapping = {"safe": 0, "toxic": 1}
    elif lowered <= {"offensive", "not-offensive"}:
        mapping = {"not-offensive": 0, "offensive": 1}
    elif lowered <= {"off", "not"}:
        mapping = {"not": 0, "off": 1}
    else:
        mapping = {str(val).lower(): i for i, val in enumerate(sorted(lowered))}

    def map_labels(example):
        v = example[label_column]
        key = str(v).lower()
        if key in mapping:
            example[label_column] = mapping[key]
        else:
            try:
                example[label_column] = int(v)
            except Exception:
                example[label_column] = 0
        return example

    return dataset.map(map_labels)

def compute_class_weights_dict(labels: List[int]) -> Dict[int, float]:
    arr = np.array(labels)
    n_classes = int(arr.max()) + 1 if arr.size else 2
    counts = np.bincount(arr, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    total = counts.sum()
    weights = total / (counts * n_classes)
    return {i: float(w) for i, w in enumerate(weights)}

def maybe_download_dataset(dataset_path: str, dataset_url: Optional[str]) -> str:
    if os.path.exists(dataset_path):
        return dataset_path
    if dataset_url:
        os.makedirs(os.path.dirname(dataset_path) or ".", exist_ok=True)
        logging.info(f"[data] Downloading dataset from {dataset_url} -> {dataset_path}")
        with urllib.request.urlopen(dataset_url) as r, open(dataset_path, "wb") as f:
            shutil.copyfileobj(r, f)
        return dataset_path
    raise FileNotFoundError(f"Dataset path not found: {dataset_path}. Provide --dataset_url to download.")

# ---------------------------
# Data module
# ---------------------------

def build_dataset(path: str, text_col: str, label_col: str, seed: int, val_size: float) -> DatasetDict:
    ftype = infer_file_type(path)
    if ftype == "csv":
        raw = load_dataset("csv", data_files={"train": path})
    else:
        raw = load_dataset("json", data_files={"train": path}, split=None)

    ds = DatasetDict(raw) if isinstance(raw, dict) else DatasetDict({"train": raw})

    if "validation" not in ds:
        split = ds["train"].train_test_split(test_size=val_size, seed=seed)
        ds = DatasetDict({"train": split["train"], "validation": split["test"]})

    for col in (text_col, label_col):
        if col not in ds["train"].column_names:
            raise ValueError(f"Column '{col}' not found in dataset. Available: {ds['train'].column_names}")

    return ds

def build_sold_dataset(
    seed: int,
    val_size: float,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    trial_path: Optional[str] = None,
    parquet_train_api: Optional[str] = None,
    parquet_test_api: Optional[str] = None,
) -> DatasetDict:
    logging.info("[SOLD] Attempting to load 'sinhala-nlp/SOLD' from Hugging Face hub")
    ds: Optional[DatasetDict] = None
    try:
        raw = load_dataset("sinhala-nlp/SOLD")
        if isinstance(raw, dict):
            if "train" in raw and "test" in raw:
                ds = DatasetDict({"train": raw["train"], "validation": raw["test"]})
            elif "train" in raw and "validation" in raw:
                ds = DatasetDict({"train": raw["train"], "validation": raw["validation"]})
            else:
                base = list(raw.values())[0]
                split = base.train_test_split(test_size=val_size, seed=seed)
                ds = DatasetDict({"train": split["train"], "validation": split["test"]})
        else:
            split = raw.train_test_split(test_size=val_size, seed=seed)
            ds = DatasetDict({"train": split["train"], "validation": split["test"]})
        logging.info("[SOLD] Hub dataset loaded. train=%d, validation=%d", len(ds["train"]), len(ds["validation"]))
    except Exception as e:
        logging.warning("[SOLD] Hub load failed: %s. Trying remote parquet fallback.", e)

    def _fetch_parquet_urls(api_url: str) -> List[str]:
        urls: List[str] = []
        try:
            with urllib.request.urlopen(api_url) as r:
                data = json.loads(r.read().decode("utf-8"))
            if isinstance(data, dict):
                items = data.get("parquet_files") or data.get("urls") or data.get("files") or []
                if isinstance(items, list):
                    for item in items:
                        u = item.get("url") if isinstance(item, dict) else item
                        if isinstance(u, str) and u:
                            urls.append(u)
            elif isinstance(data, list):
                for item in data:
                    u = item.get("url") if isinstance(item, dict) else item
                    if isinstance(u, str) and u:
                        urls.append(u)
        except Exception as e:
            logging.warning("[SOLD] Failed to fetch parquet URLs from API '%s': %s", api_url, e)
        return urls

    if ds is None and parquet_train_api and parquet_test_api:
        train_urls = _fetch_parquet_urls(parquet_train_api)
        test_urls = _fetch_parquet_urls(parquet_test_api)
        if train_urls and test_urls:
            try:
                logging.info("[SOLD] Loading parquet from API URLs (train/test)")
                raw = load_dataset("parquet", data_files={"train": train_urls, "validation": test_urls})
                ds = DatasetDict({"train": raw["train"], "validation": raw["validation"]})
                logging.info("[SOLD] Parquet dataset loaded. train=%d, validation=%d", len(ds["train"]), len(ds["validation"]))
            except Exception as e:
                logging.warning("[SOLD] Parquet load failed: %s. Falling back to local TSV.", e)

    if ds is None:
        train_p = train_path or "data/SOLD_train.tsv"
        test_p = test_path or "data/SOLD_test.tsv"
        trial_p = trial_path or "data/SOLD_trial.tsv"

        def _maybe_load_tsv(path: str, split_name: str):
            if os.path.exists(path):
                logging.info("[SOLD] Loading local TSV: %s (%s split)", path, split_name)
                return load_dataset("csv", data_files={split_name: path}, delimiter="\t")[split_name]
            else:
                logging.warning("[SOLD] Local TSV not found: %s", path)
                return None

        train_split = _maybe_load_tsv(train_p, "train")
        test_split = _maybe_load_tsv(test_p, "validation")
        _ = _maybe_load_tsv(trial_p, "trial")

        if train_split is None:
            raise FileNotFoundError("[SOLD] Local TSV fallback failed: training file not found.")
        if test_split is None:
            logging.info("[SOLD] Validation TSV not found; creating validation split from train at ratio %.2f", val_size)
            split = train_split.train_test_split(test_size=val_size, seed=seed)
            ds = DatasetDict({"train": split["train"], "validation": split["test"]})
        else:
            ds = DatasetDict({"train": train_split, "validation": test_split})

        logging.info("[SOLD] Local TSV dataset loaded. train=%d, validation=%d", len(ds["train"]), len(ds["validation"]))

    def _guess_columns(column_names: List[str]) -> Tuple[str, str]:
        text_candidates = ["text", "comment", "sentence", "content"]
        label_candidates = ["label", "labels", "gold_label", "class", "target"]
        text_col = next((c for c in text_candidates if c in column_names), None)
        label_col = next((c for c in label_candidates if c in column_names), None)
        if not text_col or not label_col:
            raise ValueError(f"[SOLD] Could not infer text/label columns from: {column_names}")
        return text_col, label_col

    tcol_train, lcol_train = _guess_columns(ds["train"].column_names)
    tcol_val, lcol_val = _guess_columns(ds["validation"].column_names)
    if tcol_train != "text" or lcol_train != "label":
        ds = ds.rename_column(tcol_train, "text").rename_column(lcol_train, "label")
    if tcol_val != "text" or lcol_val != "label":
        ds = ds.rename_column(tcol_val, "text").rename_column(lcol_val, "label")

    for col in ("text", "label"):
        if col not in ds["train"].column_names:
            raise ValueError(f"[SOLD] Column '{col}' not found. Available: {ds['train'].column_names}")
    return ds

# ---------------------------
# Normalization
# ---------------------------

def normalize_text(t: str) -> str:
    try:
        s = str(t)
    except Exception:
        s = ""
    s = s.lower().replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    # collapse >2 repeats to two
    import re
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    # remove separators between letters: s.h.i.t -> shit
    s = re.sub(r"(?:[\W_])+(?=[A-Za-z])", "", s)
    if HAS_UNIDECODE:
        try:
            s = unidecode(s)
        except Exception:
            pass
    return s

def load_additional_bad_csv(path: str) -> List[str]:
    """
    Load comma-separated Sinhala bad words from a CSV file.
    Returns a list of normalized strings.
    """
    words: List[str] = []
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        return words
    try:
        import csv
        with open(path, "r", encoding="utf-8") as f:
            # Read entire file and split robustly on commas/newlines
            text = f.read()
        # Fallback simple split to handle irregular formatting
        import re
        tokens = [tok.strip() for tok in re.split(r"[,\n;]+", text) if tok.strip()]
        # Also try csv reader for quoted entries
        try:
            with open(path, "r", encoding="utf-8", newline="") as f2:
                reader = csv.reader(f2, delimiter=",")
                for row in reader:
                    for cell in row:
                        cell = cell.strip()
                        if cell:
                            tokens.append(cell)
        except Exception:
            pass
        # Deduplicate while preserving order
        seen = set()
        for w in tokens:
            wl = w.strip()
            if not wl:
                continue
            if wl not in seen:
                seen.add(wl)
                words.append(wl)
    except Exception as e:
        logging.warning("[extra_bad_csv] Failed to load %s: %s", path, e)
    return words

# ---------------------------
# Training (scikit-learn)
# ---------------------------

def train_eval_sklearn(X_train: List[str], y_train: List[int], X_val: List[str], y_val: List[int], output_dir: str, seed: int,
                       char_ngram_min: int = 2, char_ngram_max: int = 5, word_ngram_max: int = 2,
                       classifier: str = "linearsvc") -> Dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)

    # Vectorizers
    vec_char = TfidfVectorizer(analyzer="char", ngram_range=(char_ngram_min, char_ngram_max), preprocessor=normalize_text, min_df=1)
    vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1, word_ngram_max), preprocessor=normalize_text, min_df=1)

    Xc_train = vec_char.fit_transform(X_train)
    Xw_train = vec_word.fit_transform(X_train)
    X_train_mat = hstack([Xc_train, Xw_train])

    Xc_val = vec_char.transform(X_val)
    Xw_val = vec_word.transform(X_val)
    X_val_mat = hstack([Xc_val, Xw_val])

    class_weights = compute_class_weights_dict(y_train)

    if classifier == "logreg":
        clf = LogisticRegression(max_iter=2000, solver="saga", class_weight=class_weights, random_state=seed, n_jobs=os.cpu_count())
    else:
        clf = LinearSVC(class_weight=class_weights, random_state=seed)

    t0 = time.time()
    clf.fit(X_train_mat, y_train)
    t1 = time.time()

    preds = clf.predict(X_val_mat)
    acc = accuracy_score(y_val, preds)
    precision, recall, f1_bin, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
    _, _, f1_macro, _ = precision_recall_fscore_support(y_val, preds, average="macro", zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "precision_bin": float(precision),
        "recall_bin": float(recall),
        "f1_bin": float(f1_bin),
        "f1_macro": float(f1_macro),
        "train_seconds": round(t1 - t0, 2),
    }

    # Save pipeline components
    dump({"vec_char": vec_char, "vec_word": vec_word, "classifier": clf}, os.path.join(output_dir, "model.joblib"))
    with open(os.path.join(output_dir, "inference_card.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": "scikit-learn LinearSVC (or LogisticRegression) + TF-IDF char/word n-grams",
            "model_path": output_dir,
            "task": "badword-detection-binary",
            "text_column": "text",
            "label_column": "label",
            "char_ngrams": [char_ngram_min, char_ngram_max],
            "word_ngrams": [1, word_ngram_max],
            "notes": "Compact ML model for bad-word detection. No LLM/Transformers used.",
            "metrics": metrics,
        }, f, indent=2)

    return metrics

# ---------------------------
# Training (FastText optional)
# ---------------------------

def train_eval_fasttext(X_train: List[str], y_train: List[int], X_val: List[str], y_val: List[int], output_dir: str, epoch: int = 5) -> Dict[str, float]:
    try:
        import fasttext  # type: ignore
    except Exception as e:
        logging.warning("FastText not available (%s). Falling back to sklearn.", e)
        return {}

    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "fasttext_train.txt")
    val_file = os.path.join(output_dir, "fasttext_val.txt")

    def _fmt(texts, labels, path):
        with open(path, "w", encoding="utf-8") as f:
            for t, y in zip(texts, labels):
                f.write(f"__label__{int(y)} {normalize_text(str(t))}\n")

    _fmt(X_train, y_train, train_file)
    _fmt(X_val, y_val, val_file)

    model = fasttext.train_supervised(
        input=train_file,
        lr=0.5,
        epoch=max(1, int(epoch)),
        wordNgrams=2,
        minn=2,
        maxn=5,
        dim=100,
        loss="ova",
    )
    model.save_model(os.path.join(output_dir, "fasttext.bin"))

    # Evaluate
    y_pred = []
    for t in X_val:
        labels, _ = model.predict(normalize_text(str(t)))
        lab = labels[0] if labels else "__label__0"
        y_pred.append(1 if lab.endswith("1") else 0)

    acc = accuracy_score(y_val, y_pred)
    precision, recall, f1_bin, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
    _, _, f1_macro, _ = precision_recall_fscore_support(y_val, y_pred, average="macro", zero_division=0)
    metrics = {
        "accuracy": float(acc),
        "precision_bin": float(precision),
        "recall_bin": float(recall),
        "f1_bin": float(f1_bin),
        "f1_macro": float(f1_macro),
    }

    with open(os.path.join(output_dir, "inference_card.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": "FastText supervised",
            "model_path": output_dir,
            "task": "badword-detection-binary",
            "text_column": "text",
            "label_column": "label",
            "notes": "FastText model for bad-word detection.",
            "metrics": metrics,
        }, f, indent=2)

    return metrics

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Compact ML trainer for bad-word classification (no LLM).")
    parser.add_argument("--dataset_path", type=str, default="data/auto_badwords_en_si.csv", help="Path to CSV/JSON/JSONL with text+label columns.")
    parser.add_argument("--dataset_url", type=str, default=None, help="Optional URL to download dataset if dataset_path is missing.")
    # Auto wordlist options
    parser.add_argument("--auto_wordlists", action="store_true", default=True, help="Build dataset automatically from public word lists.")
    parser.add_argument("--bad_words_langs", type=str, default="en,si", help="Comma-separated languages for bad-word lists (e.g., 'en,si').")
    parser.add_argument("--bad_words_repo_url", type=str, default=DEFAULT_BADWORDS_REPO, help="Base repo URL for bad-word lists.")
    parser.add_argument("--si_wordlist_urls", type=str, default=os.environ.get("SI_WORDLIST_URLS", ""), help="Comma-separated custom URLs for Sinhala bad-word lists (unicode/singlish). Can also be a local file path containing one URL per line.")
    parser.add_argument("--clean_words_url", type=str, default=DEFAULT_CLEANWORDS_URL, help="URL for clean words list.")
    parser.add_argument("--wordlist_clean_limit", type=int, default=10000, help="Max number of clean words to use.")
    parser.add_argument("--augment_context", action="store_true", default=True, help="Wrap words into short sentences for better generalization.")
    parser.add_argument("--regenerate_wordlist", action="store_true", help="Rebuild dataset CSV from wordlists even if file exists.")
    # Training/data params
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column.")
    parser.add_argument("--label_column", type=str, default="label", help="Name of the label column (0/1).")
    parser.add_argument("--output_dir", type=str, default="outputs/badwords-ml", help="Where to save the trained model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--model_choice", type=str, choices=["sklearn", "fasttext"], default="sklearn", help="Choose simple ML model.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Used by FastText (ignored for sklearn).")
    # Extra Sinhala bad words CSV (comma-separated) to augment training
    parser.add_argument("--extra_bad_csv", type=str, default="bad.csv", help="Comma-separated Sinhala bad words CSV to augment training.")
    # SOLD dataset
    parser.add_argument("--use_sold", action="store_true", help="Use 'sinhala-nlp/SOLD' dataset from Hugging Face hub. If hub load fails, fallback to parquet or local TSV files.")
    parser.add_argument("--sold_parquet_train_api", type=str, default="https://huggingface.co/api/datasets/sinhala-nlp/SOLD/parquet/default/train", help="API endpoint to fetch parquet URLs for SOLD train split.")
    parser.add_argument("--sold_parquet_test_api", type=str, default="https://huggingface.co/api/datasets/sinhala-nlp/SOLD/parquet/default/test", help="API endpoint to fetch parquet URLs for SOLD test split.")
    parser.add_argument("--sold_train_path", type=str, default="data/SOLD_train.tsv", help="Local path to SOLD train TSV (fallback).")
    parser.add_argument("--sold_test_path", type=str, default="data/SOLD_test.tsv", help="Local path to SOLD test TSV as validation (fallback).")
    parser.add_argument("--sold_trial_path", type=str, default="data/SOLD_trial.tsv", help="Local path to SOLD trial TSV (optional).")
    # Fresh start / cache clearing
    parser.add_argument("--fresh_start", action="store_true", help="Delete output_dir before training.")
    # Optional setup
    parser.add_argument("--setup_deps", action="store_true", help="Install missing Python packages at runtime (useful in Colab).")
    args = parser.parse_args()

    # Prepare output/logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Starting compact ML training for bad-words classifier")
    logging.info(f"Args: {vars(args)}")

    # Optional dependency setup
    if args.setup_deps or in_colab():
        try:
            ensure_dependencies()
        except Exception as e:
            logging.warning("Dependency setup encountered an issue: %s", e)

    # Fresh start
    if args.fresh_start and os.path.isdir(args.output_dir):
        logging.info("[fresh_start] Removing %s", args.output_dir)
        shutil.rmtree(args.output_dir, ignore_errors=True)
        os.makedirs(args.output_dir, exist_ok=True)

    # Auto wordlist dataset
    dataset_path = args.dataset_path
    if not args.use_sold:
        if args.auto_wordlists and (args.regenerate_wordlist or not os.path.exists(dataset_path)):
            langs = [s.strip() for s in args.bad_words_langs.split(",") if s.strip()]
            si_urls_list = None
            raw_si = (args.si_wordlist_urls or "").strip()
            if raw_si:
                if os.path.exists(raw_si) and os.path.isfile(raw_si):
                    with open(raw_si, "r", encoding="utf-8") as f:
                        si_urls_list = [ln.strip() for ln in f if ln.strip()]
                else:
                    si_urls_list = [s.strip() for s in raw_si.split(",") if s.strip()]
            if any(l.lower() == "si" for l in langs) and not si_urls_list:
                logging.info("[wordlist] Sinhala requested but no --si_wordlist_urls provided. Switching to SOLD dataset (hub/parquet/local TSV).")
                args.use_sold = True
            if not args.use_sold:
                bad_words = download_bad_words(langs, repo_base=args.bad_words_repo_url, si_overrides=si_urls_list)
                clean_words = download_clean_words(url=args.clean_words_url, limit=args.wordlist_clean_limit)
                if not bad_words:
                    logging.warning("[wordlist] No bad words downloaded; training may not be meaningful.")
                if not clean_words:
                    logging.warning("[wordlist] No clean words downloaded; training may not be meaningful.")
                build_wordlist_dataset_csv(dataset_path, bad_words, clean_words, augment_context=args.augment_context)
                _summarize_downloads()
        if not args.use_sold:
            dataset_path = maybe_download_dataset(dataset_path, args.dataset_url)

    # Dataset
    if args.use_sold:
        ds = build_sold_dataset(
            seed=args.seed,
            val_size=args.val_size,
            train_path=args.sold_train_path,
            test_path=args.sold_test_path,
            trial_path=args.sold_trial_path,
            parquet_train_api=args.sold_parquet_train_api,
            parquet_test_api=args.sold_parquet_test_api,
        )
        args.text_column = "text"
        args.label_column = "label"
        logging.info("[SOLD] Using columns text='%s', label='%s'", args.text_column, args.label_column)
    else:
        ds = build_dataset(dataset_path, args.text_column, args.label_column, seed=args.seed, val_size=args.val_size)

    ds = ensure_label_ints(ds, args.label_column)

    # Log dataset stats
    try:
        import collections
        cnt_train = collections.Counter(ds["train"][args.label_column])
        cnt_val = collections.Counter(ds["validation"][args.label_column])
        logging.info(f"Dataset sizes: train={len(ds['train'])}, validation={len(ds['validation'])}")
        logging.info(f"Label distribution (train): {dict(cnt_train)}")
        logging.info(f"Label distribution (validation): {dict(cnt_val)}")
    except Exception as e:
        logging.warning(f"Failed to log dataset stats: {e}")

    # Convert to lists
    X_train_raw = [str(x) if x is not None else "" for x in ds["train"][args.text_column]]
    y_train = [int(x) for x in ds["train"][args.label_column]]
    X_val_raw = [str(x) if x is not None else "" for x in ds["validation"][args.text_column]]
    y_val = [int(x) for x in ds["validation"][args.label_column]]

    # Augment training with additional Sinhala bad words from CSV if available
    extra_bad_path = args.extra_bad_csv
    def _make_contexts(w: str) -> List[str]:
        if args.augment_context:
            return [w, f"You are {w}.", f"That was {w}!", f"{w} behavior."]
        return [w]
    try:
        extra_words = []
        if extra_bad_path and os.path.exists(extra_bad_path) and os.path.isfile(extra_bad_path):
            from unicodedata import normalize as _normalize
            extra_words = [ _normalize("NFKC", w) for w in load_additional_bad_csv(extra_bad_path) if w ]
            # Deduplicate
            seen = set()
            extra_words = [w for w in extra_words if not (w in seen or seen.add(w))]
        if extra_words:
            added = 0
            for w in extra_words:
                for t in _make_contexts(w):
                    X_train_raw.append(t)
                    y_train.append(1)
                    added += 1
            logging.info("[extra_bad_csv] Augmented training with %d examples from %d Sinhala bad words (%s).", added, len(extra_words), extra_bad_path)
        else:
            logging.info("[extra_bad_csv] No extra bad words found at %s (skipping).", extra_bad_path)
    except Exception as e:
        logging.warning("[extra_bad_csv] Failed to augment training: %s", e)

    logging.info("=" * 70)
    logging.info("DATASET SUMMARY BEFORE TRAINING")
    if args.use_sold:
        logging.info("[SOLD] Source: HuggingFace hub 'sinhala-nlp/SOLD'")
    else:
        logging.info("[DATA] Source: local file '%s'%s",
                     dataset_path,
                     " (downloaded from URL)" if args.dataset_url else "")
    logging.info("[DATA] Train size: %d, Validation size: %d", len(X_train_raw), len(X_val_raw))
    logging.info("=" * 70)

    # Train
    metrics = {}
    if args.model_choice == "fasttext":
        logging.info("Training FastText model...")
        metrics = train_eval_fasttext(X_train_raw, y_train, X_val_raw, y_val, args.output_dir, epoch=args.num_train_epochs)
        if not metrics:
            logging.info("Falling back to sklearn due to FastText unavailability.")
            args.model_choice = "sklearn"

    if args.model_choice == "sklearn":
        logging.info("Training scikit-learn model (TF-IDF + LinearSVC)...")
        metrics = train_eval_sklearn(X_train_raw, y_train, X_val_raw, y_val, args.output_dir, seed=args.seed)

    logging.info("Training finished. Metrics: %s", metrics)
    print(f"Training complete. Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
