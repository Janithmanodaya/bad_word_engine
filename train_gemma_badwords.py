#!/usr/bin/env python3
"""
Self-bootstrapping QLoRA trainer for bad-word classification on Google Gemma.

Key features:
- One-file script: auto-installs required Python libraries if missing.
- GPU-first: 4-bit quantization (bitsandbytes) + LoRA for low VRAM training on a gaming PC.
- CPU fallback: allowed only with --force_cpu (may OOM for Gemma; not recommended).
- Handles CSV/JSON/JSONL; can also download dataset from a URL (--dataset_url).
- Can auto-build a dataset from public bad-word lists (--auto_wordlists).
- Class imbalance aware (weighted cross-entropy).
- Saves a tiny LoRA adapter usable on very small servers for inference.
- Detailed logging to console and file for progress tracking.

Quick usage (no manual installs needed):
    python train_gemma_badwords.py \\
      --dataset_path data/badwords.csv \\
      --text_column text --label_column label \\
      --base_model google/gemma-2-2b-it \\
      --output_dir outputs/gemma-badwords-qlora \\
      --preset low_vram

Notes:
- Train on your gaming PC (with NVIDIA GPU). Deploy the small adapter on the tiny server.
- Use binary labels: 1 = bad/offensive, 0 = clean.
"""

import argparse
import json
import os
import sys
import subprocess
import importlib.util
import urllib.request
from typing import Optional, List
import shutil
import time
import logging
from collections import defaultdict
import threading

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
# Bootstrap: ensure dependencies installed
# ---------------------------

REQUIRED_PKGS = [
    "numpy",
    # Transformers stack (small classifier only)
    "transformers>=4.43",
    "datasets",
    "accelerate",
    "evaluate",
    "scikit-learn",
]

def _pip_install(args: list) -> None:
    logging.getLogger("setup").info(f"Installing: {' '.join(args)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + args)

def _is_installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None

def _install_torch_best_effort():
    # If torch is installed, nothing to do
    if _is_installed("torch"):
        return
    try:
        # Prefer CUDA wheel if NVIDIA driver present
        has_nvidia = shutil.which("nvidia-smi") is not None
        if has_nvidia:
            # Default to CUDA 12.1 wheels which work on recent GPUs
            _pip_install(["torch", "--index-url", "https://download.pytorch.org/whl/cu121"])
        else:
            _pip_install(["torch"])
    except subprocess.CalledProcessError as e:
        print(f"[setup] WARNING: torch install failed: {e}. Trying CPU wheel.")
        try:
            _pip_install(["torch"])
        except Exception as e2:
            print(f"[setup] WARNING: torch CPU install also failed: {e2}. Continuing; training may not work.")

def ensure_dependencies():
    # Prefer not to disturb Colab's preinstalled CUDA/torch stack
    if not in_colab():
        _install_torch_best_effort()

    missing = []
    # Map import names for some pkgs
    import_name_map = {
        "transformers>=4.43": "transformers",
        "scikit-learn": "sklearn",
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

    # Optional: install git-lfs on Colab to accelerate model downloads
    if in_colab():
        try:
            if shutil.which("git-lfs") is None:
                subprocess.check_call(["apt-get", "update", "-y"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.check_call(["apt-get", "install", "-y", "git-lfs"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[setup] WARNING: git-lfs install failed or unnecessary: {e}")

    # Final check logging
    for spec in REQUIRED_PKGS + ["torch"]:
        name = spec.split("==")[0].split(">=")[0]
        print(f"[setup] {name}: {'OK' if _is_installed(name) else 'MISSING'}")

ensure_dependencies()

# Now safe to import heavy libraries
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
try:
    from sklearn.metrics import precision_recall_fscore_support  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
import evaluate
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback,
)
from transformers.utils import logging as hf_logging
from transformers.trainer_utils import get_last_checkpoint
from huggingface_hub import login as hf_login
from huggingface_hub.errors import GatedRepoError

# ---------------------------
# Torch perf knobs (use max available resources)
# ---------------------------
try:
    import torch.backends.cudnn as cudnn  # type: ignore
    cudnn.benchmark = True
except Exception:
    pass
try:
    torch.backends.cuda.matmul.allow_tf32 = True  # Ampere+ TF32 fast matmul
except Exception:
    pass
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

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
    """
    Download bad words for the given language codes.
    - For 'en': use LDNOOBW (try both with/without .txt).
    - For 'si': STRICT mode — only use URLsirst, then multiple mirrors; if all fail, fall back to a built-in minimal list.
    """
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
            # STRICT: require explicit Sinhala URLs; do not use mirrors or built-ins
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

        # English and other languages via LDNOOBW
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
        # Bad words = 1
        for w in bad_words:
            if augment_context:
                # Simple contexts to help model generalize beyond single-token inputs
                examples = [w, f"You are {w}.", f"That was {w}!", f"{w} behavior."]
                for t in examples:
                    writer.writerow([t, 1])
            else:
                writer.writerow([w, 1])
        # Clean words = 0
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

def _bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 0)  # Ampere+

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
    # Convert labels to ints (0/1) if they come as strings like "0"/"1" or "clean"/"bad"
    unique = set(dataset["train"][label_column])
    mapping = {}
    if all(isinstance(x, (int, np.integer)) for x in unique):
        return dataset  # already ints

    # Try common mappings
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
        # Build alphabetical mapping deterministically
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
                # Fallback: unseen label defaults to 0
                example[label_column] = 0
        return example

    return dataset.map(map_labels)

def compute_class_weights(labels: List[int]) -> torch.Tensor:
    # Weighted cross-entropy to mitigate class imbalance
    labels_arr = np.array(labels)
    n_classes = int(labels_arr.max()) + 1
    counts = np.bincount(labels_arr, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (counts * n_classes)
    return torch.tensor(weights, dtype=torch.float32)

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
# Trainer with weighted loss
# ---------------------------

class WeightedTrainer(Trainer):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    # Accept **kwargs to be compatible with Transformers v5 which passes
    # num_items_in_batch into compute_loss.
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        # Flatten in case of batched inputs
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ---------------------------
# Callback to delay evaluation
# ---------------------------

class DelayedEvalCallback(TrainerCallback):
    def __init__(self, start_step: int = 0, start_epoch: float = 0.0):
        self.start_step = int(start_step or 0)
        self.start_epoch = float(start_epoch or 0.0)

    def on_step_end(self, args, state, control, **kwargs):
        try:
            if state.global_step is not None and state.global_step < self.start_step:
                control.should_evaluate = False
        except Exception:
            pass
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        try:
            if state.epoch is not None and state.epoch < self.start_epoch:
                control.should_evaluate = False
        except Exception:
            pass
        return control

# ---------------------------
# Data module
# ---------------------------

def build_dataset(path: str, text_col: str, label_col: str, seed: int, val_size: float) -> DatasetDict:
    ftype = infer_file_type(path)
    if ftype == "csv":
        raw = load_dataset("csv", data_files={"train": path})
    else:
        raw = load_dataset("json", data_files={"train": path}, split=None)

    if isinstance(raw, dict):
        ds = DatasetDict(raw)
    else:
        ds = DatasetDict({"train": raw})

    # If no explicit validation, make a split
    if "validation" not in ds:
        # Use stratification only if the label column is a ClassLabel feature
        stratify = None
        try:
            feat = ds["train"].features.get(label_col, None)
            # Avoid importing ClassLabel directly to keep dependencies light; check by class name
            if feat is not None and feat.__class__.__name__ == "ClassLabel":
                stratify = label_col
        except Exception:
            stratify = None

        if stratify:
            split = ds["train"].train_test_split(
                test_size=val_size,
                seed=seed,
                stratify_by_column=stratify,
            )
        else:
            split = ds["train"].train_test_split(
                test_size=val_size,
                seed=seed,
            )
        ds = DatasetDict({"train": split["train"], "validation": split["test"]})

    # Ensure required columns exist
    for col in (text_col, label_col):
        if col not in ds["train"].column_names:
            raise ValueError(f"Column '{col}' not found in dataset. Available: {ds['train'].column_names}")

    return ds


def build_sold_dataset(seed: int, val_size: float) -> DatasetDict:
    """
    Load SOLD (Sinhala Offensive Language Dataset) from Hugging Face and return a DatasetDict
    with 'train' and 'validation' splits. Columns used: text (features) and label (targets).
    """
    logging.info("[SOLD] Loading 'sinhala-nlp/SOLD' from Hugging Face hub")
    raw = load_dataset("sinhala-nlp/SOLD")
    # Prefer official train/test; fall back to creating a split if necessary
    if isinstance(raw, dict):
        if "train" in raw and "test" in raw:
            ds = DatasetDict({"train": raw["train"], "validation": raw["test"]})
        elif "train" in raw and "validation" in raw:
            ds = DatasetDict({"train": raw["train"], "validation": raw["validation"]})
        else:
            # Single split present; create validation
            base = list(raw.values())[0]
            split = base.train_test_split(test_size=val_size, seed=seed)
            ds = DatasetDict({"train": split["train"], "validation": split["test"]})
    else:
        # Unexpected structure; make a split
        split = raw.train_test_split(test_size=val_size, seed=seed)
        ds = DatasetDict({"train": split["train"], "validation": split["test"]})

    # Confirm columns exist
    for col in ("text", "label"):
        if col not in ds["train"].column_names:
            raise ValueError(f"[SOLD] Column '{col}' not found. Available: {ds['train'].column_names}")
    logging.info("[SOLD] Dataset loaded. train=%d, validation=%d", len(ds["train"]), len(ds["validation"]))
    return ds

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Self-installing trainer for compact bad-word classifier (Multilingual MiniLM).")
    parser.add_argument("--dataset_path", type=str, default="data/auto_badwords_en_si.csv", help="Path to CSV/JSON/JSONL with text+label columns.")
    parser.add_argument("--dataset_url", type=str, default=None, help="Optional URL to download dataset if dataset_path is missing.")
    # Auto wordlist options
    parser.add_argument("--auto_wordlists", action="store_true", default=True, help="Build dataset automatically from public word lists.")
    parser.add_argument("--bad_words_langs", type=str, default="en,si", help="Comma-separated languages for bad-word lists (e.g., 'en,si').")
    parser.add_argument("--bad_words_repo_url", type=str, default=DEFAULT_BADWORDS_REPO, help="Base repo URL for bad-word lists.")
    parser.add_argument("--si_wordlist_urls", type=str, default="", help="Comma-separated custom URLs for Sinhala bad-word lists (unicode/singlish).")
    parser.add_argument("--clean_words_url", type=str, default=DEFAULT_CLEANWORDS_URL, help="URL for clean words list.")
    parser.add_argument("--wordlist_clean_limit", type=int, default=10000, help="Max number of clean words to use.")
    parser.add_argument("--augment_context", action="store_true", default=True, help="Wrap words into short sentences for better generalization.")
    parser.add_argument("--regenerate_wordlist", action="store_true", help="Rebuild dataset CSV from wordlists even if file exists.")
    # Training/data params
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column.")
    parser.add_argument("--label_column", type=str, default="label", help="Name of the label column (0/1).")
    parser.add_argument("--base_model", type=str, default="microsoft/Multilingual-MiniLM-L12-H384", help="Small classifier base model to fine-tune.")
    parser.add_argument("--output_dir", type=str, default="outputs/gemma-badwords-qlora", help="Where to save LoRA adapter and tokenizer.")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.1)
    
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory.")
    parser.add_argument("--preset", type=str, choices=["small_cls"], default="small_cls", help="Use compact classifier model (no LoRA/4-bit).")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU mode (discouraged; Gemma likely OOM on CPU).")
    # GPU preference/validation
    parser.add_argument("--require_gpu_name", type=str, default="", help="If set, require that CUDA device name contains this substring (e.g., 'T4').")
    parser.add_argument("--enforce_gpu_name", action="store_true", help="If true and require_gpu_name set, exit if the CUDA device name doesn't match.")
    # Fresh start / cache clearing
    parser.add_argument("--fresh_start", action="store_true", help="Delete output_dir and model/dataset caches before starting.")
    # Resume options
    parser.add_argument("--resume", action="store_true", help="Auto-resume from the last checkpoint found in output_dir if available.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume from. If set, overrides --resume.")
    # Evaluation delay options
    parser.add_argument("--eval_start_step", type=int, default=0, help="Skip evaluation until this global step is reached.")
    parser.add_argument("--eval_start_epoch", type=float, default=0.0, help="Skip evaluation until this epoch number is reached.")
    # Colab keepalive options
    parser.add_argument("--colab_keepalive", action="store_true", help="Enable a lightweight keepalive thread to reduce Colab idle disconnects.")
    parser.add_argument("--colab_keepalive_interval", type=int, default=60, help="Keepalive ping interval in seconds (default: 60).")
    # Use SOLD dataset
    parser.add_argument("--use_sold", action="store_true", help="Use 'sinhala-nlp/SOLD' dataset from Hugging Face hub instead of local file/auto wordlists.")
    # Auth and fallback
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"), help="Hugging Face token for gated models.")
    parser.add_argument("--fallback_model", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="Fallback open model if base model is gated and no token.")
    args = parser.parse_args()

    # Prepare output/logging
    os.makedirs(args.output_dir, exist_ok=True)

    # Colab-specific cache optimization
    if in_colab():
        os.environ.setdefault("HF_HOME", "/content/cache/hf")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/content/cache/transformers")
        os.environ.setdefault("DATASETS_CACHE", "/content/cache/datasets")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # Fresh start: remove output and caches
    if args.fresh_start:
        print("[fresh_start] Deleting previous outputs and caches...")
        paths_to_clear = [args.output_dir]
        for env_key in ("HF_HOME", "TRANSFORMERS_CACHE", "DATASETS_CACHE"):
            p = os.environ.get(env_key, "")
            if p:
                paths_to_clear.append(p)
        # In user envs, also consider default HF cache if env not set
        default_hf_cache = os.path.expanduser("~/.cache/huggingface")
        default_ds_cache = os.path.expanduser("~/.cache/huggingface/datasets")
        default_tf_cache = os.path.expanduser("~/.cache/huggingface/transformers")
        for p in [default_hf_cache, default_ds_cache, default_tf_cache]:
            if p:
                paths_to_clear.append(p)
        for p in paths_to_clear:
            try:
                if p and os.path.exists(p):
                    print(f"[fresh_start] Removing {p}")
                    import shutil as _shutil
                    _shutil.rmtree(p, ignore_errors=True)
            except Exception as e:
                print(f"[fresh_start] Warning: failed to remove {p}: {e}")
        # Force rebuild of auto wordlist dataset if used
        args.regenerate_wordlist = True

    # Ensure directories exist after optional clearing
    if in_colab():
        os.makedirs(os.environ["HF_HOME"], exist_ok=True)
        os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
        os.makedirs(os.environ["DATASETS_CACHE"], exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    # HF transformers logging
    hf_logging.set_verbosity_info()
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()
    logging.info("Starting bad-words classifier training")
    logging.info(f"Args: {vars(args)}")

    # Small classifier mode only (no LoRA/4-bit)
    small_model = True

    # ---------------------------
    # Colab keepalive thread
    # ---------------------------
    def _colab_keepalive_loop(interval: int):
        url = "https://clients3.google.com/generate_204"  # 204 response, light ping
        while True:
            try:
                # Small network ping
                with urllib.request.urlopen(url, timeout=10) as _r:
                    pass
                # Emit an invisible heartbeat to stdout to keep I/O active
                sys.stdout.write("\u200B")
                sys.stdout.flush()
            except Exception:
                # Avoid crashing the thread on transient failures
                pass
            time.sleep(max(10, int(interval)))

    if in_colab() and args.colab_keepalive:
        try:
            t = threading.Thread(target=_colab_keepalive_loop, args=(args.colab_keepalive_interval,), daemon=True)
            t.start()
            logging.info("[colab] Keepalive thread started (interval=%ss).", args.colab_keepalive_interval)
        except Exception as e:
            logging.warning("[colab] Failed to start keepalive thread: %s", e)

    # Apply preset for small classifier (no LoRA/4-bit quantization)
    # Good for multilingual toxicity. Defaults chosen for speed and stability.
    if args.base_model == "google/gemma-2-2b-it":
        args.base_model = "microsoft/Multilingual-MiniLM-L12-H384"
    args.per_device_train_batch_size = max(8, int(args.per_device_train_batch_size or 8))
    args.per_device_eval_batch_size = max(32, int(args.per_device_eval_batch_size or 32))
    args.max_length = min(args.max_length, 64)
    args.gradient_accumulation_steps = max(1, int(args.gradient_accumulation_steps or 1))
    args.gradient_checkpointing = False
    args.logging_steps = max(args.logging_steps, 20)
    args.eval_steps = max(args.eval_steps, 200)
    args.save_steps = max(args.save_steps, 1000)
    args.learning_rate = min(args.learning_rate, 5e-4)

    # Build dataset from wordlists if requested (skip when using SOLD)
    dataset_path = args.dataset_path
    if not args.use_sold:
        if args.auto_wordlists and (args.regenerate_wordlist or not os.path.exists(dataset_path)):
            langs = [s.strip() for s in args.bad_words_langs.split(",") if s.strip()]
            si_urls = [s.strip() for s in (args.si_wordlist_urls or "").split(",") if s.strip()] or None
            # STRICT: if Sinhala requested, URLs must be provided
            if any(l.lower() == "si" for l in langs) and not si_urls:
                raise SystemExit("Sinhala language requested in --bad_words_langs, but --si_wordlist_urls was not provided. Please supply exact Sinhala wordlist URLs.")
            bad_words = download_bad_words(langs, repo_base=args.bad_words_repo_url, si_overrides=si_urls)
            clean_words = download_clean_words(url=args.clean_words_url, limit=args.wordlist_clean_limit)
            if not bad_words:
                logging.warning("[wordlist] No bad words downloaded; training may not be meaningful.")
            if not clean_words:
                logging.warning("[wordlist] No clean words downloaded; training may not be meaningful.")
            build_wordlist_dataset_csv(dataset_path, bad_words, clean_words, augment_context=args.augment_context)
            # Summarize download results for clear verification in logs
            _summarize_downloads()

        # Dataset presence or download otherwise
        dataset_path = maybe_download_dataset(dataset_path, args.dataset_url)

    # Environment checks
    has_cuda = torch.cuda.is_available()
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {has_cuda}")
    if has_cuda:
        try:
            ndev = torch.cuda.device_count()
            torch.cuda.set_device(0)
            dev_name = torch.cuda.get_device_name(0)
            logging.info(f"CUDA device count: {ndev}")
            logging.info(f"Using CUDA device 0: {dev_name}")
            # Enforce/validate GPU name if requested
            if args.require_gpu_name:
                if args.require_gpu_name.lower() not in dev_name.lower():
                    msg = f"Detected GPU '{dev_name}' does not match required pattern '{args.require_gpu_name}'."
                    if args.enforce_gpu_name:
                        raise SystemExit(msg + " Exiting due to --enforce_gpu_name.")
                    else:
                        logging.warning(msg + " Continuing anyway.")
            else:
                # In Colab, hint if not a T4
                if in_colab() and "t4" not in dev_name.lower():
                    logging.warning("Colab GPU is '%s', not T4. If you require T4, set --require_gpu_name T4.", dev_name)
        except Exception as e:
            logging.warning(f"Failed to finalize CUDA device selection/logging: {e}")
    if not has_cuda and not args.force_cpu:
        if in_colab():
            raise SystemExit(
                "No CUDA GPU detected. In Google Colab, enable GPU via: Runtime > Change runtime type > Hardware accelerator > GPU.\n"
                "If you still want to attempt CPU training (not recommended; may OOM), pass --force_cpu."
            )
        raise SystemExit(
            "No CUDA GPU detected. A GPU is strongly recommended for training speed. "
            "If you still want to attempt CPU training (may be slow), pass --force_cpu."
        )

    set_seed(args.seed)

    # Compute dtype
    compute_dtype = torch.bfloat16 if _bf16_supported() else torch.float16

    # Optional HF login for gated models
    if args.hf_token:
        try:
            hf_login(token=args.hf_token)
            logging.info("Hugging Face login successful via token.")
        except Exception as e:
            logging.warning(f"HF login failed: {e}")

    # Tokenizer with gated fallback
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        active_model_id = args.base_model
    except Exception as e:
        # Some environments raise OSError instead of GatedRepoError for gated repos
        msg = str(e).lower()
        if isinstance(e, GatedRepoError) or "gated repo" in msg or "401" in msg or "unauthorized" in msg:
            logging.warning(f"Base model not accessible ({e}). Falling back to {args.fallback_model}")
            tokenizer = AutoTokenizer.from_pretrained(args.fallback_model, use_fast=True)
            active_model_id = args.fallback_model
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    if args.use_sold:
        ds = build_sold_dataset(seed=args.seed, val_size=args.val_size)
        # SOLD uses text/label columns by definition
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

    def tokenize_fn(batch):
        # Normalize inputs for tokenizer: ensure list of strings
        texts = batch.get(args.text_column)
        if not isinstance(texts, list):
            texts = [texts]
        safe_texts = []
        for t in texts:
            if t is None:
                safe_texts.append("")
            elif isinstance(t, str):
                safe_texts.append(t)
            else:
                try:
                    safe_texts.append(str(t))
                except Exception:
                    safe_texts.append("")
        toks = tokenizer(safe_texts, max_length=args.max_length, truncation=True)
        toks["labels"] = batch[args.label_column]
        return toks

    ds_tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in ds["train"].column_names if c not in [args.text_column, args.label_column]],
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # Labels/config
    try:
        train_max = int(max(ds["train"][args.label_column]))
        val_max = int(max(ds["validation"][args.label_column]))
        num_labels = max(train_max, val_max) + 1
    except Exception:
        # Fallback in case of empty dataset or unexpected types
        num_labels = 2
    config = AutoConfig.from_pretrained(active_model_id, num_labels=num_labels)

    # Model loading (small classifier only)
    model_kwargs = {"config": config}
    # Use float32 weights for stable training and to avoid FP16 scaler issues
    if has_cuda:
        model_kwargs.update(dict(torch_dtype=torch.float32, device_map="auto"))
    else:
        model_kwargs.update(dict(torch_dtype=torch.float32))

    logging.info(f"Loading base model: {active_model_id}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(active_model_id, **model_kwargs)
    except Exception as e:
        msg = str(e).lower()
        is_gated = isinstance(e, GatedRepoError) or "gated repo" in msg or "401" in msg or "unauthorized" in msg
        if is_gated and active_model_id != args.fallback_model:
            logging.warning(f"Model not accessible ({e}). Falling back to {args.fallback_model}")
            active_model_id = args.fallback_model
            config = AutoConfig.from_pretrained(active_model_id, num_labels=num_labels)
            model_kwargs["config"] = config
            model = AutoModelForSequenceClassification.from_pretrained(active_model_id, **model_kwargs)
        else:
            raise
    logging.info(f"Model loaded. Dtype: {getattr(model, 'dtype', 'mixed')}")
    logging.info("Small classifier active: training full model head without LoRA or 4-bit quantization.")

    # Class weights from training labels
    class_weights = compute_class_weights(list(ds["train"][args.label_column]))

    # Metrics
    accuracy = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
        f1_macro = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        if SKLEARN_AVAILABLE:
            from sklearn.metrics import precision_recall_fscore_support  # local import
            precision, recall, f1_bin, _ = precision_recall_fscore_support(
                labels, preds, average="binary", zero_division=0
            )
        else:
            precision = precision_metric.compute(predictions=preds, references=labels, average="binary")["precision"]
            recall = recall_metric.compute(predictions=preds, references=labels, average="binary")["recall"]
            f1_bin = f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"]
        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "precision_bin": precision,
            "recall_bin": recall,
            "f1_bin": f1_bin,
        }

    # Training args
    # Robustly detect which argument name TrainingArguments supports by introspection.
    import inspect
    try:
        _params = inspect.signature(TrainingArguments.__init__).parameters
        if "eval_strategy" in _params:
            eval_key = "eval_strategy"  # transformers v5+
        elif "evaluation_strategy" in _params:
            eval_key = "evaluation_strategy"  # transformers v4
        else:
            eval_key = None
    except Exception:
        eval_key = None

    training_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1_bin",
        greater_is_better=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        # Max resource utilization for dataloading
        dataloader_num_workers=max(1, (os.cpu_count() or 1) // 2),
        dataloader_pin_memory=True,
        auto_find_batch_size=True,
        group_by_length=True,
        # Avoid gradient clipping interaction with FP16 scaler
        max_grad_norm=0.0,
    )
    if eval_key:
        training_kwargs[eval_key] = "steps"
    training_kwargs["eval_steps"] = args.eval_steps

    # Explicitly disable gradient checkpointing for small classifier
    training_kwargs["gradient_checkpointing"] = False

    if has_cuda:
        training_kwargs.update(
            dict(
                # Prefer bf16 on Ampere+, otherwise disable fp16 to avoid GradScaler issues
                bf16=(compute_dtype == torch.bfloat16),
                fp16=False,
            )
        )

    # Prefer fused AdamW if available for speed
    try:
        _params = inspect.signature(TrainingArguments.__init__).parameters
        if "optim" in _params:
            training_kwargs["optim"] = "adamw_torch"
            logging.info("Using optim='adamw_torch' for potential fused speedups.")
    except Exception:
        pass

    train_args = TrainingArguments(**training_kwargs)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=ds_tokenized["train"],
        eval_dataset=ds_tokenized["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # Register evaluation delay callback if requested
    if (args.eval_start_step and args.eval_start_step > 0) or (args.eval_start_epoch and args.eval_start_epoch > 0.0):
        trainer.add_callback(DelayedEvalCallback(start_step=args.eval_start_step, start_epoch=args.eval_start_epoch))
        logging.info("[eval] Evaluation will be skipped until step >= %s and epoch >= %s", args.eval_start_step, args.eval_start_epoch)


    # Train
    logging.info("=" * 70)
    logging.info("DATASET SUMMARY BEFORE TRAINING")
    if args.use_sold:
        logging.info("[SOLD] Source: HuggingFace hub 'sinhala-nlp/SOLD' (downloaded via datasets)")
    else:
        logging.info("[DATA] Source: local file '%s'%s",
                     dataset_path,
                     " (downloaded from URL)" if args.dataset_url else "")
    logging.info("[DATA] Train size: %d, Validation size: %d", len(ds["train"]), len(ds["validation"]))
    try:
        import collections as _collections
        _cnt_train = _collections.Counter(ds["train"][args.label_column])
        _cnt_val = _collections.Counter(ds["validation"][args.label_column])
        logging.info("[DATA] Label distribution (train): %s", dict(_cnt_train))
        logging.info("[DATA] Label distribution (validation): %s", dict(_cnt_val))
    except Exception as _e:
        logging.warning("[DATA] Failed to compute label distribution: %s", _e)
    logging.info("=" * 70)

    logging.info("Beginning training...")
    # Determine resume path if requested
    resume_arg = None
    if args.resume_from_checkpoint:
        resume_arg = args.resume_from_checkpoint
        logging.info("[resume] Explicit checkpoint path provided: %s", resume_arg)
    elif args.resume and not args.fresh_start:
        try:
            if os.path.isdir(args.output_dir):
                last_ckpt = get_last_checkpoint(args.output_dir)
                if last_ckpt:
                    resume_arg = last_ckpt
                    logging.info("[resume] Found last checkpoint in output_dir: %s", resume_arg)
                else:
                    logging.info("[resume] No checkpoint found in output_dir: %s", args.output_dir)
        except Exception as e:
            logging.warning("[resume] Failed to detect last checkpoint: %s", e)

    t0 = time.time()
    if resume_arg:
        train_result = trainer.train(resume_from_checkpoint=resume_arg)
    else:
        train_result = trainer.train()
    t1 = time.time()
    logging.info(f"Training finished. Seconds: {round(t1 - t0, 2)}")
    try:
        metrics = train_result.metrics if hasattr(train_result, "metrics") else {}
        logging.info(f"Train metrics: {metrics}")
    except Exception:
        pass

    # Save model and tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info("Saving model and tokenizer...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Export minimal inference card
    card = {
        "base_model": active_model_id,
        "model_path": args.output_dir,
        "task": "sequence-classification",
        "labels": list(range(num_labels)),
        "text_column": args.text_column,
        "label_column": args.label_column,
        "max_length": args.max_length,
        "notes": "This is a compact classifier fine-tuned for bad-word detection (no LoRA/quant).",
        "train_seconds": round(t1 - t0, 2),
        "preset": args.preset,
    }
    with open(os.path.join(args.output_dir, "inference_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    print(f"Training complete in {round(t1 - t0, 2)}s. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
