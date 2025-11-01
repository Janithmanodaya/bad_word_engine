#!/usr/bin/env python3
"""
Self-bootstrapping QLoRA trainer for bad-word classification on Google Gemma.

Key features:
- One-file script: auto-installs required Python libraries if missing.
- GPU-first: 4-bit quantization (bitsandbytes) + LoRA for low VRAM training on a gaming PC.
- CPU fallback: allowed only with --force_cpu (may OOM for Gemma; not recommended).
- Handles CSV/JSON/JSONL; can also download dataset from a URL (--dataset_url).
- Class imbalance aware (weighted cross-entropy).
- Saves a tiny LoRA adapter usable on very small servers for inference.

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

# ---------------------------
# Bootstrap: ensure dependencies installed
# ---------------------------

REQUIRED_PKGS = [
    "numpy",
    # Transformers/PEFT stack
    "transformers>=4.43",
    "peft",
    "bitsandbytes",
    "datasets",
    "accelerate",
    "evaluate",
    "scikit-learn",
    # torch: installing CPU-only by default if missing. For CUDA, user can pre-install.
    "torch",
]

def _pip_install(spec: str) -> None:
    print(f"[setup] Installing: {spec}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", spec])

def _is_installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None

def ensure_dependencies():
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
        print("[setup] Missing dependencies detected. Attempting installation...")
        # Special handling: install torch first to satisfy others (avoid bnb building wheels before torch)
        torch_specs = [s for s in missing if s.startswith("torch")]
        other_specs = [s for s in missing if not s.startswith("torch")]
        for spec in torch_specs + other_specs:
            try:
                _pip_install(spec)
            except subprocess.CalledProcessError as e:
                print(f"[setup] WARNING: Failed to install {spec}: {e}")
                # Continue; some environments may still work if the package is optional

    # Final check
    for spec in REQUIRED_PKGS:
        mod_name = import_name_map.get(spec, spec.split("==")[0].split(">=")[0])
        if not _is_installed(mod_name):
            print(f"[setup] WARNING: {spec} still not importable. The script may fail if this is required.")

ensure_dependencies()

# Now safe to import heavy libraries
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from sklearn.metrics import precision_recall_fscore_support
import evaluate
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

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
    else:
        # Build alphabetical mapping deterministically
        mapping = {val: i for i, val in enumerate(sorted(unique))}

    def map_labels(example):
        v = example[label_column]
        example[label_column] = mapping[str(v).lower()] if str(v).lower() in mapping else mapping.get(v, int(v))
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
        print(f"[data] Downloading dataset from {dataset_url} -> {dataset_path}")
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

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            device = logits.device
            weights = self.class_weights.to(device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

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
        split = ds["train"].train_test_split(
            test_size=val_size,
            seed=seed,
            stratify_by_column=label_col if label_col in ds["train"].column_names else None,
        )
        ds = DatasetDict(train=split["train"], validation=split["test"])

    # Ensure required columns exist
    for col in (text_col, label_col):
        if col not in ds["train"].column_names:
            raise ValueError(f"Column '{col}' not found in dataset. Available: {ds['train'].column_names}")

    return ds

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Self-installing QLoRA trainer for Gemma bad-word classification.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to CSV/JSON/JSONL with text+label columns.")
    parser.add_argument("--dataset_url", type=str, default=None, help="Optional URL to download dataset if dataset_path is missing.")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column.")
    parser.add_argument("--label_column", type=str, default="label", help="Name of the label column (0/1).")
    parser.add_argument("--base_model", type=str, default="google/gemma-2-2b-it", help="Base Gemma model to fine-tune.")
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
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory.")
    parser.add_argument("--preset", type=str, choices=["base", "low_vram"], default="base", help="Convenience presets for VRAM.")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU mode (discouraged; Gemma likely OOM on CPU).")
    args = parser.parse_args()

    # Apply presets
    if args.preset == "low_vram":
        # Lower memory footprint settings
        if not args.gradient_accumulation_steps or args.gradient_accumulation_steps < 16:
            args.gradient_accumulation_steps = 16
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1
        args.max_length = min(args.max_length, 128)
        args.gradient_checkpointing = True
        # Slightly lower LR can stabilize with high accumulation
        args.learning_rate = min(args.learning_rate, 2e-4)

    # Dataset presence or download
    dataset_path = maybe_download_dataset(args.dataset_path, args.dataset_url)

    # Environment checks
    has_cuda = torch.cuda.is_available()
    if not has_cuda and not args.force_cpu:
        raise SystemExit(
            "No CUDA GPU detected. QLoRA with bitsandbytes requires an NVIDIA GPU. "
            "If you still want to attempt CPU training (not recommended; may OOM), pass --force_cpu."
        )

    set_seed(args.seed)

    # Compute dtype
    compute_dtype = torch.bfloat16 if _bf16_supported() else torch.float16

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    ds = build_dataset(dataset_path, args.text_column, args.label_column, seed=args.seed, val_size=args.val_size)
    ds = ensure_label_ints(ds, args.label_column)

    def tokenize_fn(batch):
        toks = tokenizer(batch[args.text_column], max_length=args.max_length, truncation=True)
        toks["labels"] = batch[args.label_column]
        return toks

    ds_tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in ds["train"].column_names if c not in [args.text_column, args.label_column]],
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # Labels/config
    num_labels = int(max(int(max(ds["train"][args.label_column])), int(max(ds["validation"][args.label_column])))) + 1
    config = AutoConfig.from_pretrained(args.base_model, num_labels=num_labels)

    # Model loading (GPU QLoRA vs CPU)
    model_kwargs = {"config": config}
    if has_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs.update(
            dict(
                quantization_config=bnb_config,
                torch_dtype=compute_dtype,
                device_map="auto",
            )
        )
    else:
        # CPU path (likely OOM for Gemma). No quantization_config.
        model_kwargs.update(dict(torch_dtype=torch.float32))

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, **model_kwargs)

    if args.gradient_checkpointing and has_cuda:
        model.gradient_checkpointing_enable()

    # Prepare for k-bit training and apply LoRA (GPU only)
    if has_cuda:
        model = prepare_model_for_kbit_training(model)

    # Typical Llama/Gemma target modules
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    model = get_peft_model(model, lora_cfg)

    # Class weights from training labels
    class_weights = compute_class_weights(list(ds["train"][args.label_column]))

    # Metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
        f1_macro = f1.compute(predictions=preds, references=labels, average="macro")["f1"]
        precision, recall, f1_bin, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "precision_bin": precision,
            "recall_bin": recall,
            "f1_bin": f1_bin,
        }

    # Training args
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
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1_bin",
        greater_is_better=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )
    if has_cuda:
        training_kwargs.update(
            dict(
                bf16=(compute_dtype == torch.bfloat16),
                fp16=(compute_dtype == torch.float16),
            )
        )

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

    # Train
    t0 = time.time()
    trainer.train()
    t1 = time.time()

    # Save adapter and tokenizer only (small)
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Export minimal inference card
    card = {
        "base_model": args.base_model,
        "adapter_path": args.output_dir,
        "task": "sequence-classification",
        "labels": list(range(num_labels)),
        "text_column": args.text_column,
        "label_column": args.label_column,
        "max_length": args.max_length,
        "notes": "Load base Gemma in 4-bit (GPU) and merge with this adapter for bad-word classification.",
        "train_seconds": round(t1 - t0, 2),
        "preset": args.preset,
    }
    with open(os.path.join(args.output_dir, "inference_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    print(f"Training complete in {round(t1 - t0, 2)}s. Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()