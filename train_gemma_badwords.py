#!/usr/bin/env python3
"""
QLoRA fine-tuning script for bad-word classification using Google Gemma.

- Uses 4-bit quantization (bitsandbytes) to fit large models on a single GPU
- Applies PEFT LoRA adapters only to attention/MLP matrices (memory efficient)
- Works with CSV/JSON/JSONL datasets
- Handles class imbalance via weighted cross-entropy
- Saves only the LoRA adapter by default (tiny footprint)
- Sensible defaults for resource-constrained setups

Example (quick start):
    pip install -U "transformers&gt;=4.43" peft bitsandbytes datasets accelerate evaluate scikit-learn
    python train_gemma_badwords.py \
        --dataset_path data/badwords.csv \
        --text_column text --label_column label \
        --base_model google/gemma-2-2b-it \
        --output_dir outputs/gemma-badwords-qlora

Notes:
- Training should be done on your gaming PC (GPU recommended). The final adapter is small and can be served on your tiny server.
- For best results, prepare a balanced dataset with clear labels: 1 = bad/offensive, 0 = clean.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, List, Any

import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
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
        split = ds["train"].train_test_split(test_size=val_size, seed=seed, stratify_by_column=label_col if label_col in ds["train"].column_names else None)
        ds = DatasetDict(train=split["train"], validation=split["test"])

    # Ensure required columns exist
    for col in (text_col, label_col):
        if col not in ds["train"].column_names:
            raise ValueError(f"Column '{col}' not found in dataset. Available: {ds['train'].column_names}")

    return ds


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
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for bad-word classification on Gemma.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to CSV/JSON/JSONL with text+label columns.")
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
    args = parser.parse_args()

    set_seed(args.seed)

    compute_dtype = torch.bfloat16 if _bf16_supported() else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    ds = build_dataset(args.dataset_path, args.text_column, args.label_column, seed=args.seed, val_size=args.val_size)
    ds = ensure_label_ints(ds, args.label_column)

    def tokenize_fn(batch):
        toks = tokenizer(batch[args.text_column], max_length=args.max_length, truncation=True)
        toks["labels"] = batch[args.label_column]
        return toks

    ds_tokenized = ds.map(tokenize_fn, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in [args.text_column, args.label_column]])
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # Config and model
    num_labels = int(max(int(max(ds["train"][args.label_column])), int(max(ds["validation"][args.label_column])))) + 1
    config = AutoConfig.from_pretrained(args.base_model, num_labels=num_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        config=config,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Prepare for k-bit training and apply LoRA
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
        precision, recall, f1_bin, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "precision_bin": precision,
            "recall_bin": recall,
            "f1_bin": f1_bin,
        }

    # Training args tuned for small VRAM
    train_args = TrainingArguments(
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
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1_bin",
        greater_is_better=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

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
    trainer.train()

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
        "notes": "Load base Gemma in 4-bit and merge with this adapter for bad-word classification.",
    }
    with open(os.path.join(args.output_dir, "inference_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    print(f"Training complete. Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()