#!/usr/bin/env python3
"""
Convenience entrypoint to run the bad-words QLoRA trainer.

This wraps train_gemma_badwords.py: it keeps Gemma as the default base model
but will automatically fall back to an open model if the Gemma repo is gated
and you are not authenticated with Hugging Face.

Usage:
  python run.py --help
  python run.py --dataset_path data/auto_badwords_en_si.csv

To use Gemma, ensure you have accepted the license and are logged in:
  export HF_TOKEN=hf_xxx
  python run.py --base_model google/gemma-2-2b-it
"""
from train_gemma_badwords import main

if __name__ == "__main__":
    main()