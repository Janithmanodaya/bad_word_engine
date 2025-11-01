#!/usr/bin/env python3
"""
Convenience entrypoint to run the compact ML bad-words trainer.

This wraps train_gemma_badwords.py which trains a small classifier (scikit-learn TF-IDF + LinearSVC
or optional FastText) without any LLM components.

Usage:
  python run.py --help
  python run.py --dataset_path data/auto_badwords_en_si.csv
  python run.py --extra_bad_csv bad.csv  # augment with your Sinhala comma-separated list
"""
from train_gemma_badwords import main

if __name__ == "__main__":
    main()