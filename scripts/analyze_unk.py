"""Analyze <unk> problem in open-ended predictions and vocab coverage.

Usage (local):
    python scripts/analyze_unk.py \
        --pred-csv outputs/centralized_no_rag/predictions/test_predictions.csv \
        --train-csv processed_data/PathVQA/raw/train.csv

Usage (Kaggle):
    python scripts/analyze_unk.py \
        --pred-csv outputs/centralized_no_rag/predictions/test_predictions.csv \
        --train-csv /kaggle/input/datasets/lekimquang/pathvqa/raw/train.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import normalize_answer, _mean_rouge_l


def analyze_predictions(pred_csv: str) -> None:
    """Analyze <unk> rate and metric impact in prediction CSV."""
    df = pd.read_csv(pred_csv)

    # --- Overall stats ---
    total = len(df)
    closed_df = df[df["question_type"] == "closed"]
    open_df = df[df["question_type"] == "open"]

    print("=" * 70)
    print("PREDICTION ANALYSIS")
    print("=" * 70)
    print(f"Total samples:   {total}")
    print(f"  Closed:        {len(closed_df)}")
    print(f"  Open:          {len(open_df)}")
    print()

    # --- <unk> analysis for open-ended ---
    pred_col = "final_prediction" if "final_prediction" in open_df.columns else "prediction"
    open_preds = open_df[pred_col].astype(str).tolist()
    open_refs = open_df["reference"].astype(str).tolist()

    unk_mask = [p.strip() == "<unk>" for p in open_preds]
    n_unk = sum(unk_mask)
    n_non_unk = len(open_preds) - n_unk
    unk_ratio = n_unk / len(open_preds) if open_preds else 0

    print(f"Open-ended <unk> analysis:")
    print(f"  Predictions == '<unk>':  {n_unk} / {len(open_preds)} ({unk_ratio:.2%})")
    print(f"  Predictions != '<unk>':  {n_non_unk} / {len(open_preds)} ({1 - unk_ratio:.2%})")
    print()

    # --- ROUGE-L breakdown ---
    overall_rouge = _mean_rouge_l(open_preds, open_refs)

    non_unk_preds = [p for p, u in zip(open_preds, unk_mask) if not u]
    non_unk_refs = [r for r, u in zip(open_refs, unk_mask) if not u]
    non_unk_rouge = _mean_rouge_l(non_unk_preds, non_unk_refs) if non_unk_preds else 0.0

    unk_preds = [p for p, u in zip(open_preds, unk_mask) if u]
    unk_refs = [r for r, u in zip(open_refs, unk_mask) if u]
    unk_rouge = _mean_rouge_l(unk_preds, unk_refs) if unk_preds else 0.0

    print(f"Open ROUGE-L breakdown:")
    print(f"  Overall (all open):         {overall_rouge:.4f}")
    print(f"  Subset pred != '<unk>':     {non_unk_rouge:.4f}  (n={n_non_unk})")
    print(f"  Subset pred == '<unk>':     {unk_rouge:.4f}  (n={n_unk})")
    print()

    # --- Impact estimate ---
    if n_non_unk > 0:
        # If we could replace all <unk> with perfect answers:
        best_possible = (non_unk_rouge * n_non_unk + 1.0 * n_unk) / len(open_preds)
        # If we replaced <unk> with a reasonable guess (say matching non-unk ROUGE):
        reasonable = (non_unk_rouge * n_non_unk + non_unk_rouge * n_unk) / len(open_preds)
        print(f"Impact estimates (if <unk> were fixed):")
        print(f"  If <unk> → perfect answer: ROUGE-L = {best_possible:.4f}  (Δ = +{best_possible - overall_rouge:.4f})")
        print(f"  If <unk> → same quality:   ROUGE-L = {reasonable:.4f}  (Δ = +{reasonable - overall_rouge:.4f})")
        print()

    # --- Top predicted open answers ---
    pred_counter = Counter(open_preds)
    print(f"Top 20 predicted open answers:")
    for ans, count in pred_counter.most_common(20):
        pct = count / len(open_preds) * 100
        display = ans if len(ans) <= 50 else ans[:47] + "..."
        print(f"  {display:<55s}  {count:5d}  ({pct:.1f}%)")
    print()

    # --- Top reference open answers ---
    ref_counter = Counter(open_refs)
    print(f"Top 20 reference open answers:")
    for ans, count in ref_counter.most_common(20):
        pct = count / len(open_refs) * 100
        display = ans if len(ans) <= 50 else ans[:47] + "..."
        print(f"  {display:<55s}  {count:5d}  ({pct:.1f}%)")
    print()

    # --- Answers where ref is in vocab but model still predicted <unk> ---
    non_unk_vocab = set(p for p in open_preds if p.strip() != "<unk>")
    unk_refs_in_vocab = [r for r, u in zip(open_refs, unk_mask)
                         if u and normalize_answer(r) in {normalize_answer(v) for v in non_unk_vocab}]
    print(f"<unk> predictions where reference IS a known vocab word: {len(unk_refs_in_vocab)} / {n_unk}")
    if unk_refs_in_vocab:
        ref_in_vocab_counter = Counter(unk_refs_in_vocab)
        print("  Top examples:")
        for ans, count in ref_in_vocab_counter.most_common(10):
            print(f"    '{ans}': {count} times")
    print()


def analyze_vocab_coverage(train_csv: str) -> None:
    """Analyze vocab coverage at different sizes."""
    df = pd.read_csv(train_csv)

    # Detect format
    if "question_type" in df.columns:
        open_df = df[df["question_type"] == "open"]
    else:
        # Fallback: infer from answer
        open_df = df[~df["answer"].astype(str).str.lower().str.strip().isin({"yes", "no"})]

    all_answers = open_df["answer"].astype(str).str.lower().str.strip().tolist()
    total_open = len(all_answers)
    unique_answers = set(all_answers)
    answer_counts = Counter(all_answers)

    print("=" * 70)
    print("VOCAB COVERAGE ANALYSIS (train set)")
    print("=" * 70)
    print(f"Total open-ended train samples:  {total_open}")
    print(f"Unique open answers:             {len(unique_answers)}")
    print()

    # Coverage at different vocab sizes
    vocab_sizes = [500, 1000, 1500, 2000, 3000, 5000, len(unique_answers)]
    print(f"{'Vocab Size':<12s}  {'Unique Covered':<16s}  {'Coverage %':<12s}  {'Sample Unk Rate':<16s}  {'Unk Samples':<12s}")
    print("-" * 70)

    for vs in vocab_sizes:
        label = f"all ({vs})" if vs == len(unique_answers) else str(vs)
        # Build vocab of top-K
        top_k = [ans for ans, _ in answer_counts.most_common(vs)]
        top_k_set = set(top_k)

        covered_unique = len(unique_answers & top_k_set)
        coverage_pct = covered_unique / len(unique_answers) * 100

        unk_samples = sum(1 for a in all_answers if a not in top_k_set)
        unk_rate = unk_samples / total_open * 100

        print(f"{label:<12s}  {covered_unique:<16d}  {coverage_pct:<12.2f}  {unk_rate:<16.2f}  {unk_samples:<12d}")

    print()

    # Show tail answers (count == 1)
    singletons = [ans for ans, cnt in answer_counts.items() if cnt == 1]
    print(f"Singleton answers (appear only once): {len(singletons)} ({len(singletons)/len(unique_answers)*100:.1f}% of unique)")
    print(f"Examples of singletons: {singletons[:10]}")
    print()

    # Distribution of answer frequencies
    freq_buckets = {1: 0, 2: 0, "3-5": 0, "6-10": 0, "11-50": 0, "50+": 0}
    for cnt in answer_counts.values():
        if cnt == 1:
            freq_buckets[1] += 1
        elif cnt == 2:
            freq_buckets[2] += 1
        elif cnt <= 5:
            freq_buckets["3-5"] += 1
        elif cnt <= 10:
            freq_buckets["6-10"] += 1
        elif cnt <= 50:
            freq_buckets["11-50"] += 1
        else:
            freq_buckets["50+"] += 1

    print("Answer frequency distribution (unique answers):")
    for bucket, count in freq_buckets.items():
        print(f"  freq={str(bucket):<6s}  {count:5d} unique answers")


def main():
    parser = argparse.ArgumentParser(description="Analyze <unk> in predictions")
    parser.add_argument(
        "--predictions",
        "--pred-csv",
        dest="pred_csv",
        type=str,
        default="outputs/centralized_no_rag/predictions/test_predictions.csv",
        help="Path to predictions CSV file"
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="processed_data/PathVQA/raw/train.csv",
        help="Path to train CSV file"
    )
    args = parser.parse_args()

    if Path(args.pred_csv).exists():
        analyze_predictions(args.pred_csv)
    else:
        print(f"[WARN] Prediction CSV not found: {args.pred_csv}")

    if Path(args.train_csv).exists():
        analyze_vocab_coverage(args.train_csv)
    else:
        print(f"[WARN] Train CSV not found: {args.train_csv}")


if __name__ == "__main__":
    main()
