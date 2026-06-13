"""Compute BERTScore from a predictions.csv file (standalone).

Reads predictions.csv (output of evaluate_vslm_mrag.py), filters
open-ended samples (question_type == "open"), computes BERTScore
between prediction and reference, and saves/prints results.

Usage:
    python scripts/compute_bertscore_from_predictions.py \\
        --predictions outputs/vslm_mrag/eval_val/predictions.csv \\
        --output-json outputs/vslm_mrag/eval_val/bertscore_metrics.json \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import compute_bertscore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("compute_bertscore_from_predictions")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute BERTScore from predictions.csv"
    )
    p.add_argument(
        "--predictions", type=str, required=True,
        help="Path to predictions.csv (must have 'prediction', 'reference', 'question_type' columns)",
    )
    p.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save BERTScore metrics JSON. If not given, prints to stdout only.",
    )
    p.add_argument(
        "--model-type", type=str, default=None,
        help="HuggingFace model for BERTScore (default: distilbert-base-uncased).",
    )
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load predictions
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        logger.error("Predictions file not found: %s", pred_path)
        sys.exit(1)

    df = pd.read_csv(pred_path)
    logger.info("Loaded predictions: %d rows", len(df))

    # Validate required columns
    required_cols = {"prediction", "reference", "question_type"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("Missing columns in predictions.csv: %s", missing)
        sys.exit(1)

    # Filter open-ended samples only
    open_df = df[df["question_type"] == "open"].copy()
    logger.info("Open-ended samples: %d / %d total", len(open_df), len(df))

    if len(open_df) == 0:
        logger.warning("No open-ended samples found. Nothing to compute.")
        result = {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
            "bertscore_model": args.model_type or "N/A",
            "n_open": 0,
            "n_total": len(df),
        }
    else:
        predictions = open_df["prediction"].fillna("").tolist()
        references = open_df["reference"].fillna("").tolist()

        result = compute_bertscore(
            predictions, references,
            model_type=args.model_type,
            device=args.device,
        )
        result["n_open"] = len(open_df)
        result["n_total"] = len(df)

    # Print results
    print("\n" + "=" * 55)
    print("  BERTScore Results (Open-Ended Only)")
    print("=" * 55)
    print(f"  Total samples : {result['n_total']}")
    print(f"  Open samples  : {result['n_open']}")
    print(f"  Model         : {result['bertscore_model']}")
    print("-" * 55)
    print(f"  Precision     : {result['bertscore_precision']:.4f}")
    print(f"  Recall        : {result['bertscore_recall']:.4f}")
    print(f"  F1            : {result['bertscore_f1']:.4f}")
    print("=" * 55 + "\n")

    # Save to JSON
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info("BERTScore metrics saved to: %s", output_path)

    # Try to merge with existing metrics.json in the same directory
    if args.output_json:
        metrics_path = output_path.parent / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, encoding="utf-8") as f:
                    existing = json.load(f)
                existing["open_bertscore"] = result["bertscore_f1"]
                existing["bertscore_precision"] = result["bertscore_precision"]
                existing["bertscore_recall"] = result["bertscore_recall"]
                existing["bertscore_model"] = result["bertscore_model"]
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=2)
                logger.info("Merged BERTScore into existing metrics.json: %s", metrics_path)
            except Exception as exc:
                logger.warning("Could not merge into metrics.json: %s", exc)

    logger.info("Done!")


if __name__ == "__main__":
    main()
