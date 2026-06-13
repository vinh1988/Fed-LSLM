"""Centralized mRAG evaluation script for PathVQA.

Two modes:
  - generation    : FAISS retrieval → Qwen2.5-0.5B answer generation
  - retrieval_only: prediction = answer of top-1 retrieved sample

Recommended test order:
  1.  --max-samples 20   (smoke test)
  2.  --max-samples 200  (sanity check)
  3.  Full validation/test

Usage (smoke test):
    python scripts/evaluate_mrag.py \\
        --eval-csv processed_data/PathVQA/raw/validation.csv \\
        --data-root processed_data/PathVQA/raw \\
        --index-dir outputs/mrag/index \\
        --output-dir outputs/mrag/debug_val20 \\
        --top-k 5 --max-samples 20 --mode generation --device cpu

Usage (full test, Kaggle GPU):
    python scripts/evaluate_mrag.py \\
        --eval-csv /kaggle/.../test.csv \\
        --data-root /kaggle/.../raw \\
        --index-dir outputs/mrag/index \\
        --output-dir outputs/mrag/centralized_qwen05b_top5/test \\
        --top-k 5 --mode generation --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.faiss_retriever import FAISSRetriever
from src.evaluation.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_mrag")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate centralized mRAG pipeline on PathVQA"
    )
    parser.add_argument("--eval-csv", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument(
        "--index-dir",
        type=str,
        default="outputs/mrag/index",
        help="Directory containing faiss.index, metadata.jsonl, index_config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/mrag/eval",
        help="Directory to save predictions and metrics",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mode",
        type=str,
        default="generation",
        choices=["generation", "retrieval_only"],
        help=(
            "generation: use Qwen to generate answer from retrieved context. "
            "retrieval_only: use top-1 retrieved answer directly."
        ),
    )
    parser.add_argument(
        "--filter-by-question-type",
        action="store_true",
        default=False,
        help="If set, closed queries retrieve from closed examples only, open from open.",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="qwen2.5-0.5b",
        help="Generator model alias (only used if mode=generation). 'qwen2.5-0.5b' → Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max tokens for Qwen generation",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of eval samples (for smoke testing)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GENERATOR_ALIASES = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
}


def resolve_model_name(alias: str) -> str:
    return _GENERATOR_ALIASES.get(alias.lower(), alias)


def resolve_kaggle_paths(eval_csv: str, data_root: str, index_dir: str) -> tuple[str, str, str]:
    """Fallback to Kaggle dataset paths when local paths don't exist."""
    kaggle_raw = Path("/kaggle/input/datasets/lekimquang/pathvqa/raw")
    if not Path(eval_csv).exists() and kaggle_raw.exists():
        logger.info("Local eval CSV not found — falling back to Kaggle paths.")
        csv_name = Path(eval_csv).name
        eval_csv = str(kaggle_raw / csv_name)
        data_root = str(kaggle_raw)
    return eval_csv, data_root, index_dir


def load_eval_df(eval_csv: str, max_samples: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(eval_csv)
    logger.info("Loaded eval CSV: %d rows", len(df))
    if max_samples is not None:
        df = df.head(max_samples).reset_index(drop=True)
        logger.info("Limiting to %d samples (--max-samples)", max_samples)
    # Ensure row_id exists
    if "row_id" not in df.columns:
        df["row_id"] = [f"eval_{i:06d}" for i in range(len(df))]
    return df


def load_generator(args: argparse.Namespace):
    """Load Qwen generator based on args.generator alias."""
    from src.generation.qwen_generator import QwenAnswerGenerator

    model_name = resolve_model_name(args.generator)
    logger.info("Loading generator: %s (alias=%s)", model_name, args.generator)
    return QwenAnswerGenerator(
        model_name=model_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    df: pd.DataFrame,
    data_root: Path,
    retriever: FAISSRetriever,
    generator,
    args: argparse.Namespace,
) -> tuple[list[dict], list[str], list[str], list[str]]:
    """Run retrieval (+ optional generation) for all samples.

    Returns:
        (rows, predictions, references, question_types)
        where rows is a list of record dicts for the CSV output.
    """
    rows = []
    predictions = []
    references = []
    question_types = []

    for _, sample in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        row_id = str(sample.get("row_id", ""))
        image_path_rel = str(sample["image_path"]).strip()
        question = str(sample["question"])
        reference = str(sample["answer"])
        question_type = str(sample["question_type"])

        # Load image
        full_img_path = data_root / image_path_rel
        try:
            with Image.open(full_img_path) as img:
                pil_image = img.convert("RGB").copy()
        except Exception as exc:
            logger.warning("Cannot load image %s: %s — skipping.", full_img_path, exc)
            continue

        # Determine filter
        filter_qt = question_type if args.filter_by_question_type else None

        # Retrieve
        retrieved = retriever.retrieve(
            pil_image, question, top_k=args.top_k, filter_question_type=filter_qt
        )

        # Generate or use top-1
        raw_output: Optional[str] = None
        if args.mode == "retrieval_only":
            if retrieved:
                prediction = str(retrieved[0]["answer"]).strip()
            else:
                prediction = "yes" if question_type == "closed" else ""
        else:
            gen_result = generator.generate_answer(question, question_type, retrieved)
            raw_output = gen_result["raw_output"]
            prediction = gen_result["final_prediction"]

        # Collect for metrics
        predictions.append(prediction)
        references.append(reference)
        question_types.append(question_type)

        # Build record for CSV
        rows.append(
            {
                "row_id": row_id,
                "image_path": image_path_rel,
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "question_type": question_type,
                "retrieved_questions": " | ".join(
                    r.get("question", "") for r in retrieved
                ),
                "retrieved_answers": " | ".join(
                    r.get("answer", "") for r in retrieved
                ),
                "retrieval_scores": " | ".join(
                    f"{r.get('score', 0):.4f}" for r in retrieved
                ),
                "correct": int(
                    prediction.strip().lower() == reference.strip().lower()
                ),
                "raw_output": raw_output if raw_output is not None else "",
            }
        )

    return rows, predictions, references, question_types


# ---------------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------------

def save_outputs(
    output_dir: Path,
    rows: list[dict],
    metrics: dict,
    retrieval_examples: list[dict],
    retriever: FAISSRetriever,
    args: argparse.Namespace,
    elapsed: float,
) -> None:
    """Save predictions.csv, metrics.json, retrieval_examples.jsonl, config.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # predictions.csv
    pred_path = output_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    logger.info("Predictions saved: %s (%d rows)", pred_path, len(rows))

    # Resolve encoder_actual from index config
    encoder_actual = retriever.index_config.get("encoder_actual", "unknown")

    # metrics.json
    metrics["encoder_actual"] = encoder_actual
    metrics["mode"] = args.mode
    metrics["filter_by_question_type"] = args.filter_by_question_type
    metrics["top_k"] = args.top_k
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved: %s", metrics_path)

    # retrieval_examples.jsonl (first 200 rows for disk efficiency)
    ret_path = output_dir / "retrieval_examples.jsonl"
    with open(ret_path, "w", encoding="utf-8") as f:
        for row in retrieval_examples[:200]:
            f.write(json.dumps(row) + "\n")
    logger.info("Retrieval examples saved: %s", ret_path)

    # config.json
    config = {
        "eval_csv": str(args.eval_csv),
        "data_root": str(args.data_root),
        "index_dir": str(args.index_dir),
        "output_dir": str(args.output_dir),
        "top_k": args.top_k,
        "mode": args.mode,
        "filter_by_question_type": args.filter_by_question_type,
        "generator": args.generator if args.mode == "generation" else None,
        "max_new_tokens": args.max_new_tokens if args.mode == "generation" else None,
        "device": args.device,
        "max_samples": args.max_samples,
        "encoder_actual": encoder_actual,
        "index_config": retriever.index_config,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Config saved: %s", config_path)


def print_summary(metrics: dict, args: argparse.Namespace) -> None:
    """Print a formatted metrics summary table."""
    print("\n" + "=" * 65)
    print(f"  mRAG Evaluation Summary | mode={args.mode} | top-k={args.top_k}")
    print("=" * 65)
    print(f"  Total samples   : {metrics.get('n_total', 0)}")
    print(f"  Closed samples  : {metrics.get('n_closed', 0)}")
    print(f"  Open samples    : {metrics.get('n_open', 0)}")
    print("-" * 65)
    print(f"  Overall EM      : {metrics.get('overall_exact_match', 0):.4f}")
    print(f"  Closed Accuracy : {metrics.get('closed_accuracy', 0):.4f}")
    print(f"  Closed F1       : {metrics.get('closed_f1', 0):.4f}")
    print(f"  Open EM         : {metrics.get('open_exact_match', 0):.4f}")
    print(f"  Open BLEU-1     : {metrics.get('open_bleu', 0):.4f}")
    print(f"  Open ROUGE-L    : {metrics.get('open_rouge_l', 0):.4f}")
    print("-" * 65)
    print(f"  Encoder actual  : {metrics.get('encoder_actual', 'N/A')}")
    print(f"  Elapsed         : {metrics.get('elapsed_seconds', 0):.1f}s")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve Kaggle paths
    args.eval_csv, args.data_root, args.index_dir = resolve_kaggle_paths(
        args.eval_csv, args.data_root, args.index_dir
    )

    output_dir = Path(args.output_dir)
    data_root = Path(args.data_root)

    # 1. Load retriever
    logger.info("Loading FAISS retriever from: %s", args.index_dir)
    retriever = FAISSRetriever().load(args.index_dir)

    # 2. Load generator (if needed)
    generator = None
    if args.mode == "generation":
        generator = load_generator(args)
    else:
        logger.info("Mode=retrieval_only — skipping generator load.")

    # 3. Load eval data
    df = load_eval_df(args.eval_csv, args.max_samples)

    # 4. Run evaluation
    logger.info(
        "Starting evaluation | mode=%s | filter_by_qt=%s | top_k=%d | n=%d",
        args.mode,
        args.filter_by_question_type,
        args.top_k,
        len(df),
    )
    t0 = time.time()
    rows, predictions, references, question_types = run_evaluation(
        df, data_root, retriever, generator, args
    )
    elapsed = time.time() - t0

    if not predictions:
        logger.error("No predictions generated — check image paths or CSV format.")
        sys.exit(1)

    # 5. Compute metrics
    logger.info("Computing metrics over %d samples...", len(predictions))
    metrics = compute_metrics(predictions, references, question_types)

    # 6. Build retrieval_examples (for .jsonl, pick rows that have retrieval info)
    retrieval_examples = [
        {
            "row_id": r["row_id"],
            "question": r["question"],
            "reference": r["reference"],
            "prediction": r["prediction"],
            "question_type": r["question_type"],
            "retrieved_questions": r["retrieved_questions"].split(" | "),
            "retrieved_answers": r["retrieved_answers"].split(" | "),
            "retrieval_scores": [
                float(s) for s in r["retrieval_scores"].split(" | ") if s
            ],
            "raw_output": r.get("raw_output", ""),
        }
        for r in rows
    ]

    # 7. Save
    save_outputs(output_dir, rows, metrics, retrieval_examples, retriever, args, elapsed)

    # 8. Print summary
    print_summary(metrics, args)

    logger.info("Evaluation complete. Results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
