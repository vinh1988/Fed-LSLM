"""Build SFT JSONL dataset for Qwen2.5-VL training using existing FAISS retriever.

For each training sample:
  1. Encode image + question with BiomedCLIP.
  2. Retrieve top-k similar examples from FAISS index.
  3. Exclude self (same row_id) and optionally same image_path to prevent leakage.
  4. Format as Qwen2.5-VL chat messages (system → user → assistant).
  5. Write one JSONL record.

Two cases:
  - text_only:  retrieved Q+A text only, current image as single image input.
  - text_image: retrieved Q+A text + retrieved images as additional image inputs.

Usage:
    python scripts/build_vslm_mrag_sft_dataset.py \\
        --csv /kaggle/input/.../train.csv \\
        --data-root /kaggle/input/.../raw \\
        --index-dir outputs/mrag/index \\
        --output-jsonl outputs/vslm_mrag/sft/train_text_only_top5.jsonl \\
        --top-k 5 --case text_only \\
        --exclude-self --exclude-same-image \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.biomedclip_encoder import BiomedCLIPEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("build_vslm_mrag_sft_dataset")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build SFT JSONL for Qwen2.5-VL mRAG training"
    )
    p.add_argument("--csv", type=str, required=True, help="Path to train.csv")
    p.add_argument("--data-root", type=str, required=True, help="Root dir containing images")
    p.add_argument("--index-dir", type=str, default="outputs/mrag/index")
    p.add_argument("--output-jsonl", type=str, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--case",
        type=str,
        default="text_only",
        choices=["text_only", "text_image"],
        help="text_only: retrieved Q+A text. text_image: include retrieved images.",
    )
    p.add_argument(
        "--exclude-self",
        action="store_true",
        default=False,
        help="Exclude current sample from retrieval results (anti-leakage).",
    )
    p.add_argument(
        "--exclude-same-image",
        action="store_true",
        default=False,
        help="Exclude samples with same image_path from retrieval (anti-leakage).",
    )
    p.add_argument("--max-samples", type=int, default=None,
                    help="Take first N rows (for quick debugging).")
    p.add_argument(
        "--sample-size", type=int, default=None,
        help="Sample N rows from the dataset (use with --sample-strategy).",
    )
    p.add_argument(
        "--sample-strategy", type=str, default="random",
        choices=["random", "stratified"],
        help="Sampling strategy: 'random' or 'stratified' (by question_type).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    p.add_argument(
        "--filter-by-question-type",
        action="store_true",
        default=False,
        help="Restrict retrieval to same question_type (open→open, closed→closed).",
    )
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_dataframe(
    df: pd.DataFrame,
    sample_size: int,
    strategy: str,
    seed: int,
) -> pd.DataFrame:
    """Sample rows from the DataFrame.

    Args:
        df: Full DataFrame.
        sample_size: Number of rows to sample.
        strategy: 'random' or 'stratified'.
        seed: Random seed for reproducibility.

    Returns:
        Sampled DataFrame.
    """
    if sample_size >= len(df):
        logger.warning(
            "sample_size=%d >= dataset size=%d — using entire dataset.",
            sample_size, len(df),
        )
        return df

    if strategy == "stratified":
        # Stratified sampling by question_type
        logger.info(
            "Stratified sampling: %d from %d rows by question_type (seed=%d)",
            sample_size, len(df), seed,
        )
        groups = df.groupby("question_type")
        group_counts = groups.size()
        total = len(df)

        sampled_parts = []
        remaining = sample_size
        group_names = sorted(group_counts.index.tolist())

        for i, gname in enumerate(group_names):
            group_df = groups.get_group(gname)
            proportion = len(group_df) / total

            if i == len(group_names) - 1:
                # Last group gets the remaining count to avoid rounding errors
                n_sample = remaining
            else:
                n_sample = int(round(proportion * sample_size))

            n_sample = min(n_sample, len(group_df), remaining)
            sampled_parts.append(
                group_df.sample(n=n_sample, random_state=seed)
            )
            remaining -= n_sample
            logger.info(
                "  %s: %d / %d (%.1f%%)",
                gname, n_sample, len(group_df), 100 * n_sample / max(len(group_df), 1),
            )

        result = pd.concat(sampled_parts).sample(
            frac=1, random_state=seed
        ).reset_index(drop=True)
        return result

    else:
        # Random sampling
        logger.info(
            "Random sampling: %d from %d rows (seed=%d)",
            sample_size, len(df), seed,
        )
        return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# FAISS retrieval with leakage filtering
# ---------------------------------------------------------------------------

def retrieve_with_filtering(
    retriever,
    image: Image.Image,
    question: str,
    current_row_id: str,
    current_image_path: str,
    top_k: int,
    exclude_self: bool,
    exclude_same_image: bool,
    filter_question_type: Optional[str] = None,
) -> list[dict]:
    """Retrieve top-k from FAISS, filtering out the current sample.

    Over-retrieves to ensure enough candidates remain after filtering.
    When *filter_question_type* is set, the retriever's built-in
    sub-index search is used so only same-type candidates are returned.
    """
    # When filtering by question_type the candidate pool is smaller,
    # so we over-retrieve more aggressively.
    over_k = max(top_k + 50, top_k * 10) if filter_question_type else top_k + 20
    candidates = retriever.retrieve(
        image, question, top_k=over_k,
        filter_question_type=filter_question_type,
    )

    filtered = []
    for c in candidates:
        # Exclude self by row_id
        if exclude_self and c.get("row_id", "") == current_row_id:
            continue
        # Exclude same image_path
        if exclude_same_image and c.get("image_path", "") == current_image_path:
            continue
        filtered.append(c)
        if len(filtered) >= top_k:
            break

    if len(filtered) < top_k:
        logger.warning(
            "Only %d/%d same-type candidates found for row_id=%s (type=%s).",
            len(filtered), top_k, current_row_id, filter_question_type or "all",
        )

    # Re-rank
    for i, item in enumerate(filtered, start=1):
        item["rank"] = i

    return filtered


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a medical visual question answering assistant. "
    "Answer concisely. Give only the final answer."
)


def build_messages_text_only(
    image_path_abs: str,
    question: str,
    question_type: str,
    answer: str,
    retrieved: list[dict],
) -> list[dict]:
    """Build Qwen2.5-VL chat messages for text_only case."""
    # Build retrieved knowledge text
    knowledge_lines = []
    for r in retrieved:
        rq = r.get("question", "").strip()
        ra = r.get("answer", "").strip()
        knowledge_lines.append(f"{r['rank']}. Q: {rq}\n   A: {ra}")
    knowledge_text = "\n".join(knowledge_lines)

    if question_type == "closed":
        instruction_hint = "Answer ONLY 'yes' or 'no'."
    else:
        instruction_hint = "Answer with a short medical phrase (1-5 words)."

    user_text = (
        f"Question type: {question_type}\n"
        f"Question: {question}\n\n"
        f"Retrieved Knowledge:\n{knowledge_text}\n\n"
        f"{instruction_hint}\n"
        f"Answer:"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path_abs},
                {"type": "text", "text": user_text},
            ],
        },
        {"role": "assistant", "content": answer},
    ]


def build_messages_text_image(
    image_path_abs: str,
    question: str,
    question_type: str,
    answer: str,
    retrieved: list[dict],
    data_root: Path,
) -> list[dict]:
    """Build Qwen2.5-VL chat messages for text_image case.

    Interleaves text and images to prevent model confusion by placing
    each image directly next to its corresponding case/question.
    """
    content_parts: list[dict] = []

    # 1. text: "Current image:"
    content_parts.append({"type": "text", "text": "Current image:\n"})
    # 2. image: current image
    content_parts.append({"type": "image", "image": image_path_abs})
    
    # 3. text: "Question type: {question_type}\nQuestion: {question}"
    content_parts.append({
        "type": "text",
        "text": f"\nQuestion type: {question_type}\nQuestion: {question}"
    })

    # 4, 5, 6. Retrieved cases
    for r in retrieved:
        rq = r.get("question", "").strip()
        ra = r.get("answer", "").strip()
        r_img_path = data_root / r.get("image_path", "")
        
        # 4. text: "Retrieved Case {i}:"
        content_parts.append({"type": "text", "text": f"\nRetrieved Case {r['rank']}:\n"})
        
        # 5. image: retrieved image i
        if r_img_path.exists():
            content_parts.append({"type": "image", "image": str(r_img_path.resolve())})
            
        # 6. text: "Q: {retrieved_question_i}\nA: {retrieved_answer_i}"
        content_parts.append({"type": "text", "text": f"\nQ: {rq}\nA: {ra}"})

    # Final instruction
    if question_type == "closed":
        instruction_hint = "\nAnswer ONLY 'yes' or 'no'.\nAnswer:"
    else:
        instruction_hint = "\nAnswer with a short medical phrase (1-5 words).\nAnswer:"

    content_parts.append({
        "type": "text",
        "text": instruction_hint
    })

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_parts},
        {"role": "assistant", "content": answer},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load retriever ---
    logger.info("Loading FAISS retriever from: %s", args.index_dir)
    from src.retrieval.faiss_retriever import FAISSRetriever
    retriever = FAISSRetriever().load(args.index_dir)

    # --- Load CSV ---
    df = pd.read_csv(args.csv)
    logger.info("Loaded CSV: %d rows", len(df))

    # Ensure row_id exists
    if "row_id" not in df.columns:
        df["row_id"] = [f"train_{i:06d}" for i in range(len(df))]

    # --- Sampling ---
    if args.max_samples is not None:
        # Old behavior: take first N rows (for quick debugging)
        df = df.head(args.max_samples).reset_index(drop=True)
        logger.info("Limiting to %d samples (--max-samples)", args.max_samples)
    elif args.sample_size is not None:
        # New behavior: random or stratified sampling
        df = _sample_dataframe(df, args.sample_size, args.sample_strategy, args.seed)

    # --- Process samples ---
    filter_qt = args.filter_by_question_type
    logger.info(
        "Building SFT dataset | case=%s | top_k=%d | exclude_self=%s | "
        "exclude_same_image=%s | filter_by_question_type=%s",
        args.case, args.top_k, args.exclude_self, args.exclude_same_image, filter_qt,
    )

    n_written = 0
    n_skipped = 0
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building SFT"):
            image_path_rel = str(row["image_path"]).strip()
            question = str(row["question"]).strip()
            answer = str(row["answer"]).strip()
            question_type = str(row["question_type"]).strip()
            row_id = str(row.get("row_id", ""))

            # Load image
            full_img_path = data_root / image_path_rel
            try:
                with Image.open(full_img_path) as img:
                    pil_image = img.convert("RGB").copy()
            except Exception as exc:
                logger.warning("Cannot load image %s: %s — skipping.", full_img_path, exc)
                n_skipped += 1
                continue

            # Skip empty answers
            if not answer:
                logger.warning("Empty answer for row_id=%s — skipping.", row_id)
                n_skipped += 1
                continue

            # Retrieve with anti-leakage filtering (+ optional question_type filter)
            qt_filter = question_type if filter_qt else None
            retrieved = retrieve_with_filtering(
                retriever=retriever,
                image=pil_image,
                question=question,
                current_row_id=row_id,
                current_image_path=image_path_rel,
                top_k=args.top_k,
                exclude_self=args.exclude_self,
                exclude_same_image=args.exclude_same_image,
                filter_question_type=qt_filter,
            )

            # Build absolute image path for JSONL
            abs_img_path = str(full_img_path.resolve())

            # Build messages
            if args.case == "text_only":
                messages = build_messages_text_only(
                    abs_img_path, question, question_type, answer, retrieved,
                )
            else:
                messages = build_messages_text_image(
                    abs_img_path, question, question_type, answer, retrieved, data_root,
                )

            # Build JSONL record
            record = {
                "id": row_id,
                "image": abs_img_path,
                "question": question,
                "question_type": question_type,
                "answer": answer,
                "retrieved": retrieved,
                "messages": messages,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    elapsed = time.time() - t0

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  V-SLM mRAG SFT Dataset Build Summary")
    print("=" * 65)
    print(f"  Case            : {args.case}")
    print(f"  Top-K           : {args.top_k}")
    print(f"  Exclude self    : {args.exclude_self}")
    print(f"  Exclude same img: {args.exclude_same_image}")
    print(f"  Filter by Q-type: {args.filter_by_question_type}")
    if args.sample_size is not None:
        print(f"  Sample size     : {args.sample_size}")
        print(f"  Sample strategy : {args.sample_strategy}")
        print(f"  Seed            : {args.seed}")
    elif args.max_samples is not None:
        print(f"  Max samples     : {args.max_samples} (first N)")
    print(f"  Records written : {n_written}")
    print(f"  Records skipped : {n_skipped}")
    print(f"  Output file     : {output_path}")
    print(f"  Elapsed         : {elapsed:.1f}s")
    print("=" * 65 + "\n")

    # Save sample IDs for reproducibility
    sample_ids_path = output_path.with_suffix(".sample_ids.json")
    sample_meta = {
        "csv": args.csv,
        "total_in_csv": len(pd.read_csv(args.csv)) if args.sample_size else n_written,
        "sample_size": args.sample_size,
        "sample_strategy": args.sample_strategy if args.sample_size else None,
        "seed": args.seed if args.sample_size else None,
        "max_samples": args.max_samples,
        "filter_by_question_type": args.filter_by_question_type,
        "n_written": n_written,
        "n_skipped": n_skipped,
        "selected_row_ids": df["row_id"].tolist(),
    }
    with open(sample_ids_path, "w", encoding="utf-8") as f:
        json.dump(sample_meta, f, indent=2)
    logger.info("Sample IDs saved to: %s", sample_ids_path)

    logger.info("Done! SFT dataset saved to: %s", output_path)


if __name__ == "__main__":
    main()
