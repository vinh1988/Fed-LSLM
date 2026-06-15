"""Evaluate V-SLM mRAG pipeline on PathVQA validation/test sets.

Loads a trained Qwen2.5-VL LoRA adapter, retrieves top-k evidence from
FAISS, generates answers, post-processes, and computes metrics.

Usage:
    python scripts/evaluate_vslm_mrag.py \\
        --model-name Qwen/Qwen2.5-VL-3B-Instruct \\
        --adapter-path outputs/vslm_mrag/qwen25vl3b_lora_text_only_top5 \\
        --eval-csv /kaggle/.../validation.csv \\
        --data-root /kaggle/.../raw \\
        --index-dir outputs/mrag/index \\
        --output-dir outputs/vslm_mrag/eval_val \\
        --top-k 5 --case text_only --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
from src.evaluation.metrics import compute_metrics, compute_bertscore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_vslm_mrag")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate V-SLM mRAG (Qwen2.5-VL + LoRA) on PathVQA"
    )
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter-path", type=str, required=True,
                    help="Path to saved LoRA adapter directory")
    p.add_argument("--eval-csv", type=str, required=True)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--index-dir", type=str, default="outputs/mrag/index")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--case", type=str, default="text_only",
                    choices=["text_only", "text_image"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use-4bit", action="store_true", default=False)
    p.add_argument("--min-pixels", type=int, default=128 * 28 * 28)
    p.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    p.add_argument(
        "--compute-bertscore", action="store_true", default=False,
        help="Compute BERTScore on open-ended samples (requires bert-score package).",
    )
    p.add_argument(
        "--bertscore-model", type=str, default=None,
        help="HuggingFace model for BERTScore (default: distilbert-base-uncased).",
    )
    p.add_argument(
        "--filter-by-question-type",
        action="store_true",
        default=False,
        help="Restrict retrieval to same question_type (open→open, closed→closed).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------

def detect_model_family(model_name: str) -> str:
    """Detect model family from model name.

    Returns 'qwen2_5_vl', 'qwen2_vl', or 'smolvlm'.
    """
    name_lower = model_name.lower()
    if "smolvlm" in name_lower:
        return "smolvlm"
    elif "qwen2.5-vl" in name_lower or "qwen2_5_vl" in name_lower:
        return "qwen2_5_vl"
    elif "qwen2-vl" in name_lower or "qwen2_vl" in name_lower:
        return "qwen2_vl"
    else:
        logger.warning(
            "Cannot detect model family from '%s', defaulting to qwen2_5_vl",
            model_name,
        )
        return "qwen2_5_vl"


def _get_model_class(family: str):
    """Import and return the correct model class for the given family."""
    if family == "smolvlm":
        try:
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq
        except ImportError:
            from transformers import Idefics3ForConditionalGeneration
            return Idefics3ForConditionalGeneration
    elif family == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(args: argparse.Namespace):
    """Load base model + LoRA adapter + processor.

    Auto-detects model family from --model-name and uses the correct class.
    """
    import torch
    from transformers import AutoProcessor

    model_family = detect_model_family(args.model_name)
    model_cls = _get_model_class(model_family)
    logger.info("Detected model family: %s -> %s", model_family, model_cls.__name__)

    logger.info("Loading processor from: %s", args.adapter_path)
    if model_family == "smolvlm":
        processor = AutoProcessor.from_pretrained(
            args.adapter_path,
            trust_remote_code=True,
        )
        # Disable image tiling: 4 images × ~1000 tokens (tiled) = 4000+ tokens > max_length
        if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "do_image_splitting"):
            processor.image_processor.do_image_splitting = False
            logger.info("SmolVLM: do_image_splitting=False (prevents token explosion)")
    else:
        processor = AutoProcessor.from_pretrained(
            args.adapter_path,
            trust_remote_code=True,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )

    model_kwargs = {"trust_remote_code": True}

    if args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
        except ImportError:
            logger.warning("bitsandbytes not available — loading in float16")
            model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model_kwargs["device_map"] = "auto"

    logger.info("Loading base model: %s (class=%s)", args.model_name, model_cls.__name__)
    model = model_cls.from_pretrained(
        args.model_name, **model_kwargs
    )

    # Load LoRA adapter
    from peft import PeftModel
    logger.info("Loading LoRA adapter from: %s", args.adapter_path)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    logger.info("Model loaded successfully (LoRA merged for inference)")
    return model, processor


# ---------------------------------------------------------------------------
# Prompt building (inference — no assistant answer)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a medical visual question answering assistant. "
    "Answer concisely. Give only the final answer."
)


def build_inference_messages_text_only(
    image_path_abs: str,
    question: str,
    question_type: str,
    retrieved: list[dict],
) -> list[dict]:
    """Build inference prompt (text_only case — no assistant answer)."""
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
    ]


def build_inference_messages_text_image(
    image_path_abs: str,
    question: str,
    question_type: str,
    retrieved: list[dict],
    data_root: Path,
) -> list[dict]:
    """Build inference prompt (text_image case — with retrieved images).

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
    ]


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def postprocess_answer(raw: str, question_type: str) -> str:
    """Clean model output into final answer."""
    text = raw.strip()

    # Remove echoed prompt artifacts
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()

    if question_type == "closed":
        match = _YES_NO_RE.search(text)
        if match:
            return match.group(0).lower()
        return "no"
    else:
        # Take first non-empty line, strip punctuation
        for line in text.splitlines():
            line = line.strip().strip(".").strip(",").strip()
            if line:
                return line
        return text[:100]


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference(
    model,
    processor,
    retriever: FAISSRetriever,
    df: pd.DataFrame,
    data_root: Path,
    args: argparse.Namespace,
) -> tuple[list[dict], list[str], list[str], list[str]]:
    """Run inference over all eval samples."""
    import torch
    from src.vslm.qwen_vl_dataset import preprocess_messages_for_vlm

    model_family = detect_model_family(args.model_name)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

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

        # Retrieve top-k from FAISS (with optional question_type filtering)
        qt_filter = question_type if args.filter_by_question_type else None
        retrieved = retriever.retrieve(
            pil_image, question, top_k=args.top_k,
            filter_question_type=qt_filter,
        )

        # Build inference messages
        abs_img_path = str(full_img_path.resolve())
        if args.case == "text_only":
            messages = build_inference_messages_text_only(
                abs_img_path, question, question_type, retrieved,
            )
        else:
            messages = build_inference_messages_text_image(
                abs_img_path, question, question_type, retrieved, data_root,
            )

        # Preprocess messages for VLM
        processed_messages, image_inputs = preprocess_messages_for_vlm(messages, model_family)

        # Process through VLM
        text = processor.apply_chat_template(
            processed_messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

        # Decode only new tokens
        generated_ids = output_ids[0][input_len:]
        raw_output = processor.decode(generated_ids, skip_special_tokens=True)
        prediction = postprocess_answer(raw_output, question_type)

        # Collect
        predictions.append(prediction)
        references.append(reference)
        question_types.append(question_type)

        rows.append({
            "row_id": row_id,
            "image_path": image_path_rel,
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "question_type": question_type,
            "retrieved_questions": " | ".join(r.get("question", "") for r in retrieved),
            "retrieved_answers": " | ".join(r.get("answer", "") for r in retrieved),
            "retrieved_image_paths": " | ".join(r.get("image_path", "") for r in retrieved),
            "retrieval_scores": " | ".join(f"{r.get('score', 0):.4f}" for r in retrieved),
            "raw_output": raw_output,
            "correct": int(prediction.strip().lower() == reference.strip().lower()),
        })

    return rows, predictions, references, question_types


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    output_dir: Path,
    rows: list[dict],
    metrics: dict,
    retriever: FAISSRetriever,
    args: argparse.Namespace,
    elapsed: float,
) -> None:
    """Save predictions, metrics, retrieval examples, and config."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # predictions.csv
    pred_path = output_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    logger.info("Predictions saved: %s (%d rows)", pred_path, len(rows))

    # metrics.json
    encoder_actual = retriever.index_config.get("encoder_actual", "unknown")
    metrics["encoder_actual"] = encoder_actual
    metrics["model_name"] = args.model_name
    metrics["adapter_path"] = args.adapter_path
    metrics["case"] = args.case
    metrics["top_k"] = args.top_k
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved: %s", metrics_path)

    # retrieval_examples.jsonl (first 200)
    ret_path = output_dir / "retrieval_examples.jsonl"
    with open(ret_path, "w", encoding="utf-8") as f:
        for row in rows[:200]:
            f.write(json.dumps({
                "row_id": row["row_id"],
                "question": row["question"],
                "reference": row["reference"],
                "prediction": row["prediction"],
                "question_type": row["question_type"],
                "retrieved_questions": row["retrieved_questions"].split(" | "),
                "retrieved_answers": row["retrieved_answers"].split(" | "),
                "retrieval_scores": [
                    float(s) for s in row["retrieval_scores"].split(" | ") if s
                ],
                "raw_output": row.get("raw_output", ""),
            }) + "\n")
    logger.info("Retrieval examples saved: %s", ret_path)

    # config.json
    config = {
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "eval_csv": args.eval_csv,
        "data_root": args.data_root,
        "index_dir": args.index_dir,
        "output_dir": args.output_dir,
        "top_k": args.top_k,
        "case": args.case,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "use_4bit": args.use_4bit,
        "filter_by_question_type": args.filter_by_question_type,
        "encoder_actual": encoder_actual,
        "index_config": retriever.index_config,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Config saved: %s", config_path)


def print_summary(metrics: dict, args: argparse.Namespace) -> None:
    """Print formatted evaluation summary."""
    print("\n" + "=" * 65)
    print(f"  V-SLM mRAG Evaluation | case={args.case} | top-k={args.top_k}")
    print(f"  filter_by_question_type={args.filter_by_question_type}")
    print("=" * 65)
    print(f"  Model         : {args.model_name}")
    print(f"  Adapter       : {args.adapter_path}")
    print(f"  Total samples : {metrics.get('n_total', 0)}")
    print(f"  Closed samples: {metrics.get('n_closed', 0)}")
    print(f"  Open samples  : {metrics.get('n_open', 0)}")
    print("-" * 65)
    print(f"  Overall EM    : {metrics.get('overall_exact_match', 0):.4f}")
    print(f"  Closed Acc    : {metrics.get('closed_accuracy', 0):.4f}")
    print(f"  Closed F1     : {metrics.get('closed_f1', 0):.4f}")
    print(f"  Open EM       : {metrics.get('open_exact_match', 0):.4f}")
    print(f"  Open BLEU-1   : {metrics.get('open_bleu', 0):.4f}")
    print(f"  Open ROUGE-L  : {metrics.get('open_rouge_l', 0):.4f}")
    print(f"  Open F1       : {metrics.get('open_f1', 0):.4f}")
    if "open_bertscore" in metrics and metrics["open_bertscore"] is not None:
        print(f"  Open BERTScore: {metrics['open_bertscore']:.4f}")
        print(f"    (model: {metrics.get('bertscore_model', 'N/A')})")
    print("-" * 65)
    print(f"  Elapsed       : {metrics.get('elapsed_seconds', 0):.1f}s")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve Kaggle paths (VQA-RAD only — skip PathVQA fallback)
    output_dir = Path(args.output_dir)
    data_root = Path(args.data_root)

    # 1. Load retriever
    logger.info("Loading FAISS retriever from: %s", args.index_dir)
    retriever = FAISSRetriever().load(args.index_dir)

    # 2. Load model + LoRA adapter
    model, processor = load_model_and_processor(args)

    # 3. Load eval data
    df = pd.read_csv(args.eval_csv)
    logger.info("Loaded eval CSV: %d rows", len(df))
    if "row_id" not in df.columns:
        df["row_id"] = [f"eval_{i:06d}" for i in range(len(df))]
    if args.max_samples is not None:
        df = df.head(args.max_samples).reset_index(drop=True)
        logger.info("Limiting to %d samples (--max-samples)", args.max_samples)

    # 4. Run inference
    logger.info(
        "Starting evaluation | case=%s | top_k=%d | n=%d",
        args.case, args.top_k, len(df),
    )
    t0 = time.time()
    rows, predictions, references, question_types = run_inference(
        model, processor, retriever, df, data_root, args,
    )
    elapsed = time.time() - t0

    if not predictions:
        logger.error("No predictions generated. Check image paths or CSV format.")
        sys.exit(1)

    # 5. Compute metrics
    logger.info("Computing metrics over %d samples...", len(predictions))
    metrics = compute_metrics(predictions, references, question_types)

    # 5b. Compute BERTScore (open-ended only)
    if args.compute_bertscore:
        open_preds = [p for p, qt in zip(predictions, question_types) if qt == "open"]
        open_refs = [r for r, qt in zip(references, question_types) if qt == "open"]
        bs_result = compute_bertscore(
            open_preds, open_refs,
            model_type=args.bertscore_model,
            device=args.device,
        )
        metrics["open_bertscore"] = bs_result["bertscore_f1"]
        metrics["bertscore_precision"] = bs_result["bertscore_precision"]
        metrics["bertscore_recall"] = bs_result["bertscore_recall"]
        metrics["bertscore_model"] = bs_result["bertscore_model"]
    else:
        metrics["open_bertscore"] = None

    # 6. Save outputs
    save_outputs(output_dir, rows, metrics, retriever, args, elapsed)

    # 7. Print summary
    print_summary(metrics, args)

    logger.info("Evaluation complete. Results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
