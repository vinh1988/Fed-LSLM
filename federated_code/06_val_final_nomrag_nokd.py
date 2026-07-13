#!/usr/bin/env python3
"""
06_val_final_nomrag_nokd.py

Final validation using the global adapter from step 05 without RAG (no KD).
Uses Qwen2-VL for generation.
uv run federated_code/06_val_final_nomrag_nokd.py --mode 05_fedavg_final_nomrag_nokd --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2
"""

import os
import json
import time
import argparse
import random
from pathlib import Path
import re as _re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor
from peft import PeftModel

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score
from sentence_transformers import SentenceTransformer, util

# ── Cache & Globals ──────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
CACHE_DIR = str(project_root / ".hf_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")


# Load medical embedding model for hallucination detection
similarity_model = SentenceTransformer('NeuML/pubmedbert-base-embeddings')

def log(msg: str):
    print(f"[VAL-FINAL] {msg}", flush=True)

# ── Prompt Logic (no RAG) ───────────────────────────────────────────────────────
def build_prompt_content(query_image: Image.Image, question: str, question_type: str):
    """
    Construct a Qwen2-VL prompt without RAG:
      [query_image] + question text + answer prompt
    """
    content = []
    
    # 1) System and Image
    content.append({"type": "text", "text": "System:\nYou are a medical visual question answering assistant.\n\nImage:\n"})
    content.append({"type": "image", "image": query_image})
    
    # 2) Question and Question Type
    q_text = (
        f"\n\nQuestion:\n{question}\n\n"
        f"Question Type:\n{question_type}\n\n"
    )
    content.append({"type": "text", "text": q_text})
    
    # 3) Instruction and rules
    instruction = (
        "Instruction:\n"
        "You must adapt your reasoning based on the question type.\n\n"
        "- If the question is CLOSED-ENDED (Yes/No only):\n"
        "  - The final answer MUST be \"yes\" or \"no\"\n"
        "  - Focus primarily on the image\n\n"
        "- If the question is OPEN-ENDED:\n"
        "  - Use the image to answer the question\n"
        "  - Provide a concise and accurate answer\n\n"
        "General rules:\n"
        "- Do not output reasoning steps\n"
        "- Output only the final answer\n\n"
        "Answer:\n"
    )
    content.append({"type": "text", "text": instruction})
    
    return content

def compute_flexible_match(pred: str, gt: str) -> float:
    """
    More flexible matching for open-ended questions.
    Returns 1.0 if:
    - Exact match after normalization
    - Prediction is a substring of ground truth (e.g., "intestine" matches "the small intestines")
    - Ground truth is a substring of prediction (e.g., "the small intestines" matches "intestines")
    """
    norm_pred = _normalize(pred)
    norm_gt = _normalize(gt)
    
    # Exact match
    if norm_pred == norm_gt:
        return 1.0
    
    # Check if prediction is substring of ground truth
    if norm_pred in norm_gt:
        return 1.0
    
    # Check if ground truth is substring of prediction
    if norm_gt in norm_pred:
        return 1.0
    
    return 0.0

# ── Metrics ──────────────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    if not text or (isinstance(text, float) and np.isnan(text)): return ""
    text = str(text).lower().strip()
    text = _re.sub(r"[^\w\s]", "", text)
    text = _re.sub(r"\s+", " ", text).strip()
    return text

def compute_exact_match(pred: str, gt: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gt) else 0.0

def compute_bleu(pred: str, gt: str) -> float:
    ref = [_normalize(gt).split()]
    hyp = _normalize(pred).split()
    if not hyp: return 0.0
    return sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1)

def compute_rouge_l(pred: str, gt: str, scorer) -> float:
    scores = scorer.score(_normalize(gt), _normalize(pred))
    return scores["rougeL"].fmeasure

def compute_bert_score(preds: list, gts: list) -> float:
    if not preds: return 0.0
    # ensure non-empty
    preds = [p if p.strip() else "empty" for p in preds]
    gts = [g if g.strip() else "empty" for g in gts]
    P, R, F1 = bert_score.score(preds, gts, lang="en", verbose=False)
    return float(F1.mean())

def evaluate_fast_hallucination(qa_type: str, gt: str, prediction: str, threshold: float = 0.85) -> int:
    """
    Lightning-fast hallucination detection using local embedding similarity.
    
    Args:
        qa_type (str): 'close-ended' or 'open-ended'.
        gt (str): Ground truth string.
        prediction (str): Model prediction string.
        threshold (float): Similarity index above which text is deemed accurate.
                           0.85 is recommended for medical synonyms.
    """
    gt_clean = gt.strip().lower()
    pred_clean = prediction.strip().lower()

    # 1. Close-Ended (Exact matching remains the fastest method possible)
    if qa_type.lower() == 'close-ended':
        return 0 if gt_clean == pred_clean else 1

    # 2. Open-Ended (Vector Similarity Match)
    elif qa_type.lower() == 'open-ended':
        if gt_clean == pred_clean:
            return 0
            
        # Encode both sentences into medical vector embeddings
        emb_gt = similarity_model.encode(gt_clean, convert_to_tensor=True)
        emb_pred = similarity_model.encode(pred_clean, convert_to_tensor=True)
        
        # Calculate Cosine Similarity
        similarity_score = util.cos_sim(emb_gt, emb_pred).item()
        
        # If similarity is high, they are medical synonyms -> No Hallucination (0)
        # If similarity is low -> Hallucination (1)
        return 0 if similarity_score >= threshold else 1
    
    else:
        raise ValueError("qa_type must be 'close-ended' or 'open-ended'")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        help="Script name suffix from step 05 (e.g., 05_fedavg_final_nomrag_nokd)")
    parser.add_argument("--dataset", type=str, default="VQA-RAD")
    parser.add_argument("--setting", type=str, choices=["iid", "noniid"], default="noniid")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="VLM model name from HuggingFace")
    parser.add_argument("--model_short", type=str, default="Qwen2.5-VL-3B-Instruct",
                        help="Short model name for directory naming")
    parser.add_argument("--global_adapter_final", type=str, default=None,
                        help="Path to global adapter from step 05 (auto-generated if not provided)")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="Path to test CSV (auto-generated if not provided)")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Path to image root (auto-generated if not provided)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Path to output directory (auto-generated if not provided)")
    args = parser.parse_args()

    # Auto-generate paths if not provided
    base = Path("experiments") / args.dataset / args.setting
    if args.setting == "noniid":
        base = base / f"alpha_{args.alpha}"
    base = base / f"k{args.k}" / args.model_short
    
    if args.global_adapter_final is None:
        args.global_adapter_final = str(base / "server" / f"global_adapter_{args.mode}")
    
    data_root = Path("processed_data") / args.dataset
    if args.test_csv is None:
        args.test_csv = str(data_root / "centralized" / "test.csv")
    if args.image_root is None:
        args.image_root = str(data_root / "centralized")
    if args.out_dir is None:
        args.out_dir = str(base / f"final_val_{args.mode}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    test_df = pd.read_csv(args.test_csv)
    image_root = Path(args.image_root)
    log(f"Test samples: {len(test_df)}")
    
    log(f"Loading processor and base model {args.model_name}...")
    processor = AutoProcessor.from_pretrained(args.model_name, min_pixels=128*28*28, max_pixels=256*28*28, cache_dir=CACHE_DIR)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    if "qwen2.5" in args.model_name.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    else:
        from transformers import Qwen2VLForConditionalGeneration as ModelClass

    base_model = ModelClass.from_pretrained(args.model_name, torch_dtype=torch_dtype, device_map="auto", cache_dir=CACHE_DIR)
    
    log(f"Loading global adapter final: {args.global_adapter_final}...")
    adapter_path = str(Path(args.global_adapter_final).as_posix())
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    rouge_scr = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    predictions = []
    
    log(f"Starting inference without RAG...")
    t_start = time.time()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        question = row["question"]
        q_type_raw = str(row.get("question_type", "open")).lower()
        q_type = "CLOSED-ENDED" if q_type_raw == "closed" else "OPEN-ENDED"
        
        img_path = image_root / row["image_path"]
        try:
            query_image = Image.open(img_path).convert("RGB")
        except:
            query_image = Image.new("RGB", (224, 224), color="black")
        
        # Build prompt without RAG
        prompt_content = build_prompt_content(query_image, question, q_type)
        messages = [{"role": "user", "content": prompt_content}]
        
        all_images = [query_image]
        inputs = processor(text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                          images=all_images, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        predictions.append({
            "question": question,
            "question_type": q_type_raw,
            "answer": str(row["answer"]),
            "prediction": output_text.strip()
        })
    
    t_end = time.time()
    total_time = t_end - t_start
    throughput = len(test_df) / total_time
    
    log(f"Inference time: {total_time:.2f}s | Throughput: {throughput:.2f} samples/sec")

    # Split predictions
    closed_preds, closed_gts = [], []
    open_preds, open_gts = [], []
    
    for pred in predictions:
        qt = pred["question_type"].lower()
        if qt == "closed":
            closed_preds.append(pred["prediction"])
            closed_gts.append(pred["answer"])
        else:
            open_preds.append(pred["prediction"])
            open_gts.append(pred["answer"])
            
    # Closed metrics
    log("Computing closed-ended metrics...")
    c_acc = float(np.mean([compute_exact_match(p, g) for p, g in zip(closed_preds, closed_gts)])) if closed_preds else 0.0
    c_em = c_acc # Exact match is same as accuracy for closed
    c_bleu = float(np.mean([compute_bleu(p, g) for p, g in zip(closed_preds, closed_gts)])) if closed_preds else 0.0
    c_rouge = float(np.mean([compute_rouge_l(p, g, rouge_scr) for p, g in zip(closed_preds, closed_gts)])) if closed_preds else 0.0
    c_bert = compute_bert_score(closed_preds, closed_gts) if closed_preds else 0.0

    # Open metrics
    log("Computing open-ended metrics...")
    o_acc = float(np.mean([compute_flexible_match(p, g) for p, g in zip(open_preds, open_gts)])) if open_preds else 0.0
    o_em = float(np.mean([compute_exact_match(p, g) for p, g in zip(open_preds, open_gts)])) if open_preds else 0.0
    o_bleu = float(np.mean([compute_bleu(p, g) for p, g in zip(open_preds, open_gts)])) if open_preds else 0.0
    o_rouge = float(np.mean([compute_rouge_l(p, g, rouge_scr) for p, g in zip(open_preds, open_gts)])) if open_preds else 0.0
    o_bert = compute_bert_score(open_preds, open_gts) if open_preds else 0.0

    # Compute hallucination rate using embedding similarity
    log("Computing hallucination rate...")
    hallucination_count = 0
    total_count = 0
    
    # Closed-ended hallucinations
    for p, g in zip(closed_preds, closed_gts):
        hallucination_count += evaluate_fast_hallucination('close-ended', g, p)
        total_count += 1
    
    # Open-ended hallucinations
    for p, g in zip(open_preds, open_gts):
        hallucination_count += evaluate_fast_hallucination('open-ended', g, p)
        total_count += 1
    
    hallucination_rate = hallucination_count / total_count if total_count > 0 else 0.0

    metrics_payload = {
        "closed-ended": {
            "n_samples": len(closed_preds),
            "Accuracy": round(c_acc, 6)
        },
        "open-ended": {
            "n_samples": len(open_preds),
            "Accuracy": round(o_acc, 6),
            "Exact Match": round(o_em, 6),
            "BLEU": round(o_bleu, 6),
            "ROUGE-L": round(o_rouge, 6),
            "BERTScore": round(o_bert, 6)
        },
        "hallucination": {
            "Hallucination Rate": round(hallucination_rate, 6)
        },
        "performance": {
            "inference_time_s": round(total_time, 2),
            "throughput_samples_per_sec": round(throughput, 2)
        }
    }

    # Save
    csv_out = out_dir / "predictions.csv"
    pd.DataFrame(predictions).to_csv(csv_out, index=False)
    log(f"Saved predictions to {csv_out}")

    json_out = out_dir / "metrics.json"
    with open(json_out, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    log(f"Saved metrics to {json_out}")

if __name__ == "__main__":
    main()
