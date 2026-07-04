#!/usr/bin/env python3
"""
06_val_final_mrag_kd.py

Final validation of the federated adapter on VQA-RAD test set.
Evaluates the global adapter from step 05.
uv run federated_code/06_val_final_mrag_kd.py --mode 04_train_client_continue_mrag_kd --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2
"""

import os
import json
import time
import argparse
import random
from pathlib import Path
import re as _re

# ── Windows DLL fix: import numpy/pandas BEFORE torch ──────────────────────
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

# ── Set HuggingFace cache (dynamic relative to project root) ───────────────
project_root = Path(__file__).parent.parent
CACHE_DIR = project_root / ".hf_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hub")

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_SHORT = "Qwen2.5-VL-3B-Instruct"

# Load medical embedding model for hallucination detection
similarity_model = SentenceTransformer('NeuML/pubmedbert-base-embeddings')

def log(msg: str):
    print(f"[VAL-FINAL] {msg}", flush=True)

# ── ICL Logic ───────────────────────────────────────────────────────────────
def build_icl_content(query_image, question, question_type, examples):
    content = []
    content.append({"type": "text", "text": "System:\nYou are a medical visual question answering assistant.\n\n"})
    
    # Add few-shot examples
    for i, ex in enumerate(examples):
        content.append({"type": "text", "text": f"Example {i+1}:\n"})
        content.append({"type": "image", "image": ex["image"]})
        content.append({"type": "text", "text": f"Question: {ex['question']}\nQuestion Type: {ex['question_type']}\nAnswer: {ex['answer']}\n\n"})
    
    # Add query
    content.append({"type": "text", "text": "Now answer the following question:\n"})
    content.append({"type": "image", "image": query_image})
    content.append({"type": "text", "text": f"Question: {question}\nQuestion Type: {question_type}\n\n"})
    
    instruction = (
        "Instruction:\n"
        "You must adapt your reasoning based on the question type.\n\n"
        "- If the question is CLOSED-ENDED (Yes/No only):\n"
        "  • The final answer MUST be \"yes\" or \"no\"\n"
        "  • Focus primarily on the image\n\n"
        "- If the question is OPEN-ENDED:\n"
        "  • Use the image to answer the question\n\n"
        "General rules:\n"
        "- Do not output reasoning steps\n"
        "- Output only the final answer\n\n"
        "Answer:\n"
    )
    content.append({"type": "text", "text": instruction})
    return content

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
                        help="Script name suffix from step 05 (e.g., 04_train_client_continue_mrag_kd)")
    parser.add_argument("--dataset", type=str, default="VQA-RAD")
    parser.add_argument("--setting", type=str, choices=["iid", "noniid"], default="noniid")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--global_adapter_final", type=str, default=None,
                        help="Path to global adapter from step 05 (auto-generated if not provided)")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="Path to VQA-RAD test CSV (auto-generated if not provided)")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Path to VQA-RAD images (auto-generated if not provided)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (auto-generated if not provided)")
    parser.add_argument("--k_examples", type=int, default=3, help="Number of few-shot examples for ICL")
    args = parser.parse_args()

    # Auto-generate paths if not provided
    base = Path("experiments") / args.dataset / args.setting
    if args.setting == "noniid":
        base = base / f"alpha_{args.alpha}"
    base = base / f"k{args.k}" / MODEL_SHORT
    data_root = Path("processed_data")
    
    if args.global_adapter_final is None:
        args.global_adapter_final = str(base / "server" / f"global_adapter_{args.mode}_continue")
    if args.test_csv is None:
        args.test_csv = str(data_root / "VQA-RAD" / "centralized" / "test.csv")
    if args.image_root is None:
        args.image_root = str(data_root / "VQA-RAD" / "centralized")
    if args.out_dir is None:
        args.out_dir = str(base / f"final_val_{args.mode}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    test_df = pd.read_csv(args.test_csv)
    image_root = Path(args.image_root)
    log(f"Test samples: {len(test_df)}")
    
    log(f"Loading processor and base model {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, min_pixels=128*28*28, max_pixels=256*28*28, cache_dir=CACHE_DIR)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    if "qwen2.5" in MODEL_NAME.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    else:
        from transformers import Qwen2VLForConditionalGeneration as ModelClass

    base_model = ModelClass.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype, device_map="auto", cache_dir=CACHE_DIR)
    
    log(f"Loading global adapter final: {args.global_adapter_final}...")
    model = PeftModel.from_pretrained(base_model, args.global_adapter_final)
    model.eval()

    rouge_scr = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    predictions = []
    
    log(f"Starting inference with k={args.k_examples} few-shot examples...")
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
        
        # Sample k examples from test set (excluding current)
        available_indices = [i for i in range(len(test_df)) if i != idx]
        example_indices = random.sample(available_indices, min(args.k_examples, len(available_indices)))
        
        examples = []
        for ex_idx in example_indices:
            ex_row = test_df.iloc[ex_idx]
            ex_q_type_raw = str(ex_row.get("question_type", "open")).lower()
            ex_question_type = "CLOSED-ENDED" if ex_q_type_raw == "closed" else "OPEN-ENDED"
            ex_img_path = image_root / ex_row["image_path"]
            try:
                ex_image = Image.open(ex_img_path).convert("RGB")
            except:
                ex_image = Image.new("RGB", (224, 224), color="black")
            examples.append({
                "image": ex_image,
                "question": ex_row['question'],
                "question_type": ex_question_type,
                "answer": str(ex_row['answer'])
            })
        
        content = build_icl_content(query_image, question, q_type, examples)
        messages = [{"role": "user", "content": content}]
        
        # Collect all images
        all_images = [query_image] + [ex["image"] for ex in examples]
        
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=all_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                do_sample=False,
            )
            
        prompt_len = inputs["input_ids"].shape[1]
        gen_text = processor.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
        predictions.append(gen_text)
        
    t_end = time.time()
    total_time = t_end - t_start
    throughput = len(test_df) / total_time
    
    log(f"Inference time: {total_time:.2f}s | Throughput: {throughput:.2f} samples/sec")

    # Split predictions
    closed_preds, closed_gts = [], []
    open_preds, open_gts = [], []
    
    test_df["prediction"] = predictions

    for pred, row in zip(predictions, test_df.itertuples()):
        qt = str(row.question_type).lower()
        if qt == "closed":
            closed_preds.append(pred)
            closed_gts.append(str(row.answer))
        else:
            open_preds.append(pred)
            open_gts.append(str(row.answer))
            
    # Closed metrics
    log("Computing closed-ended metrics...")
    c_acc = float(np.mean([compute_exact_match(p, g) for p, g in zip(closed_preds, closed_gts)])) if closed_preds else 0.0
    c_em = c_acc # Exact match is same as accuracy for closed
    c_bleu = float(np.mean([compute_bleu(p, g) for p, g in zip(closed_preds, closed_gts)])) if closed_preds else 0.0
    c_rouge = float(np.mean([compute_rouge_l(p, g, rouge_scr) for p, g in zip(closed_preds, closed_gts)])) if closed_preds else 0.0
    c_bert = compute_bert_score(closed_preds, closed_gts) if closed_preds else 0.0

    # Open metrics
    log("Computing open-ended metrics...")
    o_acc = float(np.mean([compute_exact_match(p, g) for p, g in zip(open_preds, open_gts)])) if open_preds else 0.0
    o_em = o_acc
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
    test_df.to_csv(csv_out, index=False)
    log(f"Saved predictions to {csv_out}")

    json_out = out_dir / "metrics.json"
    with open(json_out, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    log(f"Saved metrics to {json_out}")

if __name__ == "__main__":
    main()
