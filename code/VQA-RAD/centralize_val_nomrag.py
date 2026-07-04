#!/usr/bin/env python3
"""
centralize_val_nomrag.py

Final validation of the centralized non-RAG adapter on VQA-RAD test set.
Evaluates the model trained via centralize_train_nomrag.py.
"""

import os
import json
import time
import argparse
import random
import gc
from pathlib import Path
import re as _re

# ── Windows DLL fix: import numpy/pandas BEFORE torch ──────────────────────
import numpy as np
import pandas as pd

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score
from sentence_transformers import SentenceTransformer, util

# ── Set HuggingFace cache (dynamic relative to project root) ───────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()
CACHE_DIR = PROJECT_ROOT / ".hf_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hub")

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_SHORT = "Qwen2.5-VL-3B-Instruct"

# Load medical embedding model for hallucination detection
similarity_model = SentenceTransformer('NeuML/pubmedbert-base-embeddings')

def log(msg: str):
    print(f"[VAL-NOMRAG] {msg}", flush=True)

# ── Zero-Shot Prompt Logic ───────────────────────────────────────────────────
def build_zeroshot_content(query_image, question, question_type):
    """
    Construct a Qwen2-VL zero-shot prompt:
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
    preds = [p if p.strip() else "empty" for p in preds]
    gts = [g if g.strip() else "empty" for g in gts]
    P, R, F1 = bert_score.score(preds, gts, lang="en", verbose=False)
    return float(F1.mean())

def evaluate_fast_hallucination(qa_type: str, gt: str, prediction: str, threshold: float = 0.85) -> int:
    gt_clean = gt.strip().lower()
    pred_clean = prediction.strip().lower()

    if qa_type.lower() == 'close-ended':
        return 0 if gt_clean == pred_clean else 1

    elif qa_type.lower() == 'open-ended':
        if gt_clean == pred_clean:
            return 0
            
        emb_gt = similarity_model.encode(gt_clean, convert_to_tensor=True)
        emb_pred = similarity_model.encode(pred_clean, convert_to_tensor=True)
        similarity_score = util.cos_sim(emb_gt, emb_pred).item()
        return 0 if similarity_score >= threshold else 1
    
    else:
        raise ValueError("qa_type must be 'close-ended' or 'open-ended'")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=str, default=None,
                        help="Path to centralized non-RAG adapter directory")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="Path to VQA-RAD test CSV")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Path to VQA-RAD images")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (defaults to adapter_dir / val)")
    args = parser.parse_args()

    # Dynamic default paths relative to PROJECT_ROOT
    if args.adapter_dir is None:
        args.adapter_dir = str(PROJECT_ROOT / "experiments" / "VQA-RAD" / "centralized" / "Qwen2.5-VL-3B-Instruct" / "nomrag")
    if args.test_csv is None:
        args.test_csv = str(PROJECT_ROOT / "processed_data" / "VQA-RAD" / "centralized" / "test.csv")
    if args.image_root is None:
        args.image_root = str(PROJECT_ROOT / "processed_data" / "VQA-RAD" / "centralized")
    if args.out_dir is None:
        args.out_dir = str(Path(args.adapter_dir) / "val")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    test_df = pd.read_csv(args.test_csv)
    image_root = Path(args.image_root)
    log(f"Test samples: {len(test_df)}")

    # ── Step 1: Load VLM and Adapter ──────────────────────────────────────────
    log(f"Loading processor and base model {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, min_pixels=128*28*28, max_pixels=256*28*28, cache_dir=CACHE_DIR)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    if "qwen2.5" in MODEL_NAME.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    else:
        from transformers import Qwen2VLForConditionalGeneration as ModelClass

    # 4-bit quantization to drastically reduce memory footprint
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True
    )

    base_model = ModelClass.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        cache_dir=CACHE_DIR
    )
    
    log(f"Loading adapter: {args.adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    rouge_scr = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    predictions = []
    
    log("Starting inference (Zero-Shot fine-tuned model)...")
    t_start = time.time()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        question = row["question"]
        q_type_raw = str(getattr(row, "question_type", "open")).lower()
        q_type = "CLOSED-ENDED" if q_type_raw == "closed" else "OPEN-ENDED"
        
        img_path = image_root / row["image_path"]
        try:
            query_image = Image.open(img_path).convert("RGB")
        except:
            query_image = Image.new("RGB", (224, 224), color="black")
        
        zeroshot_content = build_zeroshot_content(query_image, question, q_type)
        messages = [{"role": "user", "content": zeroshot_content}]
        
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[query_image], return_tensors="pt")
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
        qt = str(getattr(row, "question_type", "open")).lower()
        if qt == "closed":
            closed_preds.append(pred)
            closed_gts.append(str(row.answer))
        else:
            open_preds.append(pred)
            open_gts.append(str(row.answer))
            
    # Closed metrics
    log("Computing closed-ended metrics...")
    c_acc = float(np.mean([compute_exact_match(p, g) for p, g in zip(closed_preds, closed_gts)])) if closed_preds else 0.0
    c_em = c_acc 
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
    
    for p, g in zip(closed_preds, closed_gts):
        hallucination_count += evaluate_fast_hallucination('close-ended', g, p)
        total_count += 1
    
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
