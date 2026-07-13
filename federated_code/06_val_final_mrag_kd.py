#!/usr/bin/env python3
"""
06_val_final_mrag_kd.py

Final validation of the federated adapter on VQA-RAD test set.
Evaluates the global adapter from step 05 using pure mRAG (text-only retrieved context).
uv run federated_code/06_val_final_mrag_kd.py --mode 04_train_client_continue_mrag_kd --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2 --rag_k 3
"""

import os
import json
import time
import argparse
from pathlib import Path
import re as _re
from collections import Counter

# ── Windows DLL fix: import numpy/pandas BEFORE torch ──────────────────────
import numpy as np
import pandas as pd

import torch
import faiss
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

# Load medical embedding model for hallucination detection
similarity_model = SentenceTransformer('NeuML/pubmedbert-base-embeddings')

# BiomedCLIP model for mRAG retrieval
BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

def log(msg: str):
    print(f"[VAL-FINAL] {msg}", flush=True)

# ── mRAG Logic ───────────────────────────────────────────────────────────────
def build_mrag_content(query_image: Image.Image, question: str, question_type: str, context_list: list):
    """
    Construct a Qwen2-VL mRAG prompt:
      [query_image] + question text + context + answer prompt
      Retrieved context is text-only (no images)
    """
    content = []
    
    # 1) System and Image
    content.append({"type": "text", "text": "System:\nYou are a medical visual question answering assistant.\n\nImage:\n"})
    content.append({"type": "image", "image": query_image})
    
    # 2) Question and Question Type
    k = len(context_list)
    q_text = (
        f"\n\nQuestion:\n{question}\n\n"
        f"Question Type:\n{question_type}\n\n"
        f"Retrieved Medical Evidence (Top-{k} QA cases ranked by similarity score):\n\n"
    )
    content.append({"type": "text", "text": q_text})
    
    # 3) Retrieved text context (text-only)
    for case_text in context_list:
        content.append({"type": "text", "text": case_text + "\n"})
    
    # 4) Instruction and rules
    instruction = (
        "Instruction:\n"
        "You must adapt your reasoning based on the question type.\n\n"
        "- If the question is CLOSED-ENDED (Yes/No only):\n"
        "  - The final answer MUST be \"yes\" or \"no\"\n"
        "  - Focus primarily on the image\n\n"
        "- If the question is OPEN-ENDED:\n"
        "  - Use the image to answer the question\n\n"
        "General rules:\n"
        "- Do not output reasoning steps\n"
        "- Output only the final answer\n\n"
        "Answer:\n"
    )
    content.append({"type": "text", "text": instruction})
    return content

def _load_image_safe(path):
    """Load image with fallback to black image."""
    try:
        return Image.open(path).convert("RGB")
    except:
        return Image.new("RGB", (224, 224), color="black")

def get_biomedclip_model():
    """Load BiomedCLIP model for image-text embedding."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL)
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL)
    return model, preprocess, tokenizer

def compute_text_embedding(text, model, tokenizer):
    """Compute BiomedCLIP text embedding."""
    import open_clip
    text_tokens = tokenizer([text])
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    return text_features / text_features.norm(dim=-1, keepdim=True)

def perform_mrag_retrieval(query_text, train_texts, train_embeddings, rag_k, model, tokenizer):
    """Retrieve top-k similar texts using FAISS."""
    query_emb = compute_text_embedding(query_text, model, tokenizer)
    query_emb_np = query_emb.cpu().numpy()
    
    # FAISS search
    distances, indices = train_index.search(query_emb_np, rag_k)
    
    retrieved_contexts = [train_texts[i] for i in indices[0]]
    return retrieved_contexts

# ── Metrics ──────────────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    if not text or (isinstance(text, float) and np.isnan(text)): return ""
    text = str(text).lower().strip()
    text = _re.sub(r"[^\w\s]", "", text)
    text = _re.sub(r"\s+", " ", text).strip()
    return text

def compute_exact_match(pred: str, gt: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gt) else 0.0

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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="VLM model name from HuggingFace")
    parser.add_argument("--model_short", type=str, default="Qwen2.5-VL-3B-Instruct",
                        help="Short model name for directory naming")
    parser.add_argument("--rag_k", type=int, default=3,
                        help="Number of retrieved cases for mRAG context")
    parser.add_argument("--global_adapter_final", type=str, default=None,
                        help="Path to global adapter from step 05 (auto-generated if not provided)")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="Path to VQA-RAD test CSV (auto-generated if not provided)")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Path to VQA-RAD images (auto-generated if not provided)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (auto-generated if not provided)")
    args = parser.parse_args()

    # Auto-generate paths if not provided
    base = Path("experiments") / args.dataset / args.setting
    if args.setting == "noniid":
        base = base / f"alpha_{args.alpha}"
    base = base / f"k{args.k}" / args.model_short
    data_root = Path("processed_data")
    
    if args.global_adapter_final is None:
        args.global_adapter_final = str(base / "server" / f"global_adapter_{args.mode}")
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
    
    # Load BiomedCLIP for mRAG retrieval
    log("Loading BiomedCLIP model for retrieval...")
    biomedclip_model, _, biomedclip_tokenizer = get_biomedclip_model()
    biomedclip_model.eval()
    
    # Precompute text embeddings for test set (use test set as retrieval corpus)
    log("Precomputing text embeddings for retrieval...")
    test_texts = []
    for _, row in test_df.iterrows():
        text = f"Question: {row['question']}"
        test_texts.append(text)
    
    test_embeddings = []
    for text in tqdm(test_texts, desc="Computing embeddings"):
        emb = compute_text_embedding(text, biomedclip_model, biomedclip_tokenizer)
        test_embeddings.append(emb.cpu().numpy())
    
    test_embeddings = np.vstack(test_embeddings).astype('float32')
    
    # Build FAISS index
    global train_index
    embedding_dim = test_embeddings.shape[1]
    train_index = faiss.IndexFlatIP(embedding_dim)
    train_index.add(test_embeddings)
    log(f"Built FAISS index with {len(test_texts)} embeddings")
    
    log(f"Starting inference with rag_k={args.rag_k}...")
    t_start = time.time()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        question = row["question"]
        q_type_raw = str(row.get("question_type", "open")).lower()
        q_type = "CLOSED-ENDED" if q_type_raw == "closed" else "OPEN-ENDED"
        
        img_path = image_root / row["image_path"]
        query_image = _load_image_safe(img_path)
        
        # Perform mRAG retrieval (text-only)
        query_text = f"Question: {question}"
        retrieved_contexts = perform_mrag_retrieval(
            query_text, test_texts, test_embeddings, args.rag_k,
            biomedclip_model, biomedclip_tokenizer
        )
        
        # Build mRAG prompt with query image and text context
        content = build_mrag_content(query_image, question, q_type, retrieved_contexts)
        messages = [{"role": "user", "content": content}]
        
        # Only query image (retrieved contexts are text-only)
        all_images = [query_image]
        
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
    test_df.to_csv(csv_out, index=False)
    log(f"Saved predictions to {csv_out}")

    json_out = out_dir / "metrics.json"
    with open(json_out, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    log(f"Saved metrics to {json_out}")

if __name__ == "__main__":
    main()
