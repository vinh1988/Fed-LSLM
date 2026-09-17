#!/usr/bin/env python3
"""
val_mrag.py

mRAG validation of the adapter on SLAKE test set.
Evaluates the adapter with text-only retrieval (no retrieved images).
Uses BiomedCLIP for embedding and FAISS for retrieval.
"""

import os
import json
import time
import argparse
import random
import gc
import warnings

# Suppress transformers warnings
warnings.filterwarnings("ignore", message=".*Kwargs passed to.*")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from pathlib import Path

# ── Windows DLL fix: import numpy/pandas BEFORE torch ──────────────────────
import numpy as np
import pandas as pd

import torch
from PIL import Image
from tqdm import tqdm

import open_clip
import faiss
from transformers import AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score
from sentence_transformers import SentenceTransformer, util

# ── Set HuggingFace cache (dynamic relative to project root) ───────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent.resolve()
CACHE_DIR = PROJECT_ROOT / ".hf_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hub")



# Load medical embedding model for hallucination detection
similarity_model = SentenceTransformer('NeuML/pubmedbert-base-embeddings')

def log(msg: str):
    print(f"[RAG-VAL] {msg}", flush=True)

# Precompute embeddings
def precompute_embeddings(csv_path, data_dir, model, preprocess, tokenizer, device):
    df = pd.read_csv(csv_path)
    embeddings = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding {csv_path.name}"):
        img_path = data_dir / row['image_path']
        question = str(row['question'])
        
        # Load and preprocess image
        try:
            image = Image.open(img_path).convert('RGB')
            image_input = preprocess(image).unsqueeze(0).to(device)
        except Exception as e:
            log(f"Warning: failed to load image {img_path}: {e}. Using zero tensor.")
            image_input = torch.zeros(1, 3, 224, 224).to(device)
            
        # Tokenize question
        text_input = tokenizer([question]).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            
            # L2-normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Combine by concatenation
            combined = torch.cat([image_features, text_features], dim=-1)
            embeddings.append(combined.cpu().numpy()[0])
            
    return np.array(embeddings)

# Format retrieved documents (text portion)
def format_retrieved_knowledge(df, indices, scores=None):
    docs = []
    if scores is None:
        scores = [0.0] * len(indices)
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        row = df.iloc[idx]
        docs.append(f"Top-{rank}:\nScore: {score:.4f}\nQuestion: {row['question']}\nAnswer: {row['answer']}\n")
    return docs

# ── mRAG Prompt Logic (text-only retrieval) ────────────────────────────────────
def build_mrag_content(query_image, question, question_type, body_part, modality, objects, context_list):
    """
    Construct a SmolVLM mRAG prompt for SLAKE:
      [query_image] + question + medical context + retrieved text QA pairs + answer prompt
      Note: Retrieved images are NOT included in the prompt (text-only retrieval)
    """
    content = []
    
    # 1) System and Image
    content.append({"type": "text", "text": "System:\nYou are a medical visual question answering assistant specializing in medical image analysis.\n\nImage:\n"})
    content.append({"type": "image", "image": query_image})
    
    # 2) Question and Question Type
    k = len(context_list)
    q_text = (
        f"\n\nQuestion:\n{question}\n\n"
        f"Question Type:\n{question_type}\n\n"
    )
    content.append({"type": "text", "text": q_text})
    
    # 3) Medical Context
    # Format objects summary from JSON
    objects_summary = ""
    if objects:
        try:
            objects_list = json.loads(objects)
            if isinstance(objects_list, list) and len(objects_list) > 0:
                object_names = [obj.get('name', 'unknown') for obj in objects_list]
                objects_summary = f"{len(objects_list)} objects detected: {', '.join(object_names)}"
        except:
            objects_summary = ""
    
    context_text = (
        "Medical Context:\n"
        f"- Body Part: {body_part}\n"
        f"- Modality: {modality}\n"
    )
    if objects_summary:
        context_text += f"- Detected Objects: {objects_summary}\n"
    context_text += "\n"
    
    content.append({"type": "text", "text": context_text})
    
    # 4) Retrieved text context (QA pairs only, no images)
    if context_list:
        rag_text = f"Retrieved Medical Evidence (Top-{k} QA cases ranked by similarity score):\n\n"
        content.append({"type": "text", "text": rag_text})
        for case_text in context_list:
            content.append({"type": "text", "text": case_text + "\n"})
    
    # 5) Instruction and rules
    instruction = (
        "Instruction:\n"
        "You must adapt your reasoning based on the question type.\n\n"
        "- If the question is CLOSED-ENDED (Yes/No only):\n"
        "  - The final answer MUST be \"yes\" or \"no\"\n"
        "  - Use the medical context and detected objects to inform your decision\n"
        "  - Focus primarily on the image\n"
        "  - Use retrieved QA cases only if they strongly match the visual evidence\n\n"
        "- If the question is OPEN-ENDED:\n"
        "  - Use the image, body part, modality, and detected objects to provide an accurate answer\n"
        "  - Consider the imaging modality when interpreting findings\n"
        "  - Use retrieved QA cases to improve reasoning and completeness\n\n"
        "General rules:\n"
        "- Retrieved cases are ranked by similarity score (higher score = more relevant)\n"
        "- Ignore low-score or irrelevant retrieved cases\n"
        "- Do not output reasoning steps\n"
        "- Output only the final answer\n\n"
        "Answer:\n"
    )
    content.append({"type": "text", "text": instruction})
    
    return content


# ── Metrics ──────────────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation, strip whitespace."""
    import string
    text = text.lower().strip()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = ' '.join(text.split())  # Remove extra whitespace
    return text

def compute_exact_match(prediction, ground_truth):
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0

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

def compute_bleu(prediction, ground_truth):
    ref = [ground_truth.strip().lower().split()]
    hyp = prediction.strip().lower().split()
    smooth = SmoothingFunction().method1
    return sentence_bleu(ref, hyp, smoothing_function=smooth)

def compute_rouge_l(prediction, ground_truth, scorer):
    scores = scorer.score(ground_truth.strip().lower(), prediction.strip().lower())
    return scores['rougeL'].fmeasure

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
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to centralized dataset folder")
    parser.add_argument("--adapter_dir", type=str, default=None,
                        help="Path to centralized adapter directory")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct",
                        help="Base VLM model path/name on HF")
    parser.add_argument("--biomed_model_name", type=str, default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                        help="BiomedCLIP model checkpoint")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of retrieved cases for RAG")
    parser.add_argument("--train_csv", type=str, default=None,
                        help="Path to SLAKE train CSV")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="Path to SLAKE test CSV")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Path to SLAKE images")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (defaults to adapter_dir / val)")
    args = parser.parse_args()

    # Dynamic default paths relative to PROJECT_ROOT
    model_short = args.model_name.split("/")[-1]
    
    # Resolve dynamic data_dir and extract dataset/setting
    if args.data_dir is None:
        args.data_dir = str(PROJECT_ROOT / "processed_data" / "Slake" / "centralized")
    
    data_path = Path(args.data_dir)
    parts = data_path.parts
    if len(parts) >= 2:
        dataset = parts[-2]
        setting = parts[-1]
    else:
        dataset = "Slake"
        setting = "centralized"
    
    if args.adapter_dir is None:
        args.adapter_dir = str(PROJECT_ROOT / "experiments" / dataset / setting / model_short / "mrag" / "best_lora_adapter")
    if args.train_csv is None:
        args.train_csv = str(data_path / "train.csv")
    if args.test_csv is None:
        args.test_csv = str(data_path / "test.csv")
    if args.image_root is None:
        args.image_root = str(data_path)
    if args.out_dir is None:
        args.out_dir = str(Path(args.adapter_dir) / "val")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    test_df = pd.read_csv(args.test_csv)
    image_root = Path(args.image_root)
    
    # Limit to first 10 samples for quick testing
    # test_df = test_df.head(10)
    
    log(f"Test samples: {len(test_df)}")

    # ----------------------------------------------------
    # Step 1: Precompute Embeddings using BiomedCLIP
    # ----------------------------------------------------
    train_csv = Path(args.train_csv)
    train_cache = out_dir / "train_biomed_embeddings.npy"
    test_cache = out_dir / "test_biomed_embeddings.npy"
    
    if train_cache.exists() and test_cache.exists():
        log("Loading precomputed BiomedCLIP embeddings from cache...")
        train_features = np.load(train_cache)
        test_features = np.load(test_cache)
    else:
        log("Loading BiomedCLIP model to compute embeddings...")
        biomed_model, preprocess = open_clip.create_model_from_pretrained(args.biomed_model_name)
        biomed_tokenizer = open_clip.get_tokenizer(args.biomed_model_name)
        biomed_model.to(device)
        biomed_model.eval()
        
        log("Computing train embeddings...")
        train_features = precompute_embeddings(train_csv, image_root, biomed_model, preprocess, biomed_tokenizer, device)
        np.save(train_cache, train_features)
        
        log("Computing test embeddings...")
        test_features = precompute_embeddings(Path(args.test_csv), image_root, biomed_model, preprocess, biomed_tokenizer, device)
        np.save(test_cache, test_features)
        
        # Free GPU memory completely
        del biomed_model
        torch.cuda.empty_cache()
        gc.collect()
        log("Cleared BiomedCLIP from memory.")
        
    # ----------------------------------------------------
    # Step 2: Build FAISS Index and Perform Retrieval
    # ----------------------------------------------------
    log("Building FAISS index on train features...")
    # Using L2-normalized vectors and Inner Product for Cosine Similarity search
    def l2_norm_rows(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms == 0, 1, norms)
        
    train_features = l2_norm_rows(train_features)
    test_features = l2_norm_rows(test_features)
    
    index = faiss.IndexFlatIP(1024)  # 1024 dimensions
    index.add(train_features)
    
    log(f"Performing RAG retrieval (k={args.k})...")
    # For test data, we retrieve top-k closest from training set
    test_contexts = []
    D_test, I_test = index.search(test_features, args.k)
    train_df = pd.read_csv(train_csv)
    for idx in range(len(test_df)):
        test_contexts.append(format_retrieved_knowledge(train_df, I_test[idx], D_test[idx]))
        
    log("RAG retrieval contexts (text-only) prepared successfully.")

    # ── Step 3: Load VLM and Adapter ──────────────────────────────────────────
    log(f"Loading processor and base model {args.model_name}...")
    processor = AutoProcessor.from_pretrained(args.model_name, min_pixels=128*28*28, max_pixels=256*28*28, cache_dir=CACHE_DIR)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    if "smolvlm" in args.model_name.lower():
        from transformers import AutoModelForImageTextToText as ModelClass
    elif "qwen2.5" in args.model_name.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    else:
        from transformers import Qwen2VLForConditionalGeneration as ModelClass

    # Skip quantization for Idefics3/SmolVLM due to compatibility issues
    if "smolvlm" in args.model_name.lower() or "idefics3" in args.model_name.lower():
        log(f"Skipping quantization for {args.model_name} (Idefics3/SmolVLM)")
        bnb_config = None
    else:
        # 4-bit quantization to drastically reduce memory footprint
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True
        )

    base_model = ModelClass.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        cache_dir=CACHE_DIR
    )
    
    log(f"Loading adapter: {args.adapter_dir}...")
    adapter_path = str(Path(args.adapter_dir).as_posix())
    model = PeftModel.from_pretrained(base_model, adapter_path, is_local=True)
    model.eval()

    rouge_scr = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    predictions = []
    
    log(f"Starting inference...")
    t_start = time.time()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        question = row["question"]
        q_type_raw = str(getattr(row, "question_type", "open")).lower()
        q_type = "CLOSED-ENDED" if q_type_raw == "closed" else "OPEN-ENDED"
        
        # Load SLAKE metadata
        body_part = row.get('body_part', '')
        modality = row.get('modality', '')
        objects = row.get('objects', '')
        
        # Get retrieved context for this sample
        context = test_contexts[idx]
        
        img_path = image_root / row["image_path"]
        try:
            query_image = Image.open(img_path).convert("RGB")
        except:
            query_image = Image.new("RGB", (224, 224), color="black")
        
        mrag_content = build_mrag_content(query_image, question, q_type, body_part, modality, objects, context)
        messages = [{"role": "user", "content": mrag_content}]
        
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
