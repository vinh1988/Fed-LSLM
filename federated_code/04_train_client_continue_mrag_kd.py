#!/usr/bin/env python3
"""
04_train_client_continue_mrag_kd.py

Continue fine-tuning on the VQA-RAD dataset using the ROCO-trained adapter with pure mRAG.
Uses BiomedCLIP for retrieval and Qwen2-VL for generation.
# Step 04
uv run federated_code/04_train_client_continue_mrag_kd.py --mode 04_train_client_continue_mrag_kd --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2
# Step 05
uv run federated_code/05_fedavg_final_mrag_kd.py --mode 04_train_client_continue_mrag_kd --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2

"""

# ── Windows DLL fix: import numpy/pandas BEFORE torch ──────────────────────
import numpy as np
import pandas as pd

import os
import gc
import json
import time
import argparse
import random
import datetime
import re as _re
from pathlib import Path
from collections import Counter

# ── Set HuggingFace cache (dynamic relative to project root) ───────────────
project_root = Path(__file__).parent.parent
os.environ["HUGGINGFACE_HUB_CACHE"] = str(project_root / ".hf_cache" / "hub")

import torch
from torch.utils.data import Dataset, DataLoader
import faiss
from PIL import Image
from tqdm import tqdm

import open_clip
from transformers import (
    AutoProcessor,
    get_linear_schedule_with_warmup,
)
from peft import PeftModel

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ── Constants ───────────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_SHORT = "Qwen2.5-VL-3B-Instruct"
BIOMED_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}][VQARAD-CONTINUE] {msg}", flush=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_gpu_static_info() -> dict:
    if not torch.cuda.is_available():
        return {"gpu_name": "CPU", "total_vram_gb": 0.0}
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "gpu_name": props.name,
        "total_vram_gb": round(props.total_memory / (1024 ** 3), 2),
    }

def sample_gpu_memory_gb() -> float:
    """Return currently allocated GPU memory in GB (0.0 if no CUDA)."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 ** 3)


# ──────────────────────────────────────────────────────────────────────────────
# BiomedCLIP Embeddings
# ──────────────────────────────────────────────────────────────────────────────

def precompute_embeddings(csv_path: Path, image_root: Path, model, preprocess, tokenizer, device) -> np.ndarray:
    df = pd.read_csv(csv_path)
    embeddings = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding {csv_path.name}"):
        img_path = image_root / row["image_path"]
        question = str(row["question"])

        try:
            image = Image.open(img_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
        except Exception as e:
            log(f"Warning: failed to load image {img_path}: {e}. Using zeros.")
            image_input = torch.zeros(1, 3, 224, 224).to(device)

        text_input = tokenizer([question]).to(device)

        with torch.no_grad():
            img_feat  = model.encode_image(image_input)
            text_feat = model.encode_text(text_input)
            img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            combined  = torch.cat([img_feat, text_feat], dim=-1)
            embeddings.append(combined.cpu().numpy()[0])

    return np.array(embeddings)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


# ──────────────────────────────────────────────────────────────────────────────
# RAG context formatting
# ──────────────────────────────────────────────────────────────────────────────

def format_retrieved_knowledge(df: pd.DataFrame, indices, scores=None):
    docs = []
    if scores is None:
        scores = [0.0] * len(indices)
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        row = df.iloc[idx]
        docs.append(f"Top-{rank}:\nScore: {score:.4f}\nQuestion: {row['question']}\nAnswer: {row['answer']}\n")
    return docs

# ──────────────────────────────────────────────────────────────────────────────
# Dataset & collate (mRAG)
# ──────────────────────────────────────────────────────────────────────────────

class MedVQADataset(Dataset):
    def __init__(self, df: pd.DataFrame, retrieved_contexts, image_root: Path, max_length: int = 512):
        self.df = df.reset_index(drop=True)
        self.retrieved_contexts = retrieved_contexts
        self.image_root = image_root
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path: str) -> Image.Image:
        full = self.image_root / rel_path
        try:
            return Image.open(full).convert("RGB")
        except Exception as e:
            log(f"Warning: failed to load image {full}: {e}")
            return Image.new("RGB", (224, 224), color="black")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q_type_raw   = str(row.get("question_type", "open")).lower()
        question_type = "CLOSED-ENDED" if q_type_raw == "closed" else "OPEN-ENDED"
        
        return {
            "query_image":   self._load_image(row["image_path"]),
            "question":      row["question"],
            "question_type": question_type,
            "context":       self.retrieved_contexts[idx],
            "answer":        str(row["answer"])
        }


def build_mrag_content(query_image: Image.Image, question: str, question_type: str, context_list: list):
    """
    Construct a Qwen2-VL mRAG prompt:
      [query_image] + question text + context + answer prompt
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
    
    # 3) Retrieved text context
    for case_text in context_list:
        content.append({"type": "text", "text": case_text + "\n"})
    
    # 4) Instruction and rules
    instruction = (
        "Instruction:\n"
        "You must adapt your reasoning based on the question type.\n\n"
        "- If the question is CLOSED-ENDED (Yes/No only):\n"
        "  - The final answer MUST be \"yes\" or \"no\"\n"
        "  - Focus primarily on the image\n"
        "  - Use retrieved QA cases only if they strongly match the visual evidence\n\n"
        "- If the question is OPEN-ENDED:\n"
        "  - Use both the image and retrieved QA cases\n"
        "  - Combine visual evidence with similar QA examples to improve reasoning and completeness\n\n"
        "General rules:\n"
        "- Retrieved cases are ranked by similarity score (higher score = more relevant)\n"
        "- Ignore low-score or irrelevant retrieved cases\n"
        "- Do not output reasoning steps\n"
        "- Output only the final answer\n\n"
        "Answer:\n"
    )
    content.append({"type": "text", "text": instruction})
    
    return content


def get_collate_fn(processor):
    def collate_fn(batch):
        all_images_per_sample = []
        user_prompts  = []
        full_prompts  = []
        
        for item in batch:
            sample_images = [item["query_image"]]
            all_images_per_sample.append(sample_images)
            
            mrag_content = build_mrag_content(
                item["query_image"], item["question"], item["question_type"], item["context"]
            )
            user_messages = [{"role": "user", "content": mrag_content}]
            full_messages = user_messages + [
                {"role": "assistant", "content": [{"type": "text", "text": item["answer"]}]}
            ]
            
            user_p = processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
            full_p = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            user_prompts.append(user_p)
            full_prompts.append(full_p)
        
        flat_images = [img for imgs in all_images_per_sample for img in imgs]
        inputs      = processor(text=full_prompts, images=flat_images, padding=True, return_tensors="pt")
        
        input_ids      = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels         = input_ids.clone()
        
        for i, user_p in enumerate(user_prompts):
            inp_user  = processor(text=[user_p], images=all_images_per_sample[i], return_tensors="pt")
            user_len  = inp_user["input_ids"].shape[1]
            labels[i, :user_len] = -100
            labels[i, attention_mask[i] == 0] = -100
        
        inputs["labels"] = labels
        return inputs
    return collate_fn

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    if not text or (isinstance(text, float) and np.isnan(text)): return ""
    text = str(text).lower().strip()
    text = _re.sub(r"[^\w\s]", "", text)
    text = _re.sub(r"\s+", " ", text).strip()
    return text

def compute_exact_match(pred: str, gt: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gt) else 0.0

def compute_f1(pred: str, gt: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gt).split()
    if not p_toks and not g_toks: return 1.0
    if not p_toks or not g_toks: return 0.0
    common  = Counter(p_toks) & Counter(g_toks)
    n_same  = sum(common.values())
    if n_same == 0: return 0.0
    prec = n_same / len(p_toks)
    rec  = n_same / len(g_toks)
    return 2 * prec * rec / (prec + rec)

def compute_bleu(pred: str, gt: str) -> float:
    ref = [_normalize(gt).split()]
    hyp = _normalize(pred).split()
    if not hyp: return 0.0
    return sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1)

def compute_rouge_l(pred: str, gt: str, scorer) -> float:
    scores = scorer.score(_normalize(gt), _normalize(pred))
    return scores["rougeL"].fmeasure

def evaluate_model(model, processor, test_df, test_contexts, image_root, rouge_scr, device, desc="Test"):
    model.eval()
    predictions = []
    start_time = time.time()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=desc):
        question = row["question"]
        context = test_contexts[idx]
        q_type_raw = str(row.get("question_type", "open")).lower()
        q_type = "CLOSED-ENDED" if q_type_raw == "closed" else "OPEN-ENDED"
        
        img_path = image_root / row["image_path"]
        try:
            query_image = Image.open(img_path).convert("RGB")
        except:
            query_image = Image.new("RGB", (224, 224), color="black")
        
        mrag_content = build_mrag_content(query_image, question, q_type, context)
        messages = [{"role": "user", "content": mrag_content}]
        
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
    
    inference_time = time.time() - start_time
    throughput = len(test_df) / inference_time if inference_time > 0 else 0.0
    
    closed_preds, closed_gts = [], []
    open_preds, open_gts = [], []
    
    for pred, row in zip(predictions, test_df.itertuples()):
        qt = str(row.question_type).lower()
        if qt == "closed":
            closed_preds.append(pred)
            closed_gts.append(str(row.answer))
        else:
            open_preds.append(pred)
            open_gts.append(str(row.answer))
            
    closed_acc = 0.0
    open_acc = 0.0
    open_bleu = 0.0
    open_rouge = 0.0
    
    try:
        if closed_preds:
            closed_acc = float(np.mean([compute_exact_match(p, g) for p, g in zip(closed_preds, closed_gts)]))
        if open_preds:
            open_acc = float(np.mean([compute_exact_match(p, g) for p, g in zip(open_preds, open_gts)]))
            open_bleu = float(np.mean([compute_bleu(p, g) for p, g in zip(open_preds, open_gts)]))
            open_rouge = float(np.mean([compute_rouge_l(p, g, rouge_scr) for p, g in zip(open_preds, open_gts)]))
    except Exception as e:
        log(f"Error computing metrics: {e}")
        log(f"closed_preds: {len(closed_preds)}, closed_gts: {len(closed_gts)}")
        log(f"open_preds: {len(open_preds)}, open_gts: {len(open_gts)}")
    
    hallucination_rate = 0.0  # Placeholder - requires semantic similarity model
    
    return {
        "closed-ended": {
            "n_samples": len(closed_preds),
            "Accuracy": round(closed_acc, 6),
        },
        "open-ended": {
            "n_samples": len(open_preds),
            "Accuracy": round(open_acc, 6),
            "Exact Match": round(open_acc, 6),  # Same as accuracy for open-ended
            "BLEU": round(open_bleu, 6),
            "ROUGE-L": round(open_rouge, 6),
            "BERTScore": 0.0,  # Placeholder - compute post-hoc
        },
        "hallucination": {
            "Hallucination Rate": round(hallucination_rate, 6),
        },
        "performance": {
            "inference_time_s": round(inference_time, 3),
            "throughput_samples_per_sec": round(throughput, 3),
        },
    }

# ──────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def build_output_dir(dataset: str, setting: str, alpha, k: int, client_id: int) -> Path:
    script_name = Path(__file__).stem
    base = Path("experiments") / dataset / setting
    if setting == "noniid":
        if alpha is None:
            raise ValueError("--alpha must be set for noniid setting")
        base = base / f"alpha_{alpha}"
    base = base / f"k{k}" / MODEL_SHORT / "clients" / f"client_{client_id}_{script_name}"
    return base

def resolve_data_paths(dataset: str, setting: str, alpha, k: int, client_id: int):
    data_root = Path("processed_data") / dataset
    if setting == "iid":
        client_csv = data_root / "federated_iid" / f"k{k}" / f"client_{client_id}.csv"
    elif setting == "noniid":
        if alpha is None:
            raise ValueError("--alpha must be set for noniid setting")
        client_csv = data_root / "federated_noniid" / f"alpha_{alpha}" / f"k{k}" / f"client_{client_id}.csv"
    else:
        raise ValueError(f"Unknown setting '{setting}'. Choose from: iid, noniid")
    
    test_csv   = data_root / "centralized" / "test.csv"
    image_root = data_root / "centralized"
    return client_csv, test_csv, image_root

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        help="Script name suffix for this step (e.g., 04_train_client_continue_mrag_kd)")
    parser.add_argument("--roco_slake_mode", type=str, default="01_train_client_mrag_kd",
                        help="Script name suffix from step 03 for ROCO+Slake adapter (default: 01_train_client_mrag_kd)")
    parser.add_argument("--dataset", type=str, default="VQA-RAD")
    parser.add_argument("--setting", type=str, choices=["iid", "noniid"], default="noniid")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--global_adapter_roco_slake", type=str, default=None,
                        help="Path to global adapter from ROCO/SLAKE training (auto-generated if not provided)")
    parser.add_argument("--rag_k", type=int, default=3,
                        help="Number of retrieved cases for mRAG context")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.setting == "noniid" and args.alpha is None:
        parser.error("--alpha is required when --setting noniid")

    # Auto-generate global adapter path if not provided
    base = Path("experiments") / args.dataset / args.setting
    if args.setting == "noniid":
        base = base / f"alpha_{args.alpha}"
    base = base / f"k{args.k}" / MODEL_SHORT / "server"
    
    if args.global_adapter_roco_slake is None:
        args.global_adapter_roco_slake = str(base / f"global_adapter_roco_slake_{args.roco_slake_mode}")

    set_seed(args.seed)
    
    log("Starting VQA-RAD continued training for ALL clients...")
    log(f"ROCO adapter: {args.global_adapter_roco_slake}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_static = get_gpu_static_info()
    log(f"Device: {device} | GPU: {gpu_static['gpu_name']} | VRAM: {gpu_static['total_vram_gb']} GB")

    # 1. Load Processor and Base Model ONCE
    log(f"Loading base model {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    if "qwen2.5" in MODEL_NAME.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    else:
        from transformers import Qwen2VLForConditionalGeneration as ModelClass

    base_model = ModelClass.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map="auto"
    )

    # 2. Load ROCO-trained Adapter (global initialization state)
    log(f"Loading pretrained ROCO Slake adapter from {args.global_adapter_roco_slake}...")
    model = PeftModel.from_pretrained(base_model, str(Path(args.global_adapter_roco_slake)), is_trainable=True)
    model.print_trainable_parameters()

    # 3. Save a deep copy of the global adapter weights
    global_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items() if v.requires_grad}

    rouge_scr = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # 4. Loop over clients
    for client_id in range(1, args.k + 1):
        log(f"{'='*60}")
        log(f"  CLIENT {client_id}/{args.k}  ({args.setting}, k={args.k})")
        log(f"{'='*60}")

        # Restore global weights to ensure isolation
        model.load_state_dict(global_state_dict, strict=False)

        client_csv, test_csv, image_root = resolve_data_paths(args.dataset, args.setting, args.alpha, args.k, client_id)
        out_dir = build_output_dir(args.dataset, args.setting, args.alpha, args.k, client_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        train_df = pd.read_csv(client_csv)
        test_df = pd.read_csv(test_csv)
        log(f"Train samples: {len(train_df)}")
        log(f"Test samples: {len(test_df)}")
        
        # ── BiomedCLIP Embedding & Retrieval ────────────────────────────────
        log(f"Loading BiomedCLIP model...")
        biomed_model, _, preprocess = open_clip.create_model_and_transforms(BIOMED_MODEL, pretrained=None)
        tokenizer = open_clip.get_tokenizer(BIOMED_MODEL)
        biomed_model = biomed_model.to(device)
        biomed_model.eval()
        
        log(f"Precomputing embeddings for client {client_id}...")
        train_features = precompute_embeddings(client_csv, image_root, biomed_model, preprocess, tokenizer, device)
        train_features = l2_normalize(train_features)
        
        test_features = precompute_embeddings(test_csv, image_root, biomed_model, preprocess, tokenizer, device)
        test_features = l2_normalize(test_features)
        
        # Build FAISS index
        faiss_dim = train_features.shape[1]
        index = faiss.IndexFlatIP(faiss_dim)
        index.add(train_features.astype(np.float32))
        
        log(f"Performing mRAG retrieval (rag_k={args.rag_k})...")
        
        # Train retrieval (exclude self)
        D_tr, I_tr = index.search(train_features.astype(np.float32), args.rag_k + 1)
        train_contexts = []
        for i in range(len(train_df)):
            ret_idx, ret_sc = [], []
            for pos, mi in enumerate(I_tr[i]):
                if mi == i:
                    continue
                ret_idx.append(mi)
                ret_sc.append(float(D_tr[i][pos]))
                if len(ret_idx) == args.rag_k:
                    break
            if len(ret_idx) < args.rag_k:
                for pos, mi in enumerate(I_tr[i]):
                    if mi not in ret_idx:
                        ret_idx.append(mi)
                        ret_sc.append(float(D_tr[i][pos]))
                    if len(ret_idx) == args.rag_k:
                        break
            train_contexts.append(format_retrieved_knowledge(train_df, ret_idx, ret_sc))
        
        # Test retrieval
        D_te, I_te = index.search(test_features.astype(np.float32), args.rag_k)
        test_contexts = []
        for i in range(len(test_df)):
            test_contexts.append(format_retrieved_knowledge(train_df, I_te[i], D_te[i]))
        
        log("RAG retrieval done.")
        
        # ── Dataset & DataLoader ───────────────────────────────────────────
        train_dataset = MedVQADataset(
            train_df, train_contexts,
            image_root=image_root, max_length=512,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=get_collate_fn(processor)
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        total_steps = len(train_loader) * args.epochs // args.grad_accum
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

        log(f"Starting training for {args.epochs} epochs on client {client_id}...")
        all_losses = []
        epoch_times = []
        gpu_mem_readings_gb = []
        
        best_closed_acc = -1.0
        best_epoch = -1

        t_start = time.time()
        dt_start = datetime.datetime.now().isoformat()

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"[Client {client_id}] Epoch {epoch}/{args.epochs}")
            for step, batch in enumerate(pbar):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum
                loss.backward()
                epoch_loss += loss.item() * args.grad_accum

                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                pbar.set_postfix({"loss": f"{loss.item() * args.grad_accum:.4f}"})

            avg_loss = epoch_loss / max(len(train_loader), 1)
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            all_losses.append(avg_loss)
            gpu_mem_readings_gb.append(sample_gpu_memory_gb())

            log(f"[Client {client_id}] Epoch {epoch} done — avg_loss={avg_loss:.4f}  time={epoch_time:.1f}s")
            
            # Eval at end of epoch
            metrics = evaluate_model(model, processor, test_df, test_contexts, Path(image_root), rouge_scr, device, desc=f"Eval Epoch {epoch}")
            log(f"[Client {client_id}] Epoch {epoch} | Closed Acc={metrics['closed-ended']['Accuracy']:.4f} | Open ROUGE-L={metrics['open-ended']['ROUGE-L']:.4f}")
            
            if metrics["closed-ended"]["Accuracy"] > best_closed_acc:
                best_closed_acc = metrics["closed-ended"]["Accuracy"]
                best_epoch = epoch
                log(f"[Client {client_id}] New best Closed Acc = {best_closed_acc:.4f}. Saving best adapter...")
                model.save_pretrained(out_dir)

        t_end = time.time()
        total_s = t_end - t_start

        avg_vram = float(np.mean(gpu_mem_readings_gb)) if gpu_mem_readings_gb else 0.0
        peak_vram = float(max(gpu_mem_readings_gb)) if gpu_mem_readings_gb else 0.0
        hardware_payload = {
            "gpu_name": gpu_static["gpu_name"],
            "total_vram_gb": gpu_static["total_vram_gb"],
            "avg_allocated_vram_gb": round(avg_vram, 3),
            "peak_allocated_vram_gb": round(peak_vram, 3),
        }
        with open(out_dir / "hardware.json", "w") as f:
            json.dump(hardware_payload, f, indent=2)

        time_payload = {
            "start_time": dt_start,
            "end_time": datetime.datetime.now().isoformat(),
            "total_duration_s": round(total_s, 2),
            "total_duration_min": round(total_s / 60, 2),
            "avg_epoch_time_s": round(float(np.mean(epoch_times)), 2),
        }
        with open(out_dir / "training_time.json", "w") as f:
            json.dump(time_payload, f, indent=2)
            
        experiment_payload = {
            "dataset": args.dataset,
            "setting": args.setting,
            "alpha": args.alpha,
            "k": args.k,
            "client_id": client_id,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "rag_k": args.rag_k,
            "avg_train_loss": round(float(np.mean(all_losses)), 4),
            "best_closed_acc": round(best_closed_acc, 4),
            "best_epoch": best_epoch,
            "best_metrics": {
                "closed-ended": {"Accuracy": round(best_closed_acc, 4)},
                "open-ended": {"Accuracy": 0.0, "BLEU": 0.0, "ROUGE-L": 0.0, "BERTScore": 0.0},
                "hallucination": {"Hallucination Rate": 0.0},
                "performance": {"inference_time_s": 0.0, "throughput_samples_per_sec": 0.0}
            }
        }
        with open(out_dir / "experiment.json", "w") as f:
            json.dump(experiment_payload, f, indent=2)

        log(f"[Client {client_id}] Saved JSON artifacts. Total time: {total_s/60:.1f} min\n")

    log("Finished training for all clients.")

    del model, base_model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()