#!/usr/bin/env python3
"""
03_train_kd_roco_slake.py

Continue training the FedAvg global LoRA adapter on unified reasoning dataset (ROCO + Slake).

Pipeline:
Base Qwen2.5-VL-3B
    -> Load server/global_adapter_{mode} (from federated training)
    -> Continue supervised LoRA training on ROCO (caption-based) + Slake (QA-based) reasoning
    -> Save server/global_adapter_roco_slake_{mode}

ROCO: Open-ended caption reasoning
Slake: Closed-ended Yes/No question reasoning

uv run federated_code/03_train_kd_roco_slake.py --mode 01_train_client_mrag_kd --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2
"""

import os
import gc
import json
import time
import argparse
import random
import datetime
from pathlib import Path
from collections import Counter
import re as _re

# ── Windows DLL fix: import numpy/pandas BEFORE torch ──────────────────────
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

from transformers import (
    AutoProcessor,
    get_linear_schedule_with_warmup,
)
from peft import PeftModel

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ── Set HuggingFace cache (dynamic relative to project root) ───────────────
project_root = Path(__file__).parent.parent
CACHE_DIR = project_root / ".hf_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hub")

# ── Constants ───────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}][ROCO-TRAIN] {msg}", flush=True)

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
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 ** 3)

# ──────────────────────────────────────────────────────────────────────────────
# Dataset & collate
# ──────────────────────────────────────────────────────────────────────────────

class UnifiedReasoningDataset(Dataset):
    def __init__(self, df_roco=None, df_slake=None, roco_image_root=None, slake_image_root=None):
        """
        Unified dataset for ROCO (caption-based) and Slake (QA-based) reasoning.
        
        Args:
            df_roco: ROCO DataFrame with columns: image_id, caption, reasoning
            df_slake: Slake DataFrame with columns: img_id, img_name, question, answer, reasoning
            roco_image_root: Path to ROCO images
            slake_image_root: Path to Slake images (extracted imgs.zip)
        """
        self.samples = []
        
        # Add ROCO samples
        if df_roco is not None and roco_image_root is not None:
            df_roco = df_roco.dropna(subset=['image_id']).reset_index(drop=True)
            for _, row in df_roco.iterrows():
                image_id = row["image_id"]
                numeric_id = image_id.split("_")[-1]
                image_path = Path(roco_image_root) / f"{numeric_id}.jpg"
                self.samples.append({
                    "type": "roco",
                    "image_path": image_path,
                    "caption": row["caption"],
                    "reasoning": row["reasoning"],
                })
        
        # Add Slake samples
        if df_slake is not None and slake_image_root is not None:
            df_slake = df_slake.dropna(subset=['img_name']).reset_index(drop=True)
            for _, row in df_slake.iterrows():
                img_name = row["img_name"]
                image_path = Path(slake_image_root) / "imgs" / img_name
                self.samples.append({
                    "type": "slake",
                    "image_path": image_path,
                    "question": row["question"],
                    "answer": row["answer"],
                    "reasoning": row["reasoning"],
                })
        
        log(f"Loaded {len(self.samples)} total samples (ROCO: {sum(1 for s in self.samples if s['type']=='roco')}, Slake: {sum(1 for s in self.samples if s['type']=='slake')})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to load {sample['image_path']}: {e}")
            image = Image.new("RGB", (224, 224), color="black")
        
        return {
            "image": image,
            "type": sample["type"],
            "caption": sample.get("caption", ""),
            "question": sample.get("question", ""),
            "answer": sample.get("answer", ""),
            "reasoning": sample["reasoning"],
        }

def get_collate_fn(processor):
    def collate_fn(batch):
        all_images_per_sample = []
        user_prompts  = []
        full_prompts  = []

        for item in batch:
            all_images_per_sample.append([item["image"]])

            system_prompt = "You are a professional radiologist."
            
            # Build different prompts based on data type
            if item["type"] == "roco":
                # ROCO: Caption-based reasoning
                user_text = (
                    f"Medical Caption:\n{item['caption']}\n\n"
                    f"Task:\nExplain step-by-step why this caption matches the image.\n"
                    f"Focus only on visible evidence."
                )
            else:
                # Slake: QA-based reasoning (closed-ended Yes/No)
                user_text = (
                    f"Question:\n{item['question']}\n\n"
                    f"Answer:\n{item['answer']}\n\n"
                    f"Task:\nExplain step-by-step why this answer is correct for the question.\n"
                    f"Focus only on visible evidence in the image."
                )

            user_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "image", "image": item["image"]}, {"type": "text", "text": user_text}]}
            ]
            full_messages = user_messages + [
                {"role": "assistant", "content": [{"type": "text", "text": str(item["reasoning"])}]}
            ]

            user_p = processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
            full_p = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            
            user_prompts.append(user_p)
            full_prompts.append(full_p)

        flat_images = [img for imgs in all_images_per_sample for img in imgs]
        inputs = processor(text=full_prompts, images=flat_images, padding=True, return_tensors="pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = input_ids.clone()

        for i, user_p in enumerate(user_prompts):
            inp_user = processor(text=[user_p], images=all_images_per_sample[i], return_tensors="pt")
            user_len = inp_user["input_ids"].shape[1]
            labels[i, :user_len] = -100
            labels[i, attention_mask[i] == 0] = -100

        inputs["labels"] = labels
        return inputs
    return collate_fn

# ──────────────────────────────────────────────────────────────────────────────
# Zero-Shot VQA-RAD Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    if not text or (isinstance(text, float) and np.isnan(text)):
        return ""
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

def build_zeroshot_content(query_image, question, question_type):
    content = []
    content.append({"type": "text", "text": "System:\nYou are a medical visual question answering assistant.\n\nImage:\n"})
    content.append({"type": "image", "image": query_image})
    
    q_text = f"\n\nQuestion:\n{question}\n\nQuestion Type:\n{question_type}\n\n"
    content.append({"type": "text", "text": q_text})
    
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

def evaluate_model_zeroshot(model, processor, test_df, image_root, rouge_scr, device, desc="Zero-Shot Test"):
    model.eval()
    predictions = []
    start_time = time.time()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=desc):
        question = row["question"]
        q_type_raw = str(row.get("question_type", "open")).lower()
        q_type = "CLOSED-ENDED" if q_type_raw == "closed" else "OPEN-ENDED"
        
        img_path = image_root / row["image_path"]
        try:
            query_image = Image.open(img_path).convert("RGB")
        except:
            query_image = Image.new("RGB", (224, 224), color="black")
            
        content = build_zeroshot_content(query_image, question, q_type)
        messages = [{"role": "user", "content": content}]
        
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
            
    closed_acc = float(np.mean([compute_exact_match(p, g) for p, g in zip(closed_preds, closed_gts)])) if closed_preds else 0.0
    open_acc = float(np.mean([compute_exact_match(p, g) for p, g in zip(open_preds, open_gts)])) if open_preds else 0.0
    open_bleu = float(np.mean([compute_bleu(p, g) for p, g in zip(open_preds, open_gts)])) if open_preds else 0.0
    open_rouge = float(np.mean([compute_rouge_l(p, g, rouge_scr) for p, g in zip(open_preds, open_gts)])) if open_preds else 0.0
    
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        help="Script name suffix from federated training (e.g., 01_train_client_mrag_kd)")
    parser.add_argument("--dataset", type=str, default="VQA-RAD")
    parser.add_argument("--setting", type=str, choices=["iid", "noniid"], default="noniid")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="VLM model name from HuggingFace")
    parser.add_argument("--model_short", type=str, default="Qwen2.5-VL-3B-Instruct",
                        help="Short model name for directory naming")
    parser.add_argument("--global_adapter", type=str, default=None, help="Path to server/global_adapter (auto-generated if not provided)")
    parser.add_argument("--roco_csv", type=str, default=None, help="Path to ROCO reasoning CSV (auto-generated if not provided)")
    parser.add_argument("--roco_image_root", type=str, default=None, help="Path to ROCO images (auto-generated if not provided)")
    parser.add_argument("--slake_csv", type=str, default=None, help="Path to Slake reasoning CSV (auto-generated if not provided)")
    parser.add_argument("--slake_image_root", type=str, default=None, help="Path to Slake images (auto-generated if not provided)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (auto-generated if not provided)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--test_csv", type=str, default=None, help="Path to VQA-RAD test CSV (auto-generated if not provided)")
    parser.add_argument("--image_root", type=str, default=None, help="Path to VQA-RAD images (auto-generated if not provided)")
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Auto-generate paths if not provided
    base = Path("experiments") / args.dataset / args.setting
    if args.setting == "noniid":
        base = base / f"alpha_{args.alpha}"
    base = base / f"k{args.k}" / args.model_short / "server"
    
    data_root = Path("processed_data")
    
    if args.global_adapter is None:
        args.global_adapter = str(base / f"global_adapter_{args.mode}")
    if args.out_dir is None:
        args.out_dir = str(base / f"global_adapter_roco_slake_{args.mode}")
    if args.roco_csv is None:
        args.roco_csv = str(data_root / "ROCO" / "roco_reasoning_train.csv")
    if args.roco_image_root is None:
        args.roco_image_root = str(data_root / "ROCO" / "train" / "images")
    if args.slake_csv is None:
        args.slake_csv = str(data_root / "Slake" / "slake_closed_reasoning.csv")
    if args.slake_image_root is None:
        args.slake_image_root = str(data_root / "Slake" / "images")
    if args.test_csv is None:
        args.test_csv = str(data_root / "VQA-RAD" / "centralized" / "test.csv")
    if args.image_root is None:
        args.image_root = str(data_root / "VQA-RAD" / "centralized")

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log(f"Starting unified reasoning training (ROCO + Slake)")
    log(f"Global adapter: {args.global_adapter}")
    log(f"ROCO CSV: {args.roco_csv}")
    log(f"Slake CSV: {args.slake_csv}")
    log(f"Output directory: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_static = get_gpu_static_info()
    log(f"Device: {device} | GPU: {gpu_static['gpu_name']} | VRAM: {gpu_static['total_vram_gb']} GB")

    # 1. Load Data
    log("Loading reasoning datasets...")
    
    # Load ROCO
    df_roco = None
    if args.roco_csv and Path(args.roco_csv).exists():
        df_roco = pd.read_csv(args.roco_csv)
        log(f"Loaded ROCO: {len(df_roco)} samples")
    
    # Load Slake
    df_slake = None
    if args.slake_csv and Path(args.slake_csv).exists():
        df_slake = pd.read_csv(args.slake_csv)
        log(f"Loaded Slake: {len(df_slake)} samples")
    
    train_dataset = UnifiedReasoningDataset(
        df_roco=df_roco,
        df_slake=df_slake,
        roco_image_root=args.roco_image_root,
        slake_image_root=args.slake_image_root
    )
    
    # 2. Load Processor and Base Model
    log(f"Loading base model {args.model_name}...")
    processor = AutoProcessor.from_pretrained(args.model_name, min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28, cache_dir=CACHE_DIR)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    if "qwen2.5" in args.model_name.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    else:
        from transformers import Qwen2VLForConditionalGeneration as ModelClass

    base_model = ModelClass.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        cache_dir=CACHE_DIR
    )

    # 3. Load FedAvg Adapter
    log(f"Loading global adapter from {args.global_adapter}...")
    adapter_path = str(Path(args.global_adapter).as_posix())
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True, is_local=True)
    model.print_trainable_parameters()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=get_collate_fn(processor)
    )

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # 5. Training Loop
    log(f"Starting training for {args.epochs} epochs...")
    all_losses = []
    epoch_times = []
    gpu_mem_readings_gb = []

    t_start = time.time()
    dt_start = datetime.datetime.now().isoformat()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
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

        log(f"Epoch {epoch} done — avg_loss={avg_loss:.4f}  time={epoch_time:.1f}s")
    
    # 6. Evaluate Zero-Shot on VQA-RAD
    log("Running zero-shot evaluation on VQA-RAD test set...")
    test_df = pd.read_csv(args.test_csv)
    rouge_scr = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    best_metrics = evaluate_model_zeroshot(
        model=model,
        processor=processor,
        test_df=test_df,
        image_root=Path(args.image_root),
        rouge_scr=rouge_scr,
        device=device
    )
    log(f"Evaluation finished: {best_metrics}")

    # 7. Save Updated Adapter
    log(f"Training finished. Saving updated adapter to {out_dir}")
    model.save_pretrained(out_dir)

    t_end = time.time()
    total_s = t_end - t_start

    # 8. Save Artifacts
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
        "dataset": "ROCO+Slake",
        "roco_samples": len(df_roco) if df_roco is not None else 0,
        "slake_samples": len(df_slake) if df_slake is not None else 0,
        "total_samples": len(train_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "avg_train_loss": round(float(np.mean(all_losses)), 4),
        "best_metrics": best_metrics
    }
    with open(out_dir / "experiment.json", "w") as f:
        json.dump(experiment_payload, f, indent=2)

    log("Saved JSON artifacts.")
    log(f"Total time: {total_s/60:.1f} min")

    del model, base_model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
