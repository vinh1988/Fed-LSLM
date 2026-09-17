#!/usr/bin/env python3
"""
01_train_client_nomrag_nokd.py

Local-client training script for Federated Learning WITHOUT RAG (No-RAG) and WITHOUT KD.

Architecture:
  - Qwen/Qwen2-VL-2B-Instruct + LoRA : answer generator (bfloat16, base frozen, only LoRA adapters trained).
  - No retrieval (no BiomedCLIP, no FAISS)
  - No Knowledge Distillation (no ROCO/VQA-RAD KD step)
  - Uses Slake metadata (body_part, modality, objects) in prompts

Output structure (per Experiment Space spec):
  experiments/{dataset}/{setting}/[alpha_{alpha}/]k{k}/Qwen2-VL-2B-Instruct/clients/client_{client_id}_01_train_client_nomrag_nokd/
       ├── adapter_config.json    # LoRA adapter config (PEFT save_pretrained)
       ├── adapter_model.safetensors # LoRA adapter weights (PEFT save_pretrained)
       ├── experiment.json        # summary metrics (avg + best across epochs)
       ├── hardware.json          # GPU name + average VRAM usage during training
       └── training_time.json     # wall-clock timing (total + average per epoch)

Windows note: numpy & pandas MUST be imported before torch to avoid DLL conflicts.
"""
# federated_code/

# 01_train_client.py

# Produces

# client_i/adapter

# # Non-IID α=0.1, k=2 → tự chạy client_1, client_2
# uv run federated_code/01_train_client_mrag_kd.py --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2
# uv run federated_code/01_train_client_nonicl.py --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2


# ── Windows DLL fix: import numpy/pandas BEFORE torch ──────────────────────
import numpy as np
import pandas as pd

import os
import gc
import json
import time
import argparse
import random
import platform
import datetime
from pathlib import Path
from collections import Counter

# ── Set HuggingFace cache (dynamic relative to project root) ───────────────
project_root = Path(__file__).parent.parent
CACHE_DIR = project_root / ".hf_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hub")

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ── Constants ───────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}][LOCAL-CLIENT] {msg}", flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_output_dir(dataset: str, setting: str, alpha, k: int, client_id: int, model_short: str) -> Path:
    """
    Construct the experiment output directory following the Experiment Space spec:
        experiments/{dataset}/{setting}/[alpha_{alpha}/]k{k}/{model}/clients/client_{id}_scriptname/
    """
    script_name = Path(__file__).stem
    base = Path("experiments") / dataset / setting
    if setting == "noniid":
        if alpha is None:
            raise ValueError("--alpha must be set for noniid setting")
        base = base / f"alpha_{alpha}"
    base = base / f"k{k}" / model_short / "clients" / f"client_{client_id}_{script_name}"
    return base


def resolve_data_paths(dataset: str, setting: str, alpha, k: int, client_id: int):
    """
    Returns (client_csv_path, test_csv_path, image_base_dir).

    Data layout on disk:
        processed_data/{dataset}/federated_iid/k{k}/client_{id}.csv
        processed_data/{dataset}/federated_noniid/alpha_{alpha}/k{k}/client_{id}.csv
        processed_data/{dataset}/centralized/test.csv
        processed_data/{dataset}/centralized/images/...   ← image root
    """
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
    image_root = data_root / "centralized"      # images paths inside CSV are relative to here

    return client_csv, test_csv, image_root


# ──────────────────────────────────────────────────────────────────────────────
# Hardware info
# ──────────────────────────────────────────────────────────────────────────────

def get_gpu_static_info() -> dict:
    """Return static GPU hardware info (name + total VRAM)."""
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
# No RAG - No retrieval components needed
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# No RAG - No retrieval components needed
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Dataset & collate
# ──────────────────────────────────────────────────────────────────────────────

class MedVQADataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: Path, max_length: int = 512):
        self.df = df.reset_index(drop=True)
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
        
        # Load SLAKE metadata
        body_part = row.get('body_part', '')
        modality = row.get('modality', '')
        objects = row.get('objects', '')
        
        return {
            "query_image":   self._load_image(row["image_path"]),
            "question":      row["question"],
            "question_type": question_type,
            "answer":        str(row["answer"]),
            "body_part":     body_part,
            "modality":      modality,
            "objects":       objects
        }


def build_content(query_image: Image.Image, question: str, question_type: str, body_part: str = "", modality: str = "", objects: str = ""):
    """
    Construct a Qwen2-VL prompt with Slake metadata (No-RAG):
      [query_image] + question + medical context + answer prompt
    """
    content = []
    
    # 1) System and Image
    content.append({"type": "text", "text": "System:\nYou are a medical visual question answering assistant specializing in medical image analysis.\n\nImage:\n"})
    content.append({"type": "image", "image": query_image})
    
    # 2) Question and Question Type
    q_text = (
        f"\n\nQuestion:\n{question}\n\n"
        f"Question Type:\n{question_type}\n\n"
    )
    content.append({"type": "text", "text": q_text})
    
    # 3) Medical Context (from Slake metadata)
    # Format objects summary from JSON
    objects_summary = ""
    if objects:
        try:
            import json
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
    
    # 4) Instruction and rules
    instruction = (
        "Instruction:\n"
        "You must adapt your reasoning based on the question type.\n\n"
        "- If the question is CLOSED-ENDED (Yes/No only):\n"
        "  • The final answer MUST be \"yes\" or \"no\"\n"
        "  • Use the medical context and detected objects to inform your decision\n"
        "  • Focus primarily on the image\n\n"
        "- If the question is OPEN-ENDED:\n"
        "  • Use the image, body part, modality, and detected objects to provide an accurate answer\n"
        "  • Consider the imaging modality when interpreting findings\n\n"
        "General rules:\n"
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
            
            content = build_content(
                item["query_image"], item["question"], item["question_type"],
                item["body_part"], item["modality"], item["objects"]
            )
            user_messages = [{"role": "user", "content": content}]
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
# Evaluation metrics
# ──────────────────────────────────────────────────────────────────────────────

import re as _re

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — matches metric.py reference."""
    if not text or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text).lower().strip()
    text = _re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = _re.sub(r"\s+", " ", text).strip()
    return text


def compute_exact_match(pred: str, gt: str) -> float:
    """Closed: exact match after normalization (= Accuracy for closed questions)."""
    return 1.0 if _normalize(pred) == _normalize(gt) else 0.0

def compute_f1(pred: str, gt: str) -> float:
    """Token-level F1 after normalization."""
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gt).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common  = Counter(p_toks) & Counter(g_toks)
    n_same  = sum(common.values())
    if n_same == 0:
        return 0.0
    prec = n_same / len(p_toks)
    rec  = n_same / len(g_toks)
    return 2 * prec * rec / (prec + rec)

def compute_bleu(pred: str, gt: str) -> float:
    """BLEU-4 after normalization (for open-ended questions)."""
    ref    = [_normalize(gt).split()]
    hyp    = _normalize(pred).split()
    if not hyp:
        return 0.0
    smooth = SmoothingFunction().method1
    return sentence_bleu(ref, hyp, smoothing_function=smooth)

def compute_rouge_l(pred: str, gt: str, scorer) -> float:
    """ROUGE-L after normalization (for open-ended questions)."""
    scores = scorer.score(_normalize(gt), _normalize(pred))
    return scores["rougeL"].fmeasure

def _load_image_safe(img_path: Path) -> Image.Image:
    try:
        return Image.open(img_path).convert("RGB")
    except Exception as e:
        log(f"Warning: failed to load {img_path}: {e}")
        return Image.new("RGB", (224, 224), color="black")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, processor, test_df: pd.DataFrame, image_root: Path, rouge_scr, device, desc: str = "Test"):
    """
    Evaluate model on the test set (No-RAG).
    Returns metrics dict with the new format:
      {
        "closed-ended": {"n_samples": ..., "Accuracy": ...},
        "open-ended": {"n_samples": ..., "Accuracy": ..., "Exact Match": ..., "BLEU": ..., "ROUGE-L": ..., "BERTScore": ...},
        "hallucination": {"Hallucination Rate": ...},
        "performance": {"inference_time_s": ..., "throughput_samples_per_sec": ...}
      }
    Note: BERTScore is NOT computed here (too slow during training).
          Run metric.py post-hoc on best_predictions.csv to get BERTScore.
    """
    model.eval()
    predictions = []
    start_time = time.time()

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=desc):
        question  = row["question"]
        q_type_raw = str(row.get("question_type", "open")).lower()
        q_type    = "CLOSED-ENDED" if q_type_raw == "closed" else "OPEN-ENDED"
        body_part = row.get('body_part', '')
        modality = row.get('modality', '')
        objects = row.get('objects', '')

        query_image  = _load_image_safe(image_root / row["image_path"])
        
        content = build_content(query_image, question, q_type, body_part, modality, objects)
        messages     = [{"role": "user", "content": content}]
        
        all_images = [query_image]

        prompt  = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs  = processor(text=[prompt], images=all_images, return_tensors="pt")
        inputs  = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        gen_text   = processor.tokenizer.decode(
            output_ids[0][prompt_len:], skip_special_tokens=True
        ).strip()
        predictions.append(gen_text)
    
    inference_time = time.time() - start_time
    throughput = len(test_df) / inference_time if inference_time > 0 else 0.0

    # ── Split by question type ───────────────────────────────────────────────
    closed_preds, closed_gts = [], []
    open_preds,   open_gts   = [], []

    for pred, row in zip(predictions, test_df.itertuples()):
        qt = str(row.question_type).lower()
        if qt == "closed":
            closed_preds.append(pred)
            closed_gts.append(str(row.answer))
        else:
            open_preds.append(pred)
            open_gts.append(str(row.answer))

    # ── Closed metrics: Accuracy ────────────────────────────────────────
    if closed_preds:
        closed_acc = float(np.mean([compute_exact_match(p, g) for p, g in zip(closed_preds, closed_gts)]))
    else:
        closed_acc = 0.0

    # ── Open metrics: Accuracy, Exact Match, BLEU, ROUGE-L ──────────────
    if open_preds:
        open_acc     = float(np.mean([compute_exact_match(p, g) for p, g in zip(open_preds, open_gts)]))
        open_bleu    = float(np.mean([compute_bleu(p, g)             for p, g in zip(open_preds, open_gts)]))
        open_rouge   = float(np.mean([compute_rouge_l(p, g, rouge_scr) for p, g in zip(open_preds, open_gts)]))
    else:
        open_acc, open_bleu, open_rouge = 0.0, 0.0, 0.0

    # ── Hallucination: Placeholder (requires semantic similarity model) ──
    hallucination_rate = 0.0  # TODO: Implement with semantic similarity model

    metrics = {
        "closed-ended": {
            "n_samples": len(closed_preds),
            "Accuracy":  round(closed_acc, 6),
        },
        "open-ended": {
            "n_samples": len(open_preds),
            "Accuracy":    round(open_acc, 6),
            "Exact Match": round(open_acc, 6),  # Same as accuracy for open-ended
            "BLEU":        round(open_bleu, 6),
            "ROUGE-L":     round(open_rouge, 6),
            "BERTScore":   0.0,  # Placeholder - compute post-hoc
        },
        "hallucination": {
            "Hallucination Rate": round(hallucination_rate, 6),
        },
        "performance": {
            "inference_time_s": round(inference_time, 3),
            "throughput_samples_per_sec": round(throughput, 3),
        },
    }
    return metrics, predictions


# ──────────────────────────────────────────────────────────────────────────────
# Single-client training routine
# ──────────────────────────────────────────────────────────────────────────────

def train_single_client(args, client_id: int):
    """
    Train one client end-to-end (No-RAG):
      1) Qwen2-VL + LoRA training
      2) Save JSON artifacts

    Returns the output directory Path so the caller can log it.
    """
    log(f"{'='*60}")
    log(f"  CLIENT {client_id}/{args.k}  ({args.setting}, k={args.k})")
    log(f"{'='*60}")

    set_seed(args.seed)

    # ── Paths ────────────────────────────────────────────────────────────────
    client_csv, test_csv, image_root = resolve_data_paths(
        args.dataset, args.setting, args.alpha, args.k, client_id
    )
    out_dir = build_output_dir(args.dataset, args.setting, args.alpha, args.k, client_id, args.model_short)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_str = f"alpha_{args.alpha}" if args.alpha is not None else "N/A"
    log(f"Dataset   : {args.dataset}")
    log(f"Setting   : {args.setting}  alpha={alpha_str}  k={args.k}  client_id={client_id}")
    log(f"Client CSV: {client_csv}")
    log(f"Test CSV  : {test_csv}")
    log(f"Output dir: {out_dir}")

    assert client_csv.exists(), f"Client CSV not found: {client_csv}"
    assert test_csv.exists(),   f"Test CSV not found: {test_csv}"

    # ── Static GPU info ──────────────────────────────────────────────────────
    gpu_static           = get_gpu_static_info()
    gpu_mem_readings_gb  = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}  GPU: {gpu_static['gpu_name']}  VRAM: {gpu_static['total_vram_gb']:.1f} GB")

    # ── Load CSVs ────────────────────────────────────────────────────────────
    train_df = pd.read_csv(client_csv).reset_index(drop=True)
    test_df  = pd.read_csv(test_csv).reset_index(drop=True)
    
    # Limit to 1 sample for testing
    # train_df = train_df.head(1)
    # test_df = test_df.head(1)
    
    log(f"Train samples (client_{client_id}): {len(train_df)}")
    log(f"Test  samples (centralized):       {len(test_df)}")

    # ── Timing start ─────────────────────────────────────────────────────────
    t_start  = time.time()
    dt_start = datetime.datetime.now().isoformat()

    # ── Step 1: Load Qwen2-VL + LoRA ─────────────────────────────────────────

    # ── Step 2: Load Qwen2-VL + LoRA ─────────────────────────────────────────
    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    log(f"Loading processor & VLM: {args.model_name}...")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
        cache_dir=CACHE_DIR,
    )
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if "smolvlm" in args.model_name.lower() or "idefics3" in args.model_name.lower():
        from transformers import Idefics3ForConditionalGeneration as ModelClass
    elif "qwen2.5" in args.model_name.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    else:
        from transformers import Qwen2VLForConditionalGeneration as ModelClass

    base_model = ModelClass.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True,
    )

    for param in base_model.parameters():
        param.requires_grad = False

    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ── Step 3: Dataset & DataLoader ─────────────────────────────────────────
    train_dataset = MedVQADataset(
        train_df,
        image_root=image_root, max_length=args.max_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=get_collate_fn(processor),
    )

    rouge_scr = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # ── Step 4: Optimizer & Scheduler ────────────────────────────────────────
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ── Step 5: Training loop ────────────────────────────────────────────────
    log(f"Starting training — {args.epochs} epochs, {len(train_df)} samples ...")
    all_losses      = []
    epoch_times     = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss  = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"[Client {client_id}] Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(**batch)
            loss    = outputs.loss / args.grad_accum
            loss.backward()

            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item() * args.grad_accum:.4f}"})

        avg_loss   = epoch_loss / max(len(train_loader), 1)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        all_losses.append(avg_loss)

        gpu_mem_readings_gb.append(sample_gpu_memory_gb())

        log(f"Epoch {epoch} done — avg_loss={avg_loss:.4f}  time={epoch_time:.1f}s")

    # ── Save final adapter ─────────────────────────────────────────────────────
    log(f"Saving final adapter to {out_dir}")
    model.save_pretrained(out_dir)

    # ── Step 6: Save JSON artifacts ──────────────────────────────────────────
    t_end   = time.time()
    dt_end  = datetime.datetime.now().isoformat()
    total_s = t_end - t_start

    experiment_payload = {
        "dataset":           args.dataset,
        "setting":           args.setting,
        "alpha":             args.alpha,
        "k":                 args.k,
        "client_id":         client_id,
        "model":             args.model_short,
        "epochs":            args.epochs,
        "lr":                args.lr,
        "batch_size":        args.batch_size,
        "grad_accum":        args.grad_accum,
        "seed":              args.seed,
        "num_train_samples": len(train_df),
        "num_test_samples":  len(test_df),
        "avg_train_loss":    round(float(np.mean(all_losses)), 6),
    }
    with open(out_dir / "experiment.json", "w") as f:
        json.dump(experiment_payload, f, indent=2)

    avg_vram  = float(np.mean(gpu_mem_readings_gb)) if gpu_mem_readings_gb else 0.0
    peak_vram = float(max(gpu_mem_readings_gb))     if gpu_mem_readings_gb else 0.0
    hardware_payload = {
        "gpu_name":               gpu_static["gpu_name"],
        "total_vram_gb":          gpu_static["total_vram_gb"],
        "avg_allocated_vram_gb":  round(avg_vram,  3),
        "peak_allocated_vram_gb": round(peak_vram, 3),
    }
    with open(out_dir / "hardware.json", "w") as f:
        json.dump(hardware_payload, f, indent=2)

    time_payload = {
        "start_time":         dt_start,
        "end_time":           dt_end,
        "total_duration_s":   round(total_s, 2),
        "total_duration_min": round(total_s / 60, 2),
        "avg_epoch_time_s":   round(float(np.mean(epoch_times)), 2),
    }
    with open(out_dir / "training_time.json", "w") as f:
        json.dump(time_payload, f, indent=2)

    log(f"All artifacts saved to: {out_dir}")
    log(f"Total time: {total_s/60:.1f} min")

    # ── Cleanup GPU memory before next client ────────────────────────────────
    del model, base_model, optimizer, scheduler
    del train_dataset, train_loader, processor
    torch.cuda.empty_cache()
    gc.collect()
    log(f"GPU memory freed after client {client_id}.\n")

    return out_dir


# ──────────────────────────────────────────────────────────────────────────────
# Main: parse args → loop through ALL clients
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Local client training for Federated mRAG Med-VQA. "
                    "Automatically trains ALL clients (1 … k) sequentially."
    )

    # ── Experiment identity ──────────────────────────────────────────────────
    parser.add_argument("--dataset", type=str,   default="Slake",
                        help="Dataset name (Slake | VQA-RAD | PathVQA)")
    parser.add_argument("--setting", type=str,   default="iid",
                        help="Federated setting (iid | noniid)")
    parser.add_argument("--alpha",   type=float, default=None,
                        help="Dirichlet alpha for noniid (e.g. 0.5, 0.1)")
    parser.add_argument("--k",       type=int,   default=3,
                        help="Total number of clients")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
                        help="VLM model name from HuggingFace")
    parser.add_argument("--model_short", type=str, default="Qwen2-VL-2B-Instruct",
                        help="Short model name for directory naming")

    # ── RAG retrieval ────────────────────────────────────────────────────────
    parser.add_argument("--rag_k",     type=int, default=3,
                        help="Number of retrieved cases for mRAG context")

    # ── Training hyperparams ─────────────────────────────────────────────────
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=1)
    parser.add_argument("--grad_accum", type=int,   default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--resume_from", type=int, default=None,
                        help="Resume training from this client ID (1-indexed)")

    args = parser.parse_args()

    # ── Validate ─────────────────────────────────────────────────────────────
    if args.setting == "noniid" and args.alpha is None:
        parser.error("--alpha is required when --setting noniid")

    log(f"Starting sequential training for ALL {args.k} clients.")
    log(f"Dataset={args.dataset}  setting={args.setting}  "
        f"alpha={args.alpha}  k={args.k}  epochs={args.epochs}")
    if args.resume_from:
        log(f"Resuming from client {args.resume_from}")
    log("")

    total_start = time.time()
    results     = []

    start_cid = args.resume_from if args.resume_from else 1
    for cid in range(start_cid, args.k + 1):
        out = train_single_client(args, client_id=cid)
        results.append((cid, out))

    total_min = (time.time() - total_start) / 60
    log(f"{'='*60}")
    log(f"ALL {args.k} CLIENTS COMPLETED in {total_min:.1f} min")
    for cid, out in results:
        log(f"  client_{cid} to {out}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
