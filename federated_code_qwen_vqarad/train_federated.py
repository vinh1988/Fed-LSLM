#!/usr/bin/env python3
"""
train_federated.py

Federated Learning (FedAvg) with SmolVLM-500M-Instruct + mRAG for Medical VQA.
Sequential client simulation on VQA-RAD dataset.

Architecture:
    Retriever : BiomedCLIP (frozen) → 1024-dim embeddings
    Index     : FAISS IndexFlatIP (cosine similarity, CPU)
    Generator : SmolVLM-500M-Instruct (Idefics3) + LoRA (PEFT)
    Algorithm : FedAvg — weighted average by client sample count

Usage:
    # IID k=3
    python train_federated.py --setting iid --num_clients 3

    # Non-IID alpha=0.5 k=3
    python train_federated.py --setting noniid --alpha 0.5 --num_clients 3

    # Non-IID alpha=0.1 k=3
    python train_federated.py --setting noniid --alpha 0.1 --num_clients 3
"""

# ============================================================
# CRITICAL: Import numpy/pandas BEFORE torch
# Prevents silent crash from PyArrow DLL conflict on Windows.
# ============================================================
import numpy as np
import pandas as pd

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HUGGINGFACE_HUB_CACHE"] = (
    r"D:\nckh_teacher_student\code\teacher_student\.hf_cache\hub"
)

import gc
import json
import argparse
import random
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
import faiss
from PIL import Image
from tqdm import tqdm

import open_clip
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# ============================================================
# LOGGING
# ============================================================
def log(msg: str) -> None:
    """Print a timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[FedAvg {ts}] {msg}")


# ============================================================
# SEED
# ============================================================
def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# METRIC HELPERS
# ============================================================
def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.strip().lower().split()
    gt_tokens = ground_truth.strip().lower().split()
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_bleu(prediction: str, ground_truth: str) -> float:
    ref = [ground_truth.strip().lower().split()]
    hyp = prediction.strip().lower().split()
    smooth = SmoothingFunction().method1
    return sentence_bleu(ref, hyp, smoothing_function=smooth)


def compute_rouge_l(prediction: str, ground_truth: str, scorer) -> float:
    scores = scorer.score(ground_truth.strip().lower(), prediction.strip().lower())
    return scores["rougeL"].fmeasure


# ============================================================
# BIOMEDCLIP EMBEDDING
# ============================================================
def _precompute_embeddings_single(
    csv_path: Path,
    data_dir: Path,
    model,
    preprocess,
    tokenizer,
    device: torch.device,
) -> np.ndarray:
    """Compute BiomedCLIP image+text embeddings for every row in a CSV."""
    df = pd.read_csv(csv_path)
    embeddings: List[np.ndarray] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Embedding {csv_path.name}"):
        img_path = data_dir / row["image_path"]
        question = str(row["question"])

        try:
            image = Image.open(img_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
        except Exception as e:
            log(f"  Warning: failed to load image {img_path}: {e}")
            image_input = torch.zeros(1, 3, 224, 224).to(device)

        text_input = tokenizer([question]).to(device)

        with torch.no_grad():
            img_feat = model.encode_image(image_input)
            txt_feat = model.encode_text(text_input)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            combined = torch.cat([img_feat, txt_feat], dim=-1)
            embeddings.append(combined.cpu().numpy()[0])

    return np.array(embeddings)


def precompute_all_embeddings(
    client_csv_paths: List[Path],
    centralized_train_csv: Path,
    test_csv: Path,
    data_dir: Path,
    cache_dir: Path,
    biomed_model_name: str,
    device: torch.device,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Precompute and cache BiomedCLIP embeddings for:
      - Each client CSV (for per-client FAISS during training)
      - Centralized train CSV (for test-time retrieval)
      - Test CSV (queries at evaluation time)
    BiomedCLIP is loaded once, then fully freed.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    client_caches = [
        cache_dir / f"client_{i + 1}_embeddings.npy"
        for i in range(len(client_csv_paths))
    ]
    train_cache = cache_dir / "centralized_train_embeddings.npy"
    test_cache = cache_dir / "test_embeddings.npy"

    all_cached = (
        all(c.exists() for c in client_caches)
        and train_cache.exists()
        and test_cache.exists()
    )

    if all_cached:
        log("All BiomedCLIP embeddings already cached – loading from disk...")
        client_embs = [np.load(c) for c in client_caches]
        train_embs = np.load(train_cache)
        test_embs = np.load(test_cache)
        return client_embs, train_embs, test_embs

    # ---------- Load BiomedCLIP ----------
    log("Loading BiomedCLIP to compute embeddings (one-time)...")
    biomed_model, preprocess = open_clip.create_model_from_pretrained(biomed_model_name)
    biomed_tokenizer = open_clip.get_tokenizer(biomed_model_name)
    biomed_model.to(device)
    biomed_model.eval()

    client_embs: List[np.ndarray] = []
    for i, csv_path in enumerate(client_csv_paths):
        log(f"Computing embeddings for client_{i + 1}...")
        emb = _precompute_embeddings_single(
            csv_path, data_dir, biomed_model, preprocess, biomed_tokenizer, device
        )
        np.save(client_caches[i], emb)
        client_embs.append(emb)

    if not train_cache.exists():
        log("Computing centralized train embeddings (for test-time retrieval)...")
        train_embs = _precompute_embeddings_single(
            centralized_train_csv, data_dir, biomed_model, preprocess, biomed_tokenizer, device
        )
        np.save(train_cache, train_embs)
    else:
        train_embs = np.load(train_cache)

    if not test_cache.exists():
        log("Computing test embeddings...")
        test_embs = _precompute_embeddings_single(
            test_csv, data_dir, biomed_model, preprocess, biomed_tokenizer, device
        )
        np.save(test_cache, test_embs)
    else:
        test_embs = np.load(test_cache)

    # ---------- Free BiomedCLIP ----------
    del biomed_model
    torch.cuda.empty_cache()
    gc.collect()
    log("BiomedCLIP freed from GPU memory.")

    return client_embs, train_embs, test_embs


# ============================================================
# FAISS INDEX  &  RETRIEVAL
# ============================================================
def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


def build_faiss_index(embeddings: np.ndarray) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """Build a FAISS inner-product index from L2-normalised embeddings."""
    normed = _l2_normalize(embeddings)
    index = faiss.IndexFlatIP(normed.shape[1])  # 1024-dim
    index.add(normed)
    return index, normed


def format_retrieved_knowledge(
    df: pd.DataFrame,
    indices: List[int],
    scores: List[float],
) -> List[str]:
    """Format retrieved QA pairs into text context strings."""
    docs: List[str] = []
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        row = df.iloc[idx]
        docs.append(
            f"Top-{rank}:\n"
            f"Score: {score:.4f}\n"
            f"Question: {row['question']}\n"
            f"Answer: {row['answer']}\n"
        )
    return docs


def retrieve_contexts_for_training(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    k: int,
) -> List[List[str]]:
    """
    Retrieve top-k contexts for training rows.
    Excludes the query row itself to prevent data leakage.
    """
    index, normed = build_faiss_index(embeddings)
    D, I = index.search(normed, k + 1)

    all_contexts: List[List[str]] = []
    for idx in range(len(df)):
        retrieved_indices: List[int] = []
        retrieved_scores: List[float] = []
        for pos, match_idx in enumerate(I[idx]):
            if match_idx == idx:
                continue
            retrieved_indices.append(int(match_idx))
            retrieved_scores.append(float(D[idx][pos]))
            if len(retrieved_indices) == k:
                break
        # Fallback if fewer than k neighbours
        while len(retrieved_indices) < k:
            for pos, match_idx in enumerate(I[idx]):
                if int(match_idx) not in retrieved_indices:
                    retrieved_indices.append(int(match_idx))
                    retrieved_scores.append(float(D[idx][pos]))
                    break
            if len(retrieved_indices) >= k:
                break

        contexts = format_retrieved_knowledge(df, retrieved_indices, retrieved_scores)
        all_contexts.append(contexts)

    return all_contexts


def retrieve_contexts_for_test(
    train_df: pd.DataFrame,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    k: int,
) -> List[List[str]]:
    """
    Retrieve top-k contexts for test queries from the centralized training set.
    NOTE: For global evaluation we use ALL centralized training data as the
    retrieval corpus so that the evaluation faithfully measures generator quality.
    """
    index, _ = build_faiss_index(train_embeddings)
    test_normed = _l2_normalize(test_embeddings)
    D, I = index.search(test_normed, k)

    all_contexts: List[List[str]] = []
    for idx in range(len(D)):
        contexts = format_retrieved_knowledge(
            train_df, I[idx].tolist(), D[idx].tolist()
        )
        all_contexts.append(contexts)

    return all_contexts


# ============================================================
# mRAG PROMPT BUILDING  (SmolVLM / Idefics3 format)
# ============================================================
def build_mrag_content(
    question: str,
    question_type: str,
    context_list: List[str],
) -> List[Dict[str, Any]]:
    """
    Build the *user-message content* list for SmolVLM mRAG prompt.

    SmolVLM uses ``{"type": "image"}`` as a **placeholder** —
    the actual PIL image is passed separately to ``processor(images=[...])``.
    """
    content: List[Dict[str, Any]] = []

    # 1) System text + image placeholder
    content.append({
        "type": "text",
        "text": "System:\nYou are a medical visual question answering assistant.\n\nImage:\n",
    })
    content.append({"type": "image"})  # SmolVLM placeholder

    # 2) Question, type, retrieved evidence header
    k = len(context_list)
    q_text = (
        f"\n\nQuestion:\n{question}\n\n"
        f"Question Type:\n{question_type}\n\n"
        f"Retrieved Medical Evidence (Top-{k} QA cases ranked by similarity score):\n\n"
    )
    content.append({"type": "text", "text": q_text})

    # 3) Retrieved text contexts
    for case_text in context_list:
        content.append({"type": "text", "text": case_text + "\n"})

    # 4) Instruction block
    instruction = (
        "Instruction:\n"
        "You must adapt your reasoning based on the question type.\n\n"
        "- If the question is CLOSED-ENDED (Yes/No only):\n"
        '  • The final answer MUST be "yes" or "no"\n'
        "  • Focus primarily on the image\n"
        "  • Use retrieved QA cases only if they strongly match the visual evidence\n\n"
        "- If the question is OPEN-ENDED:\n"
        "  • Use both the image and retrieved QA cases\n"
        "  • Combine visual evidence with similar QA examples to improve reasoning\n\n"
        "General rules:\n"
        "- Retrieved cases are ranked by similarity score (higher = more relevant)\n"
        "- Ignore low-score or irrelevant retrieved cases\n"
        "- Do not output reasoning steps\n"
        "- Output only the final answer\n\n"
        "Answer:\n"
    )
    content.append({"type": "text", "text": instruction})

    return content


# ============================================================
# DATASET  &  COLLATE
# ============================================================
class MedVQADataset(Dataset):
    """PyTorch Dataset that returns raw items for the collate function."""

    def __init__(
        self,
        df: pd.DataFrame,
        retrieved_contexts: List[List[str]],
        data_dir: Path,
    ):
        self.df = df.reset_index(drop=True)
        self.retrieved_contexts = retrieved_contexts
        self.data_dir = data_dir

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, img_path: Path) -> Image.Image:
        try:
            return Image.open(img_path).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), color="black")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        q_type_raw = str(row.get("question_type", "open")).lower()
        question_type = "CLOSED-ENDED" if q_type_raw == "closed" else "OPEN-ENDED"

        return {
            "query_image": self._load_image(self.data_dir / row["image_path"]),
            "question": str(row["question"]),
            "question_type": question_type,
            "context": self.retrieved_contexts[idx],
            "answer": str(row["answer"]),
        }


def get_collate_fn(processor):
    """Return a collate function that tokenises with the SmolVLM processor."""

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        all_images: List[Image.Image] = []
        user_prompts: List[str] = []
        full_prompts: List[str] = []

        for item in batch:
            all_images.append(item["query_image"])

            mrag_content = build_mrag_content(
                item["question"], item["question_type"], item["context"]
            )

            user_msgs = [{"role": "user", "content": mrag_content}]
            full_msgs = user_msgs + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["answer"]}],
                }
            ]

            user_p = processor.apply_chat_template(
                user_msgs, tokenize=False, add_generation_prompt=True
            )
            full_p = processor.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )

            user_prompts.append(user_p)
            full_prompts.append(full_p)

        # Tokenise full prompts with images
        inputs = processor(
            text=full_prompts, images=all_images, padding=True, return_tensors="pt"
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = input_ids.clone()

        # Mask the prompt portion so loss is computed on the answer only
        for i, user_p in enumerate(user_prompts):
            user_inputs = processor(
                text=[user_p], images=[all_images[i]], return_tensors="pt"
            )
            user_len = user_inputs["input_ids"].shape[1]
            labels[i, :user_len] = -100
            labels[i, attention_mask[i] == 0] = -100

        inputs["labels"] = labels
        return inputs

    return collate_fn


# ============================================================
# HELPER: Load SmolVLM + LoRA
# ============================================================
def _load_model_with_lora(
    model_name: str,
    peft_config: LoraConfig,
    global_lora_state: Optional[Dict[str, torch.Tensor]],
) -> Tuple[Any, Any]:
    """
    Load SmolVLM base model, apply LoRA adapter, optionally inject
    global LoRA weights, and return (model, processor).
    """
    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    processor = AutoProcessor.from_pretrained(model_name)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    base_model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    # Gradient checkpointing to save VRAM
    try:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except Exception:
        pass  # Some models may not support this

    model = get_peft_model(base_model, peft_config)

    # Inject global LoRA weights (skip on very first round)
    if global_lora_state is not None:
        set_peft_model_state_dict(model, global_lora_state)

    return model, processor


def _free_model(*objs) -> None:
    """Delete objects and free GPU memory."""
    for o in objs:
        del o
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# CORE:  train_client_local
# ============================================================
def train_client_local(
    client_id: int,
    client_df: pd.DataFrame,
    client_contexts: List[List[str]],
    data_dir: Path,
    model_name: str,
    peft_config: LoraConfig,
    global_lora_state: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
    lr: float,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> Dict[str, Any]:
    """
    Train one client for exactly 1 local epoch.

    Returns
    -------
    dict with keys: lora_state_dict, train_loss, num_samples
    """
    log(f"  Client {client_id}: Loading SmolVLM + injecting global LoRA...")
    model, processor = _load_model_with_lora(model_name, peft_config, global_lora_state)

    # Print trainable params only for the first client in the first round
    if global_lora_state is None and client_id == 0:
        model.print_trainable_parameters()

    # ---------- Dataset & Loader ----------
    dataset = MedVQADataset(client_df, client_contexts, data_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=get_collate_fn(processor),
    )

    # ---------- Optimizer & Scheduler ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_opt_steps = max(1, len(loader) // gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_opt_steps)),
        num_training_steps=total_opt_steps,
    )

    # ---------- Train 1 epoch ----------
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()

    progress = tqdm(loader, desc=f"  Client {client_id} training", leave=False)
    for step, batch in enumerate(progress):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        epoch_loss += loss.item() * gradient_accumulation_steps

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        progress.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

    avg_loss = epoch_loss / max(len(loader), 1)

    # ---------- Extract LoRA state dict ----------
    lora_state = {
        k: v.cpu().clone() for k, v in get_peft_model_state_dict(model).items()
    }
    num_samples = len(client_df)

    log(f"  Client {client_id}: Avg Loss = {avg_loss:.4f}  |  Samples = {num_samples}")

    # ---------- Free GPU ----------
    _free_model(model, processor, optimizer, scheduler, loader, dataset)

    return {
        "lora_state_dict": lora_state,
        "train_loss": avg_loss,
        "num_samples": num_samples,
    }


# ============================================================
# CORE:  federated_average
# ============================================================
def federated_average(
    client_results: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """
    Weighted FedAvg:
        θ_global = Σ (N_i / N_total) * θ_client_i
    Averaging is done in float32 for numerical stability.
    """
    total_samples = sum(r["num_samples"] for r in client_results)
    reference_keys = client_results[0]["lora_state_dict"].keys()

    avg_state: Dict[str, torch.Tensor] = {}
    for key in reference_keys:
        weighted_sum = torch.zeros_like(
            client_results[0]["lora_state_dict"][key], dtype=torch.float32
        )
        for r in client_results:
            weight = r["num_samples"] / total_samples
            weighted_sum += r["lora_state_dict"][key].float() * weight
        avg_state[key] = weighted_sum

    return avg_state


# ============================================================
# CORE:  evaluate_global
# ============================================================
def evaluate_global(
    global_lora_state: Dict[str, torch.Tensor],
    test_df: pd.DataFrame,
    test_contexts: List[List[str]],
    data_dir: Path,
    model_name: str,
    peft_config: LoraConfig,
    device: torch.device,
    scorer,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Evaluate the aggregated global model on the test set.

    Returns (metrics_dict, predictions_list).
    Metrics: overall_exact_match, closed_accuracy, closed_f1, open_bleu, open_rouge_l
    """
    log("  Loading global model for evaluation...")
    model, processor = _load_model_with_lora(model_name, peft_config, global_lora_state)
    model.eval()

    predictions: List[str] = []

    for idx, row in tqdm(
        test_df.iterrows(), total=len(test_df), desc="  Evaluating", leave=False
    ):
        question = str(row["question"])
        context = test_contexts[idx]
        q_type_raw = str(row.get("question_type", "open")).lower()
        question_type = "CLOSED-ENDED" if q_type_raw == "closed" else "OPEN-ENDED"

        # Load image
        img_path = data_dir / row["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color="black")

        # Build mRAG prompt
        mrag_content = build_mrag_content(question, question_type, context)
        messages = [{"role": "user", "content": mrag_content}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
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
        gen_text = processor.tokenizer.decode(
            output_ids[0][prompt_len:], skip_special_tokens=True
        ).strip()
        predictions.append(gen_text)

    # ---------- Compute metrics ----------
    all_ems: List[float] = []
    closed_ems: List[float] = []
    closed_f1s: List[float] = []
    open_bleus: List[float] = []
    open_rouges: List[float] = []

    for pred, row in zip(predictions, test_df.itertuples()):
        gt = str(row.answer)
        em = compute_exact_match(pred, gt)
        all_ems.append(em)

        if str(row.question_type).lower() == "closed":
            closed_ems.append(em)
            closed_f1s.append(compute_f1(pred, gt))
        else:
            open_bleus.append(compute_bleu(pred, gt))
            open_rouges.append(compute_rouge_l(pred, gt, scorer))

    metrics = {
        "overall_exact_match": float(np.mean(all_ems)) if all_ems else 0.0,
        "closed_accuracy": float(np.mean(closed_ems)) if closed_ems else 0.0,
        "closed_f1": float(np.mean(closed_f1s)) if closed_f1s else 0.0,
        "open_bleu": float(np.mean(open_bleus)) if open_bleus else 0.0,
        "open_rouge_l": float(np.mean(open_rouges)) if open_rouges else 0.0,
    }

    # ---------- Free GPU ----------
    _free_model(model, processor)

    return metrics, predictions


# ============================================================
# BERTSCORE (computed only for the best round)
# ============================================================
def compute_bertscore_for_best(
    test_df: pd.DataFrame,
    predictions: List[str],
) -> float:
    """Compute BERTScore-F1 for open-ended questions only."""
    from bert_score import score as bert_score_fn

    open_mask = test_df["question_type"].str.lower() == "open"
    open_preds = [p for p, m in zip(predictions, open_mask) if m]
    open_gts = [str(a) for a, m in zip(test_df["answer"], open_mask) if m]

    if not open_preds:
        return 0.0

    _, _, F1 = bert_score_fn(open_preds, open_gts, lang="en", verbose=False)
    return float(F1.mean().item())


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated Learning (FedAvg) with SmolVLM + mRAG for Medical VQA"
    )
    parser.add_argument(
        "--setting", type=str, default="iid", choices=["iid", "noniid"],
        help="Federated data setting: 'iid' or 'noniid'",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Dirichlet alpha for Non-IID partitioning (only used when --setting=noniid)",
    )
    parser.add_argument("--num_clients", type=int, default=3, help="Number of clients (K)")
    parser.add_argument("--num_rounds", type=int, default=10, help="Communication rounds")
    parser.add_argument(
        "--data_root", type=str, default="processed_data/VQA-RAD",
        help="Root directory for VQA-RAD processed data",
    )
    parser.add_argument(
        "--model_name", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="SmolVLM model ID on HuggingFace",
    )
    parser.add_argument(
        "--biomed_model_name", type=str,
        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        help="BiomedCLIP model checkpoint",
    )
    parser.add_argument("--k", type=int, default=3, help="Top-k retrieval")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    # --------------------------------------------------------
    # Resolve directory paths
    # --------------------------------------------------------
    data_root = Path(args.data_root)
    centralized_dir = data_root / "centralized"
    data_dir = centralized_dir  # all images live under centralized/images/

    K = args.num_clients
    if args.setting == "iid":
        fed_data_dir = data_root / f"federated_iid/k{K}"
        output_dir = Path(__file__).parent / f"results_federated/iid_k{K}"
    else:
        fed_data_dir = data_root / f"federated_noniid/alpha_{args.alpha}/k{K}"
        output_dir = Path(__file__).parent / f"results_federated/noniid_alpha{args.alpha}_k{K}"

    assert fed_data_dir.exists(), f"Federated data dir not found: {fed_data_dir}"
    assert centralized_dir.exists(), f"Centralized dir not found: {centralized_dir}"

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "embeddings_cache"
    pred_dir = output_dir / "round_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------
    log("=" * 60)
    log("Federated Learning Configuration")
    log("=" * 60)
    log(f"Setting       : {args.setting.upper()}"
        + (f" (α={args.alpha})" if args.setting == "noniid" else ""))
    log(f"Clients (K)   : {K}")
    log(f"Rounds        : {args.num_rounds}")
    log(f"Model         : {args.model_name}")
    log(f"Learning Rate : {args.lr}")
    log(f"Batch Size    : {args.batch_size} (grad accum={args.gradient_accumulation_steps})")
    log(f"Top-k RAG     : {args.k}")
    log(f"Fed Data Dir  : {fed_data_dir}")
    log(f"Output Dir    : {output_dir}")
    log("=" * 60)

    # --------------------------------------------------------
    # Load CSV data
    # --------------------------------------------------------
    # Client files are 1-indexed: client_1.csv, client_2.csv, ...
    client_csv_paths = [fed_data_dir / f"client_{i + 1}.csv" for i in range(K)]
    for p in client_csv_paths:
        assert p.exists(), f"Client CSV not found: {p}"

    client_dfs = [pd.read_csv(p) for p in client_csv_paths]

    centralized_train_csv = centralized_dir / "train.csv"
    test_csv = centralized_dir / "test.csv"
    assert centralized_train_csv.exists(), f"Not found: {centralized_train_csv}"
    assert test_csv.exists(), f"Not found: {test_csv}"

    centralized_train_df = pd.read_csv(centralized_train_csv)
    test_df = pd.read_csv(test_csv)

    for i, df in enumerate(client_dfs):
        log(f"Client {i} (client_{i + 1}.csv): {len(df)} samples")
    log(f"Centralized train: {len(centralized_train_df)} samples")
    log(f"Test set: {len(test_df)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # --------------------------------------------------------
    # Step 1: Precompute all BiomedCLIP embeddings (one-time)
    # --------------------------------------------------------
    client_embeddings, train_embeddings, test_embeddings = precompute_all_embeddings(
        client_csv_paths,
        centralized_train_csv,
        test_csv,
        data_dir,
        cache_dir,
        args.biomed_model_name,
        device,
    )

    # --------------------------------------------------------
    # Step 2: Precompute retrieval contexts (reused every round)
    # --------------------------------------------------------
    log("Computing mRAG retrieval contexts for each client (private FAISS)...")
    client_contexts: List[List[List[str]]] = []
    for i in range(K):
        ctx = retrieve_contexts_for_training(client_dfs[i], client_embeddings[i], args.k)
        client_contexts.append(ctx)
        log(f"  Client {i}: {len(ctx)} context sets ready")

    log("Computing mRAG retrieval contexts for test set (centralized FAISS)...")
    test_contexts = retrieve_contexts_for_test(
        centralized_train_df, train_embeddings, test_embeddings, args.k
    )
    log(f"  Test: {len(test_contexts)} context sets ready")

    # --------------------------------------------------------
    # Step 3: LoRA configuration
    # --------------------------------------------------------
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # --------------------------------------------------------
    # Step 4: Federated Training Loop
    # --------------------------------------------------------
    global_lora_state: Optional[Dict[str, torch.Tensor]] = None
    best_em = -1.0
    best_round = -1
    best_predictions: List[str] = []
    all_round_metrics: List[Dict[str, Any]] = []

    for round_id in range(1, args.num_rounds + 1):
        log(f"\n{'=' * 60}")
        log(f"ROUND {round_id} / {args.num_rounds}")
        log(f"{'=' * 60}")

        # ----- Client Local Training (sequential) -----
        client_results: List[Dict[str, Any]] = []
        for client_id in range(K):
            log(f"\n--- Client {client_id} (client_{client_id + 1}.csv) ---")
            result = train_client_local(
                client_id=client_id,
                client_df=client_dfs[client_id],
                client_contexts=client_contexts[client_id],
                data_dir=data_dir,
                model_name=args.model_name,
                peft_config=peft_config,
                global_lora_state=global_lora_state,
                device=device,
                lr=args.lr,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
            client_results.append(result)

        # ----- FedAvg Aggregation -----
        log("\nAggregating client LoRA weights with FedAvg (weighted by sample count)...")
        weights_info = ", ".join(
            f"C{i}={r['num_samples']}" for i, r in enumerate(client_results)
        )
        log(f"  Sample weights: {weights_info}")
        global_lora_state = federated_average(client_results)

        avg_train_loss = float(np.mean([r["train_loss"] for r in client_results]))
        log(f"  Avg Train Loss (all clients): {avg_train_loss:.4f}")

        # ----- Global Evaluation -----
        log("\nGlobal Evaluation on Test Set...")
        metrics, predictions = evaluate_global(
            global_lora_state,
            test_df,
            test_contexts,
            data_dir,
            args.model_name,
            peft_config,
            device,
            scorer,
        )

        round_info: Dict[str, Any] = {
            "round": round_id,
            "avg_train_loss": avg_train_loss,
            **metrics,
        }
        all_round_metrics.append(round_info)

        log(f"\nRound {round_id} Results:")
        log(f"  Overall EM   : {metrics['overall_exact_match']:.4f}")
        log(f"  Closed Acc   : {metrics['closed_accuracy']:.4f}")
        log(f"  Closed F1    : {metrics['closed_f1']:.4f}")
        log(f"  Open BLEU    : {metrics['open_bleu']:.4f}")
        log(f"  Open ROUGE-L : {metrics['open_rouge_l']:.4f}")

        # ----- Save best model -----
        if metrics["overall_exact_match"] > best_em:
            best_em = metrics["overall_exact_match"]
            best_round = round_id
            best_predictions = predictions.copy()

            log(f"  ★ New best EM = {best_em:.4f} at Round {round_id}. Saving global LoRA...")
            best_lora_dir = output_dir / "best_global_lora"
            best_lora_dir.mkdir(parents=True, exist_ok=True)

            # Temporarily load model to save adapter via PEFT
            model_tmp, _ = _load_model_with_lora(
                args.model_name, peft_config, global_lora_state
            )
            model_tmp.save_pretrained(best_lora_dir)
            _free_model(model_tmp)

        # ----- Persist metrics JSON after every round -----
        with open(output_dir / "fed_metrics.json", "w", encoding="utf-8") as f:
            json.dump(all_round_metrics, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # Step 5: BERTScore for the best round (computed once)
    # --------------------------------------------------------
    log(f"\n{'=' * 60}")
    log(f"Training Complete!  Best Round = {best_round}  (EM = {best_em:.4f})")
    log("Computing BERTScore for best round predictions (open-ended only)...")

    bert_f1 = compute_bertscore_for_best(test_df, best_predictions)
    log(f"Open BERTScore F1: {bert_f1:.4f}")

    # Update the best round entry in metrics
    all_round_metrics[best_round - 1]["open_bertscore"] = bert_f1
    with open(output_dir / "fed_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_round_metrics, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # Step 6: Save best predictions CSV
    # --------------------------------------------------------
    results_df = test_df.copy()
    results_df["prediction"] = best_predictions
    results_df["retrieved_knowledge"] = ["\n".join(ctx) for ctx in test_contexts]
    results_df["EM"] = [
        compute_exact_match(p, str(r.answer))
        for p, r in zip(best_predictions, test_df.itertuples())
    ]
    results_df["F1"] = [
        compute_f1(p, str(r.answer))
        for p, r in zip(best_predictions, test_df.itertuples())
    ]
    results_df["BLEU"] = [
        compute_bleu(p, str(r.answer))
        for p, r in zip(best_predictions, test_df.itertuples())
    ]
    results_df["ROUGE-L"] = [
        compute_rouge_l(p, str(r.answer), scorer)
        for p, r in zip(best_predictions, test_df.itertuples())
    ]
    results_df.to_csv(pred_dir / "final_best_preds.csv", index=False)

    log(f"\nAll results saved to: {output_dir}")
    log(f"  ├── best_global_lora/        (LoRA adapter of best round)")
    log(f"  ├── fed_metrics.json          (metrics history, all rounds)")
    log(f"  ├── embeddings_cache/         (BiomedCLIP embeddings)")
    log(f"  └── round_predictions/")
    log(f"      └── final_best_preds.csv  (detailed predictions)")
    log("Done! ✓")


if __name__ == "__main__":
    main()
