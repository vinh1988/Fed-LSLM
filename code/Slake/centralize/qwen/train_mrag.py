#!/usr/bin/env python3
"""
train_mrag.py

Generative mRAG (Multimodal RAG) Medical VQA training script for SLAKE.
Uses:
- Frozen BiomedCLIP for image-question embedding.
- FAISS on CPU for retrieval of relevant cases from the training set.
- SmolVLM + LoRA for answer generation.
- Text-only mRAG prompt: query image + k retrieved reference Q&A pairs (no retrieved images).
- Evaluation metrics: Exact Match (EM), F1 Score, BLEU, ROUGE-L.

Optimized to run under 7GB VRAM by precomputing embeddings, releasing BiomedCLIP,
and applying LoRA on SmolVLM.
"""

import os
import gc
import argparse
import random
import time
import warnings

# Suppress transformers warnings
warnings.filterwarnings("ignore", message=".*Kwargs passed to.*")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from torch.utils.data import Dataset, DataLoader
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import Counter

import open_clip
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# ── Set HuggingFace cache ───────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".hf_cache")

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")


# Set up logging/printing helper
def log(msg):
    print(f"[RAG-VQA] {msg}")

# Seed setting
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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


# Custom PyTorch Dataset
class MedVQADataset(Dataset):
    def __init__(self, df, retrieved_contexts, data_dir, max_length=512):
        self.df = df
        self.retrieved_contexts = retrieved_contexts
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def _load_image(self, img_path):
        try:
            return Image.open(img_path).convert('RGB')
        except Exception as e:
            log(f"Warning: failed to load image {img_path}: {e}")
            return Image.new('RGB', (224, 224), color='black')
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row['question']
        context = self.retrieved_contexts[idx]
        answer = row['answer']
        
        # Determine question type
        q_type_raw = str(row.get('question_type', 'open')).lower()
        question_type = "CLOSED-ENDED" if q_type_raw == "closed" else "OPEN-ENDED"
        
        # Load SLAKE metadata
        body_part = row.get('body_part', '')
        modality = row.get('modality', '')
        objects = row.get('objects', '')
        
        # Load query image
        image_path = row['image_path']
        if pd.isna(image_path) or not isinstance(image_path, (str, Path)):
            raise ValueError(f"Invalid image_path: {image_path} at row {idx}. Row data: {row.to_dict()}")
        query_image = self._load_image(self.data_dir / str(image_path))
            
        return {
            "query_image": query_image,
            "question": question,
            "question_type": question_type,
            "context": context,
            "answer": answer,
            "body_part": body_part,
            "modality": modality,
            "objects": objects
        }

# Build mRAG content for the user message (text-only retrieval)
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
        "  • The final answer MUST be \"yes\" or \"no\"\n"
        "  • Use the medical context and detected objects to inform your decision\n"
        "  • Focus primarily on the image\n"
        "  • Use retrieved QA cases only if they strongly match the visual evidence\n\n"
        "- If the question is OPEN-ENDED:\n"
        "  • Use the image, body part, modality, and detected objects to provide an accurate answer\n"
        "  • Consider the imaging modality when interpreting findings\n"
        "  • Use retrieved QA cases to improve reasoning and completeness\n\n"
        "General rules:\n"
        "- Retrieved cases are ranked by similarity score (higher score = more relevant)\n"
        "- Ignore low-score or irrelevant retrieved cases\n"
        "- Do not output reasoning steps\n"
        "- Output only the final answer\n\n"
        "Answer:\n"
    )
    content.append({"type": "text", "text": instruction})
    
    return content


# Collate function for dynamic padding using processor
def get_collate_fn(processor):
    def collate_fn(batch):
        # Collect all images per sample: query only (no retrieved images)
        all_images_per_sample = []
        user_prompts = []
        full_prompts = []
        
        for item in batch:
            sample_images = [item['query_image']]
            all_images_per_sample.append(sample_images)
            
            mrag_content = build_mrag_content(
                item['query_image'],
                item['question'],
                item['question_type'],
                item['body_part'],
                item['modality'],
                item['objects'],
                item['context']
            )
            
            user_messages = [{"role": "user", "content": mrag_content}]
            full_messages = user_messages + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(item['answer'])}]
                }
            ]
            
            user_p = processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
            full_p = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            
            user_prompts.append(user_p)
            full_prompts.append(full_p)
        
        # Flatten all images for the processor (it expects a flat list)
        flat_images = [img for sample_imgs in all_images_per_sample for img in sample_imgs]
        
        inputs = processor(text=full_prompts, images=flat_images, padding=True, return_tensors="pt")
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = input_ids.clone()
        
        for i, user_p in enumerate(user_prompts):
            inputs_user = processor(text=[user_p], images=all_images_per_sample[i], return_tensors="pt")
            user_len = inputs_user['input_ids'].shape[1]
            labels[i, :user_len] = -100
            labels[i, attention_mask[i] == 0] = -100
            
        inputs['labels'] = labels
        return inputs
    return collate_fn

# Evaluation metric helpers
def compute_exact_match(prediction, ground_truth):
    gt_str = str(ground_truth).strip() if ground_truth is not None else ""
    pred_str = str(prediction).strip() if prediction is not None else ""
    return 1.0 if pred_str.lower() == gt_str.lower() else 0.0

def compute_f1(prediction, ground_truth):
    gt_str = str(ground_truth).strip() if ground_truth is not None else ""
    pred_str = str(prediction).strip() if prediction is not None else ""
    pred_tokens = pred_str.lower().split()
    gt_tokens = gt_str.lower().split()
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_bleu(prediction, ground_truth):
    gt_str = str(ground_truth).strip() if ground_truth is not None else ""
    pred_str = str(prediction).strip() if prediction is not None else ""
    ref = [gt_str.lower().split()]
    hyp = pred_str.lower().split()
    smooth = SmoothingFunction().method1
    return sentence_bleu(ref, hyp, smoothing_function=smooth)

def compute_rouge_l(prediction, ground_truth, scorer):
    gt_str = str(ground_truth).strip() if ground_truth is not None else ""
    pred_str = str(prediction).strip() if prediction is not None else ""
    scores = scorer.score(gt_str.lower(), pred_str.lower())
    return scores['rougeL'].fmeasure

# Helper to load image with fallback
def _load_image_safe(img_path):
    try:
        return Image.open(img_path).convert('RGB')
    except Exception as e:
        log(f"Warning: failed to load image {img_path}: {e}")
        return Image.new('RGB', (224, 224), color='black')

# Evaluation loop
def evaluate_model(model, processor, test_df, test_contexts, data_dir, scorer, device, desc="Evaluating"):
    model.eval()
    predictions = []
    
    log(f"Running generation on {desc} set...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=desc):
        question = row['question']
        context = test_contexts[idx]
        
        # Determine question type
        q_type_raw = str(row.get('question_type', 'open')).lower()
        question_type = "CLOSED-ENDED" if q_type_raw == "closed" else "OPEN-ENDED"
        
        # Get medical context fields
        body_part = row.get('body_part', '')
        modality = row.get('modality', '')
        objects = row.get('objects', '')
        
        # Load query image
        query_image = _load_image_safe(data_dir / row['image_path'])
        
        mrag_content = build_mrag_content(query_image, question, question_type, body_part, modality, objects, context)
        
        messages = [{"role": "user", "content": mrag_content}]
        
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_images = [query_image]
        inputs = processor(text=[prompt], images=all_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                do_sample=False
            )
            
        # Decode only the generated part
        prompt_len = inputs['input_ids'].shape[1]
        gen_text = processor.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
        predictions.append(gen_text)
        
    # Calculate metrics
    ems = []
    f1s = []
    bleus = []
    rouges = []
    
    for pred, row in zip(predictions, test_df.itertuples()):
        gt = row.answer
        ems.append(compute_exact_match(pred, gt))
        f1s.append(compute_f1(pred, gt))
        bleus.append(compute_bleu(pred, gt))
        rouges.append(compute_rouge_l(pred, gt, scorer))
        
    metrics = {
        "EM": np.mean(ems),
        "F1": np.mean(f1s),
        "BLEU": np.mean(bleus),
        "ROUGE-L": np.mean(rouges)
    }
    
    return metrics, predictions

def main():
    parser = argparse.ArgumentParser(description="Train mRAG Med-VQA on SLAKE")
    parser.add_argument("--data_dir", type=str, 
                        default=os.path.join(PROJECT_ROOT, "processed_data", "Slake", "centralized"),
                        help="Path to centralized SLAKE dataset folder")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
                        help="Generator VLM model path/name on HF")
    parser.add_argument("--biomed_model_name", type=str, default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                        help="BiomedCLIP model checkpoint")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of retrieved cases for RAG")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the checkpoints and predictions (auto-generated if None)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for LoRA training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Resolve dynamic output directory based on data_dir and model_name
    if args.output_dir is None:
        data_path = Path(args.data_dir)
        parts = data_path.parts
        if len(parts) >= 2:
            dataset = parts[-2]
            setting = parts[-1]
        else:
            dataset = "Slake"
            setting = "centralized"
        model_short = args.model_name.split("/")[-1]
        args.output_dir = os.path.join(PROJECT_ROOT, "experiments", dataset, setting, model_short, "mrag")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    
    assert train_csv.exists(), f"Train CSV not found at {train_csv}"
    assert test_csv.exists(), f"Test CSV not found at {test_csv}"
    
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    
    # ----------------------------------------------------
    # Step 1: Precompute Embeddings using BiomedCLIP
    # ----------------------------------------------------
    train_cache = output_dir / "train_biomed_embeddings.npy"
    test_cache = output_dir / "test_biomed_embeddings.npy"
    
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
        train_features = precompute_embeddings(train_csv, data_dir, biomed_model, preprocess, biomed_tokenizer, device)
        np.save(train_cache, train_features)
        
        log("Computing test embeddings...")
        test_features = precompute_embeddings(test_csv, data_dir, biomed_model, preprocess, biomed_tokenizer, device)
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
    # For training data, we query the index but exclude the query sample itself to prevent ground-truth leakage.
    train_contexts = []
    # Search for k+1 neighbors in training set
    D_train, I_train = index.search(train_features, args.k + 1)
    for idx in range(len(train_df)):
        retrieved_indices = []
        retrieved_scores = []
        for match_idx_pos, match_idx in enumerate(I_train[idx]):
            if match_idx == idx:
                continue
            retrieved_indices.append(match_idx)
            retrieved_scores.append(D_train[idx][match_idx_pos])
            if len(retrieved_indices) == args.k:
                break
        # Fallback if somehow len < k
        if len(retrieved_indices) < args.k:
            for match_idx_pos, match_idx in enumerate(I_train[idx]):
                if match_idx not in retrieved_indices:
                    retrieved_indices.append(match_idx)
                    retrieved_scores.append(D_train[idx][match_idx_pos])
                if len(retrieved_indices) == args.k:
                    break
        train_contexts.append(format_retrieved_knowledge(train_df, retrieved_indices, retrieved_scores))
        
    # For test data, we just retrieve top-k closest from training set
    test_contexts = []
    D_test, I_test = index.search(test_features, args.k)
    for idx in range(len(test_df)):
        test_contexts.append(format_retrieved_knowledge(train_df, I_test[idx], D_test[idx]))
        
    log("RAG retrieval contexts (text-only) prepared successfully.")
    
    # ----------------------------------------------------
    # Step 3: Load SmolVLM Generator VLM & Apply LoRA
    # ----------------------------------------------------
    # Set model dtype to float16 / bfloat16 to fit under VRAM limit
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    log(f"Loading processor & VLM: {args.model_name}...")
    # Constrain image resolution to reduce vision token count and VRAM usage
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        min_pixels=128*28*28,
        max_pixels=256*28*28,
        cache_dir=CACHE_DIR
    )
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
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

    if "qwen2.5" in args.model_name.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    elif "smolvlm" in args.model_name.lower() or "idefics3" in args.model_name.lower():
        from transformers import Idefics3ForConditionalGeneration as ModelClass
    else:
        from transformers import Qwen2VLForConditionalGeneration as ModelClass

    base_model = ModelClass.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        cache_dir=CACHE_DIR
    )
    
    # Enable gradient checkpointing to trade compute for memory
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    # PEFT LoRA configuration
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    
    # ----------------------------------------------------
    # Step 4: Setup Dataset & DataLoader
    # ----------------------------------------------------
    # Limit to first 10 samples for quick testing
    # train_df = train_df.head(30)
    # test_df = test_df.head(10)
    # train_contexts = train_contexts[:30]
    # test_contexts = test_contexts[:10]
    
    train_dataset = MedVQADataset(train_df, train_contexts, data_dir, max_length=args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=get_collate_fn(processor)
    )
    
    # Scorer for Rouge
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # ----------------------------------------------------
    # Step 3: Training Loop
    # ----------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    log("Starting training...")
    best_em = -1.0
    epoch_times = []
    epoch_vram_usage = []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        epoch_start_time = time.time()
        vram_samples = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Track VRAM usage
            if torch.cuda.is_available():
                vram_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                vram_samples.append(vram_allocated)
            
            outputs = model(**batch)
            
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.set_postfix({"loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}"})
            
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        avg_vram = sum(vram_samples) / len(vram_samples) if vram_samples else 0.0
        epoch_vram_usage.append(avg_vram)
        
        avg_loss = epoch_loss / len(train_loader)
        log(f"Epoch {epoch} finished. Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s, Avg VRAM: {avg_vram:.2f}GB")
        
        # Evaluate model after each epoch
        metrics, predictions = evaluate_model(model, processor, test_df, test_contexts, data_dir, scorer, device, desc="Test")
        log(f"Epoch {epoch} Eval Metrics: EM={metrics['EM']:.4f}, F1={metrics['F1']:.4f}, BLEU={metrics['BLEU']:.4f}, ROUGE-L={metrics['ROUGE-L']:.4f}")
        
        # Save best model
        if metrics['EM'] > best_em:
            best_em = metrics['EM']
            log(f"New best Exact Match: {best_em:.4f}. Saving adapter weights...")
            model.save_pretrained(output_dir / "best_lora_adapter")
            
            # Save predictions
            results_df = test_df.copy()
            results_df['prediction'] = predictions
            results_df['EM'] = [compute_exact_match(p, r.answer) for p, r in zip(predictions, test_df.itertuples())]
            results_df['F1'] = [compute_f1(p, r.answer) for p, r in zip(predictions, test_df.itertuples())]
            results_df['BLEU'] = [compute_bleu(p, r.answer) for p, r in zip(predictions, test_df.itertuples())]
            results_df['ROUGE-L'] = [compute_rouge_l(p, r.answer, scorer) for p, r in zip(predictions, test_df.itertuples())]
            results_df.to_csv(output_dir / "best_predictions.csv", index=False)
            log(f"Saved predictions and scores to {output_dir / 'best_predictions.csv'}")
            
            # Save training performance metrics
            avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
            avg_vram = sum(epoch_vram_usage) / len(epoch_vram_usage) if epoch_vram_usage else 0.0
            
            training_metrics = {
                "avg_epoch_time_s": avg_epoch_time,
                "avg_allocated_vram_gb": avg_vram,
                "epoch_times": epoch_times,
                "epoch_vram_usage_gb": epoch_vram_usage
            }
            
            import json
            with open(output_dir / "best_lora_adapter" / "training_performance.json", "w") as f:
                json.dump(training_metrics, f, indent=2)
            log(f"Saved training performance metrics to {output_dir / 'best_lora_adapter' / 'training_performance.json'}")

    log("Training completed successfully!")

if __name__ == "__main__":
    main()
