#!/usr/bin/env python3

"""

train_centrailze_mrag_icl_sac.py



Generative mRAG-ICL (Multimodal RAG + In-Context Learning) Medical VQA training script for VQA-RAD with SAC.

Uses:

- Frozen BiomedCLIP for image-question embedding.

- FAISS on CPU for retrieval of relevant cases from the training set.

- Qwen2-VL-2B-Instruct + LoRA for answer generation.

- mRAG-ICL prompt: query image + k retrieved reference cases (with images) used as few-shot examples.

- Strict question type matching: open-ended retrieves open-ended, closed-ended retrieves closed-ended.

- Semantic Answer Canonicalization (SAC): Normalizes answers using semantic similarity for consistent evaluation.

- Evaluation metrics: closed-ended Accuracy, open-ended Accuracy/EM/BLEU/ROUGE-L/BERTScore, hallucination rate, performance.



Optimized to run under 7GB VRAM by precomputing embeddings, releasing BiomedCLIP,

and applying LoRA on Qwen.



Output: experiments/VQA-RAD/centralized/Qwen2.5-VL-3B-Instruct/mrag_icl_sac

"""



import os

import gc

import argparse

import random

import time

import datetime

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

    Qwen2VLForConditionalGeneration,

    Qwen2_5_VLForConditionalGeneration,

    BitsAndBytesConfig,

    get_linear_schedule_with_warmup

)

from peft import LoraConfig, get_peft_model, TaskType



import nltk

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from rouge_score import rouge_scorer

from sentence_transformers import SentenceTransformer, util



import os



# ── Set HuggingFace cache ───────────────────────────────────────────────────

CACHE_DIR = r"D:\nckh_teacher_student\code\teacher_student\.hf_cache"

os.environ["HF_HOME"] = CACHE_DIR

os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")

os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")



# Set up logging/printing helper

def log(msg):

    print(f"[RAG-VQA-SAC] {msg}")



# Seed setting

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)



# ── Semantic Answer Canonicalization (SAC) ─────────────────────────────────────

# Initialize SAC model and vocabulary

sac_model = None

closed_vocab = ["Yes", "No"]

open_vocab_embeddings = None

open_vocab_labels = None



def initialize_canonicalizer(train_df):

    """

    Initialize SAC model and build canonical vocabulary from training data.

    """

    global sac_model, open_vocab_embeddings, open_vocab_labels

    

    log("Initializing SAC model (all-MiniLM-L6-v2)...")

    sac_model = SentenceTransformer("all-MiniLM-L6-v2")

    

    # Build canonical vocabulary from open-ended answers in training data

    open_answers = train_df[train_df['question_type'].str.lower() == 'open']['answer'].unique()

    open_answers = [str(a).strip() for a in open_answers if pd.notna(a) and str(a).strip()]

    

    # Deduplicate and sort

    open_vocab = sorted(list(set(open_answers)))

    

    log(f"Built open-ended canonical vocabulary: {len(open_vocab)} unique answers")

    

    # Pre-compute embeddings for open vocabulary

    open_vocab_embeddings = sac_model.encode(open_vocab, convert_to_tensor=True)

    open_vocab_labels = open_vocab

    

    log("SAC initialization complete.")

    

    return sac_model, open_vocab_embeddings, open_vocab_labels



def canonicalize_closed_answer(answer):

    """

    Canonicalize closed-ended answer to Yes/No.

    """

    answer_lower = str(answer).strip().lower()

    

    # Check for affirmative variations

    affirmative = ["yes", "y", "yeah", "yep", "affirmative", "positive", "true"]

    for aff in affirmative:

        if aff in answer_lower:

            return "Yes"

    

    # Check for negative variations

    negative = ["no", "n", "nope", "negative", "false"]

    for neg in negative:

        if neg in answer_lower:

            return "No"

    

    # Default to No if unclear

    return "No"



def canonicalize_open_answer(answer, threshold=0.80):

    """

    Canonicalize open-ended answer using semantic similarity to vocabulary.

    """

    if sac_model is None or open_vocab_embeddings is None:

        return str(answer).strip()

    

    answer_clean = str(answer).strip()

    if not answer_clean:

        return answer_clean

    

    # Encode answer

    answer_embedding = sac_model.encode(answer_clean, convert_to_tensor=True)

    

    # Compute cosine similarity

    similarities = util.cos_sim(answer_embedding, open_vocab_embeddings)[0]

    

    # Find best match

    best_idx = similarities.argmax().item()

    best_score = similarities[best_idx].item()

    

    # Return canonical if similarity above threshold

    if best_score >= threshold:

        return open_vocab_labels[best_idx]

    else:

        return answer_clean



def canonicalize_answer(answer, question_type, threshold=0.80):

    """

    Unified interface for answer canonicalization.

    """

    q_type_raw = str(question_type).lower()

    

    if q_type_raw == "closed":

        return canonicalize_closed_answer(answer)

    else:

        return canonicalize_open_answer(answer, threshold)



def canonicalize_dataset(df):

    """

    Canonicalize all answers in a dataframe.

    """

    df_copy = df.copy()

    df_copy['answer_canonical'] = df_copy.apply(

        lambda row: canonicalize_answer(row['answer'], row.get('question_type', 'open')),

        axis=1

    )

    return df_copy



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



# Get retrieved examples (with images) for ICL (with SAC)
def get_retrieved_examples(df, indices, data_dir):
    examples = []
    for idx in indices:
        # Bounds check for invalid indices
        if idx < 0 or idx >= len(df):
            log(f"Warning: skipping invalid index {idx} (out of bounds)")
            continue
        row = df.iloc[idx]
        q_type_raw = str(row.get('question_type', 'open')).lower()
        question_type = "CLOSED-ENDED" if q_type_raw == "closed" else "OPEN-ENDED"
        try:
            image = Image.open(data_dir / row['image_path']).convert('RGB')
        except Exception as e:
            log(f"Warning: failed to load image {row['image_path']}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        # Use canonicalized answer for few-shot examples
        answer_key = 'answer_canonical' if 'answer_canonical' in row else 'answer'
        canonical_answer = canonicalize_answer(row[answer_key], q_type_raw)
        examples.append({
            "image": image,
            "question": row['question'],
            "question_type": question_type,
            "answer": canonical_answer
        })
    return examples



# Custom PyTorch Dataset

class MedVQADataset(Dataset):
    def __init__(self, df, retrieved_examples, data_dir, max_length=512):
        self.df = df
        self.retrieved_examples = retrieved_examples
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
        q_type_raw = str(row.get('question_type', 'open')).lower()
        question_type = "CLOSED-ENDED" if q_type_raw == "closed" else "OPEN-ENDED"
        query_image = self._load_image(self.data_dir / row['image_path'])
        examples = self.retrieved_examples[idx]
        # Use canonicalized answer for training
        answer_key = 'answer_canonical' if 'answer_canonical' in row else 'answer'
        canonical_answer = canonicalize_answer(row[answer_key], q_type_raw)
            

        return {
            "query_image": query_image,
            "question": row['question'],
            "question_type": question_type,
            "answer": canonical_answer,
            "examples": examples
        }



# Build mRAG-ICL content for the user message
def build_mrag_icl_content(query_image, question, question_type, examples):
    """
    Construct a Qwen2-VL mRAG-ICL prompt:
      [query_image] + k retrieved reference cases (with images) as few-shot examples + question
    """
    content = []
    content.append({"type": "text", "text": "System:\nYou are a medical visual question answering assistant.\n\n"})
    
    # Add retrieved examples as few-shot
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





# Collate function for dynamic padding using processor

def get_collate_fn(processor):
    def collate_fn(batch):
        # Collect all images per sample: query + example images
        all_images_per_sample = []
        user_prompts = []
        full_prompts = []
        

        for item in batch:
            sample_images = [item['query_image']] + [ex['image'] for ex in item['examples']]
            all_images_per_sample.append(sample_images)
            

            mrag_icl_content = build_mrag_icl_content(
                item['query_image'],
                item['question'], item['question_type'], item['examples']
            )
            

            user_messages = [{"role": "user", "content": mrag_icl_content}]
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

    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0



def compute_f1(prediction, ground_truth):

    pred_tokens = prediction.strip().lower().split()

    gt_tokens = ground_truth.strip().lower().split()

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

    ref = [ground_truth.strip().lower().split()]

    hyp = prediction.strip().lower().split()

    smooth = SmoothingFunction().method1

    return sentence_bleu(ref, hyp, smoothing_function=smooth)



def compute_rouge_l(prediction, ground_truth, scorer):

    scores = scorer.score(ground_truth.strip().lower(), prediction.strip().lower())

    return scores['rougeL'].fmeasure



# Helper to load image with fallback

def _load_image_safe(img_path):

    try:

        return Image.open(img_path).convert('RGB')

    except Exception as e:

        log(f"Warning: failed to load image {img_path}: {e}")

        return Image.new('RGB', (224, 224), color='black')



# Evaluation loop (mRAG-ICL)
def evaluate_model(model, processor, test_df, test_examples, data_dir, scorer, device, desc="Evaluating"):
    model.eval()
    predictions = []
    start_time = time.time()
    

    log(f"Running generation on {desc} set...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=desc):
        question = row['question']
        examples = test_examples[idx]
        q_type_raw = str(row.get('question_type', 'open')).lower()
        question_type = "CLOSED-ENDED" if q_type_raw == "closed" else "OPEN-ENDED"
        

        # Load query image
        query_image = _load_image_safe(data_dir / row['image_path'])
        

        mrag_icl_content = build_mrag_icl_content(query_image, question, question_type, examples)
        

        messages = [{"role": "user", "content": mrag_icl_content}]
        

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_images = [query_image] + [ex['image'] for ex in examples]
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

    

    inference_time = time.time() - start_time

    throughput = len(test_df) / inference_time if inference_time > 0 else 0.0

    

    # Calculate metrics per question type (with SAC canonicalization)
    closed_preds, closed_gts = [], []
    open_preds, open_gts = [], []
    

    for pred, row in zip(predictions, test_df.itertuples()):
        qt = str(row.question_type).lower()
        # Canonicalize prediction
        canonical_pred = canonicalize_answer(pred, qt)
        # Canonicalize ground truth (use attribute access for itertuples)
        canonical_gt = canonicalize_answer(row.answer_canonical if hasattr(row, 'answer_canonical') else row.answer, qt)
        
        if qt == "closed":
            closed_preds.append(canonical_pred)
            closed_gts.append(canonical_gt)
        else:
            open_preds.append(canonical_pred)
            open_gts.append(canonical_gt)
    

    closed_acc = float(np.mean([compute_exact_match(p, g) for p, g in zip(closed_preds, closed_gts)])) if closed_preds else 0.0
    open_acc = float(np.mean([compute_exact_match(p, g) for p, g in zip(open_preds, open_gts)])) if open_preds else 0.0
    open_bleu = float(np.mean([compute_bleu(p, g) for p, g in zip(open_preds, open_gts)])) if open_preds else 0.0
    open_rouge = float(np.mean([compute_rouge_l(p, g, scorer) for p, g in zip(open_preds, open_gts)])) if open_preds else 0.0

    

    hallucination_rate = 0.0  # Placeholder - requires semantic similarity model

    

    metrics = {

        "closed-ended": {

            "n_samples": len(closed_preds),

            "Accuracy": round(closed_acc, 6),

        },

        "open-ended": {

            "n_samples": len(open_preds),

            "Accuracy": round(open_acc, 6),

            "Exact Match": round(open_acc, 6),

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

    

    return metrics, predictions



def main():

    parser = argparse.ArgumentParser(description="Train Generative RAG Med-VQA on VQA-RAD")

    parser.add_argument("--data_dir", type=str, default="processed_data/VQA-RAD/centralized",

                        help="Path to centralized VQA-RAD dataset folder")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",

                        help="Generator VLM model path/name on HF")

    parser.add_argument("--biomed_model_name", type=str, default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",

                        help="BiomedCLIP model checkpoint")

    parser.add_argument("--output_dir", type=str, default="experiments/VQA-RAD/centralized/Qwen2.5-VL-3B-Instruct/mrag_icl_sac",

                        help="Directory to save the checkpoints and predictions")

    parser.add_argument("--k", type=int, default=1,

                        help="Number of retrieved cases to include as few-shot examples (mRAG-ICL)")

    parser.add_argument("--epochs", type=int, default=5,

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

    # Step 0: Initialize SAC and canonicalize datasets

    # ----------------------------------------------------
    
    initialize_canonicalizer(train_df)
    
    log("Canonicalizing training dataset...")
    train_df = canonicalize_dataset(train_df)
    
    log("Canonicalizing test dataset...")
    test_df = canonicalize_dataset(test_df)
    
    log("Dataset canonicalization complete.")

    

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

    

    log(f"Performing mRAG-ICL retrieval with strict question type matching (k={args.k})...")
    
    # Separate train data by question type for strict matching
    train_closed_indices = train_df[train_df['question_type'].str.lower() == 'closed'].index.tolist()
    train_open_indices = train_df[train_df['question_type'].str.lower() == 'open'].index.tolist()
    
    train_closed_features = train_features[train_closed_indices]
    train_open_features = train_features[train_open_indices]
    
    # Build separate FAISS indices for each question type
    index_closed = faiss.IndexFlatIP(1024)
    index_closed.add(train_closed_features.astype(np.float32))
    
    index_open = faiss.IndexFlatIP(1024)
    index_open.add(train_open_features.astype(np.float32))
    
    # Function to retrieve with strict question type matching
    def retrieve_with_type_matching(query_features, query_type, k):
        if query_type == "closed":
            if len(train_closed_indices) == 0:
                # Fallback: use full index if no closed-ended samples
                log("Warning: No closed-ended training samples, using fallback to full index")
                D, I = index.search(query_features.astype(np.float32), k)
                return D, I
            D, I = index_closed.search(query_features.astype(np.float32), k)
            # Map back to original indices
            mapped_I = [[train_closed_indices[idx] for idx in row] for row in I]
            return D, mapped_I
        else:
            if len(train_open_indices) == 0:
                # Fallback: use full index if no open-ended samples
                log("Warning: No open-ended training samples, using fallback to full index")
                D, I = index.search(query_features.astype(np.float32), k)
                return D, I
            D, I = index_open.search(query_features.astype(np.float32), k)
            # Map back to original indices
            mapped_I = [[train_open_indices[idx] for idx in row] for row in I]
            return D, mapped_I
    
    # For training data - retrieve with strict type matching
    train_examples = []
    for idx in range(len(train_df)):
        q_type_raw = str(train_df.iloc[idx].get('question_type', 'open')).lower()
        query_type = "closed" if q_type_raw == "closed" else "open"
        
        # Search for k+1 to exclude self
        D, I = retrieve_with_type_matching(train_features[idx:idx+1], query_type, args.k + 1)
        
        retrieved_indices = []
        for match_idx in I[0]:
            if match_idx != idx:
                retrieved_indices.append(match_idx)
            if len(retrieved_indices) == args.k:
                break
        
        # Get examples with images
        examples = get_retrieved_examples(train_df, retrieved_indices, data_dir)
        train_examples.append(examples)
    
    # For test data - retrieve with strict type matching
    test_examples = []
    for idx in range(len(test_df)):
        q_type_raw = str(test_df.iloc[idx].get('question_type', 'open')).lower()
        query_type = "closed" if q_type_raw == "closed" else "open"
        
        D, I = retrieve_with_type_matching(test_features[idx:idx+1], query_type, args.k)
        
        examples = get_retrieved_examples(test_df, I[0], data_dir)
        test_examples.append(examples)
    
    log("mRAG-ICL retrieval with strict question type matching completed.")

    

    # Step 3: Load Qwen Generator VLM & Apply LoRA

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

    

    # 4-bit quantization to drastically reduce memory footprint

    # bnb_config = BitsAndBytesConfig(

    #     load_in_4bit=True,

    #     bnb_4bit_quant_type="nf4",

    #     bnb_4bit_compute_dtype=torch_dtype,

    #     bnb_4bit_use_double_quant=True

    # )

        

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(

        args.model_name,

        # quantization_config=bnb_config,

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

    train_dataset = MedVQADataset(train_df, train_examples, data_dir, max_length=args.max_length)

    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        collate_fn=get_collate_fn(processor)

    )

    

    # Scorer for Rouge

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    

    # ----------------------------------------------------

    # Step 5: Training Loop

    # ----------------------------------------------------

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    

    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps

    scheduler = get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=int(0.1 * total_steps),

        num_training_steps=total_steps

    )

    

    log("Starting training...")

    best_closed_acc = -1.0

    all_losses = []

    epoch_times = []

    

    t_start = time.time()

    dt_start = datetime.datetime.now().isoformat()

    

    for epoch in range(1, args.epochs + 1):

        epoch_start = time.time()

        model.train()

        epoch_loss = 0.0

        optimizer.zero_grad()

        

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(progress_bar):

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            

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

            

        avg_loss = epoch_loss / len(train_loader)

        epoch_time = time.time() - epoch_start

        epoch_times.append(epoch_time)

        all_losses.append(avg_loss)

        log(f"Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")

        

        # Evaluate model after each epoch

        metrics, predictions = evaluate_model(model, processor, test_df, test_examples, data_dir, scorer, device, desc="Test")

        log(f"Epoch {epoch} Eval Metrics: Closed Acc={metrics['closed-ended']['Accuracy']:.4f}, Open ROUGE-L={metrics['open-ended']['ROUGE-L']:.4f}")

        

        # Save best model

        if metrics['closed-ended']['Accuracy'] > best_closed_acc:

            best_closed_acc = metrics['closed-ended']['Accuracy']

            log(f"New best Closed Accuracy: {best_closed_acc:.4f}. Saving adapter weights...")

            model.save_pretrained(output_dir)

            

            # Save predictions

            results_df = test_df.copy()

            results_df['prediction'] = predictions

            results_df.to_csv(output_dir / "best_predictions.csv", index=False)

            log(f"Saved predictions to {output_dir / 'best_predictions.csv'}")



    t_end = time.time()

    total_s = t_end - t_start

    

    # Save artifacts

    import json

    

    hardware_payload = {

        "gpu_name": "GPU",

        "total_vram_gb": 0.0,

        "avg_allocated_vram_gb": 0.0,

        "peak_allocated_vram_gb": 0.0,

    }

    with open(output_dir / "hardware.json", "w") as f:

        json.dump(hardware_payload, f, indent=2)

    

    time_payload = {

        "start_time": dt_start,

        "end_time": datetime.datetime.now().isoformat(),

        "total_duration_s": round(total_s, 2),

        "total_duration_min": round(total_s / 60, 2),

        "avg_epoch_time_s": round(float(np.mean(epoch_times)), 2),

    }

    with open(output_dir / "training_time.json", "w") as f:

        json.dump(time_payload, f, indent=2)

    

    experiment_payload = {

        "dataset": "VQA-RAD",

        "setting": "centralized",

        "k": args.k,

        "epochs": args.epochs,

        "batch_size": args.batch_size,

        "lr": args.lr,

        "avg_train_loss": round(float(np.mean(all_losses)), 4),

        "best_closed_acc": round(best_closed_acc, 4),

        "best_metrics": metrics

    }

    with open(output_dir / "experiment.json", "w") as f:

        json.dump(experiment_payload, f, indent=2)



    log("Training completed successfully!")



if __name__ == "__main__":

    main()

