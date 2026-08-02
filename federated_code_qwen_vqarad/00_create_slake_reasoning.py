import os
import time
import requests
import zipfile
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
# from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

# =====================================================
# CONFIG
# =====================================================

CACHE_DIR = r"D:\nckh_teacher_student\code\teacher_student\.hf_cache"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

OUT_DIR = Path(
    r"D:\nckh_teacher_student\code\teacher_student\processed_data\Slake"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Image directory for downloaded Slake images
IMAGE_DIR = OUT_DIR / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# MAX_NEW_TOKENS = 512

MAX_NEW_TOKENS = 192

# Output CSV for closed-ended reasoning
OUTPUT_CSV = OUT_DIR / "slake_closed_reasoning.csv"

# =====================================================
# HELPER: format seconds to human-readable string
# =====================================================

def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m {s}s"

def download_and_extract_slake_images(image_dir: Path):
    """Download and extract imgs.zip from HuggingFace"""
    zip_path = OUT_DIR / "imgs.zip"
    
    if zip_path.exists() and (image_dir / "train").exists():
        print(f"  [SKIP] Images already downloaded and extracted")
        return
    
    print(f"  [DOWNLOAD] Downloading imgs.zip from HuggingFace...")
    zip_url = "https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip"
    
    try:
        response = requests.get(zip_url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"  [EXTRACT] Extracting imgs.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(image_dir)
        
        print(f"  [DONE] Images extracted to {image_dir}")
    except Exception as e:
        print(f"  [ERROR] Failed to download/extract images: {e}")
        raise

def load_slake_image(img_name: str, image_dir: Path) -> Image.Image | None:
    """Load image from extracted Slake images directory"""
    # img_name format: xmlab1/source.jpg
    # Extracted structure: image_dir/imgs/xmlab1/source.jpg
    local_path = image_dir / "imgs" / img_name
    
    if local_path.exists():
        return Image.open(local_path).convert("RGB")
    else:
        return None

# =====================================================
# LOAD DATASET
# =====================================================

print("Loading Slake dataset...")

dataset = load_dataset(
    "BoKelvin/SLAKE",
    cache_dir=CACHE_DIR
)

print(f"  Total samples: {len(dataset['train'])}")
print(f"  Available columns: {dataset['train'].column_names}")

# =====================================================
# LOAD MODEL
# =====================================================
print("Loading Qwen2.5-VL...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR
)

torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    device_map="auto",
    cache_dir=CACHE_DIR,
)
model.eval()

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    print(f"Device: CUDA ({num_gpus} GPU(s))")
    for i, name in enumerate(gpu_names):
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  GPU {i}: {name} | VRAM: {mem:.1f} GB")
else:
    print("Device: CPU")

print("Model ready.")

# =====================================================
# PROMPT (IMPORTANT FIXED VERSION)
# =====================================================

SYSTEM_PROMPT = """
You are a professional radiologist.

IMPORTANT:
- Only describe visible findings in the image.
- Do NOT hallucinate medical details.
- If uncertain, say "uncertain".
- For closed-ended questions (Yes/No), provide clear reasoning.
- The Final Answer must be the exact answer (Yes/No or the specific answer provided).

You MUST follow this exact format for your response:

Visual Findings:
[Describe what you see in the image]

Clinical Interpretation:
[Provide clinical interpretation]

Reasoning:
[Explain step-by-step why the answer is correct]

Final Answer: [exact answer from the provided answer]

Do NOT use "Step 1", "Step 2", etc. Use the exact format above.
"""

def generate_reasoning(image, question, answer):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": f"""
Question:
{question}

Answer:
{answer}

Task:
Explain step-by-step why this answer is correct for the question.
Focus only on visible evidence in the image.
"""
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    )

    # inputs = {k: v.to(model.device) for k, v in inputs.items()}
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}   

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            top_p=1.0
        )

    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]

    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

# =====================================================
# PROCESS CLOSED-ENDED QUESTIONS (limit to 1k)
# =====================================================

def validate_reasoning(reasoning: str, answer: str) -> bool:
    """Check if reasoning contains the correct answer and is complete"""
    answer_lower = answer.lower().strip()
    reasoning_lower = reasoning.lower()
    
    # Check if reasoning is too short or incomplete (likely cut off)
    if len(reasoning) < 50:
        return False
    
    # Check if reasoning follows the required format
    required_headers = ['visual findings:', 'clinical interpretation:', 'reasoning:', 'final answer:']
    for header in required_headers:
        if header not in reasoning_lower:
            return False
    
    # Reject if model used "Step" format instead of required format
    if 'step 1:' in reasoning_lower or 'step 2:' in reasoning_lower:
        return False
    
    # Check if reasoning ends abruptly (common signs of cutoff)
    cutoff_indicators = [
        'the heart size', 'the lung', 'the diaphragm', 'the mediastinum', 'the ribs',
        'while there are', 'there are no', 'the presence of', 'the absence of',
        'this finding', 'based on', 'given the', 'the structure', 'the image shows'
    ]
    for indicator in cutoff_indicators:
        if reasoning_lower.strip().endswith(indicator):
            return False
    
    # Check if reasoning ends with incomplete sentence (no ending punctuation)
    # and is relatively short - likely cut off
    last_char = reasoning.strip()[-1] if reasoning.strip() else ''
    if last_char not in ['.', '!', '?', ':'] and len(reasoning) < 300:
        return False
    
    # Try to extract Final Answer line
    final_answer = None
    if "final answer:" in reasoning_lower:
        # Extract text after "final answer:"
        idx = reasoning_lower.index("final answer:")
        final_answer_text = reasoning[idx + len("final answer:"):].strip()
        final_answer = final_answer_text.lower()
    
    # If Final Answer line exists, check if it contains the expected answer
    if final_answer:
        # Direct match or substring match
        if answer_lower in final_answer or final_answer in answer_lower:
            return True
        # For Yes/No, check exact match or common variations
        if answer_lower == "yes" and ("yes" in final_answer or "affirmative" in final_answer):
            return True
        if answer_lower == "no" and ("no" in final_answer or "negative" in final_answer):
            return True
    
    return False

def process_closed_ended(dataset, output_csv: Path, max_samples=1000):
    # Filter for closed-ended questions with Yes/No answers only
    closed_samples = [
        s for s in dataset['train'] 
        if s.get('answer_type', '').upper() == 'CLOSED' 
        and s.get('answer', '').lower() in ['yes', 'no']
    ]
    print(f"Found {len(closed_samples)} closed-ended Yes/No questions")
    
    # Print first sample keys for debugging
    if len(closed_samples) > 0:
        print(f"First sample keys: {closed_samples[0].keys()}")
    
    # ── Resume ──
    valid_count = 0
    processed_count = 0
    if output_csv.exists():
        existing_df = pd.read_csv(output_csv, encoding="utf-8")
        valid_count = len(existing_df)
        print(f"  [RESUME] Found {output_csv.name} with {valid_count} valid rows — resuming")
    else:
        print(f"  [FRESH] No existing CSV — starting from scratch")

    if valid_count >= max_samples:
        print(f"  [DONE] Already have {valid_count} valid samples. Skipping.\n")
        return

    remaining_valid = max_samples - valid_count
    print(f"  Need {remaining_valid} more valid samples (target: {max_samples})\n")

    t_start = time.time()
    skipped_count = 0

    for i, sample in enumerate(closed_samples):
        if valid_count >= max_samples:
            break
        
        processed_count += 1
        img_name = sample["img_name"]
        question = sample["question"]
        answer = sample["answer"]

        # Load image
        image = load_slake_image(img_name, IMAGE_DIR)
        if image is None:
            print(f"[SKIP {processed_count}] Image not found: {img_name}")
            skipped_count += 1
            continue

        t_sample_start = time.time()

        print(f"[Processing {processed_count}] Valid: {valid_count}/{max_samples} | Question: {question[:120]}")

        reasoning = generate_reasoning(image, question, answer)

        t_sample_end = time.time()
        sample_time = t_sample_end - t_sample_start

        # Validate reasoning
        if not validate_reasoning(reasoning, answer):
            print(f"  [SKIP] Reasoning doesn't match answer '{answer}'")
            skipped_count += 1
            continue

        # Valid sample - save to CSV
        row = {
            "img_id": sample.get("img_id", str(i)),
            "img_name": sample.get("img_name", ""),
            "question": question,
            "answer": answer,
            "reasoning": reasoning
        }
        row_df = pd.DataFrame([row])
        write_header = not output_csv.exists() or os.path.getsize(output_csv) == 0
        row_df.to_csv(output_csv, mode="a", header=write_header, index=False, encoding="utf-8")
        
        valid_count += 1

        # ── ETA ──
        elapsed = t_sample_end - t_start
        avg_per_sample = elapsed / valid_count if valid_count > 0 else 0
        left = max_samples - valid_count
        eta = avg_per_sample * left if avg_per_sample > 0 else 0

        print(f"  reasoning={reasoning[:200]}...")
        print(f"  [VALID {valid_count}/{max_samples}] sample={sample_time:.1f}s | elapsed={fmt_time(elapsed)} | avg={avg_per_sample:.1f}s/sample | ETA={fmt_time(eta)} ({left} left) | skipped={skipped_count}\n")

    total_time = time.time() - t_start
    print(f"[Closed] Finished! Valid samples: {valid_count}/{max_samples} | Skipped: {skipped_count} | Processed: {processed_count}")
    print(f"[Closed] Total time: {fmt_time(total_time)}")
    print(f"[Closed] Saved to: {output_csv}\n")

# =====================================================
# RUN
# =====================================================

print(f"\n{'='*60}")
print(f"  Processing closed-ended questions (max 1k)")
print(f"{'='*60}")

# Download and extract images first
print("Downloading Slake images...")
download_and_extract_slake_images(IMAGE_DIR)

process_closed_ended(dataset, OUTPUT_CSV, max_samples=1000)

print("\nDone!")