import os
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# =====================================================
# CONFIG
# =====================================================

CACHE_DIR = r"D:\nckh_teacher_student\code\teacher_student\.hf_cache"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

OUT_DIR = Path(
    r"D:\nckh_teacher_student\code\teacher_student\processed_data\ROCO"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_NEW_TOKENS = 512

# Output CSVs for each split
SPLITS_CONFIG = {
    "train": OUT_DIR / "roco_reasoning_train.csv",
    "validation": OUT_DIR / "roco_reasoning_validation.csv",
}

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

# =====================================================
# LOAD DATASET
# =====================================================

print("Loading ROCOv2 dataset...")

dataset = load_dataset(
    "eltorio/ROCOv2-radiology",
    cache_dir=CACHE_DIR
)

for split_name in SPLITS_CONFIG:
    print(f"  {split_name}: {len(dataset[split_name])} samples")

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading Qwen2.5-VL...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    cache_dir=CACHE_DIR
)

model.eval()

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

Return:

Visual Findings:
Clinical Interpretation:
Reasoning:
Final Answer:
"""

def generate_reasoning(image, caption):

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
Medical Caption:
{caption}

Task:
Explain step-by-step why this caption matches the image.
Focus only on visible evidence.
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

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

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
# PROCESS EACH SPLIT (train + validation)
# =====================================================

def process_split(split_name: str, split_ds, output_csv: Path):
    num_samples = len(split_ds)

    # ── Resume ──
    start_index = 0
    if output_csv.exists():
        existing_df = pd.read_csv(output_csv, encoding="utf-8")
        start_index = len(existing_df)
        print(f"  [RESUME] Found {output_csv.name} with {start_index} rows — resuming from index {start_index}")
    else:
        print(f"  [FRESH] No existing CSV — starting from scratch")

    if start_index >= num_samples:
        print(f"  [DONE] All {num_samples} samples already processed. Skipping.\n")
        return

    remaining = num_samples - start_index
    print(f"  Generating reasoning for {remaining} remaining samples (index {start_index} → {num_samples - 1})...\n")

    t_start = time.time()

    for count, i in enumerate(range(start_index, num_samples)):

        sample = split_ds[i]
        image = sample["image"]
        caption = sample["caption"]

        t_sample_start = time.time()

        print(f"[{split_name}][{i+1}/{num_samples}] Caption: {caption[:120]}")

        reasoning = generate_reasoning(image, caption)

        t_sample_end = time.time()
        sample_time = t_sample_end - t_sample_start

        # ── Append to CSV immediately (resume-safe) ──
        row = {
            "image_id": sample.get("image_id", str(i)),
            "caption": caption,
            "reasoning": reasoning
        }
        row_df = pd.DataFrame([row])
        write_header = not output_csv.exists() or os.path.getsize(output_csv) == 0
        row_df.to_csv(output_csv, mode="a", header=write_header, index=False, encoding="utf-8")

        # ── ETA ──
        done = count + 1
        elapsed = t_sample_end - t_start
        avg_per_sample = elapsed / done
        left = remaining - done
        eta = avg_per_sample * left

        print(f"  reasoning={reasoning[:200]}...")
        print(f"  [{i+1}/{num_samples}] sample={sample_time:.1f}s | elapsed={fmt_time(elapsed)} | avg={avg_per_sample:.1f}s/sample | ETA={fmt_time(eta)} ({left} left)\n")

    total_time = time.time() - t_start
    print(f"[{split_name}] Finished! Processed {remaining} samples in {fmt_time(total_time)}")
    print(f"[{split_name}] Saved to: {output_csv}\n")

# =====================================================
# RUN
# =====================================================

for split_name, output_csv in SPLITS_CONFIG.items():
    print(f"\n{'='*60}")
    print(f"  Processing split: {split_name} ({len(dataset[split_name])} samples)")
    print(f"{'='*60}")
    process_split(split_name, dataset[split_name], output_csv)

print("\nAll splits done!")