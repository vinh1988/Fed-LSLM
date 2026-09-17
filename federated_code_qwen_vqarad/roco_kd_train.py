# =====================================================
# 0. HF CACHE (MUST BE FIRST)
# =====================================================

import os

CACHE_DIR = r"D:\nckh_teacher_student\code\teacher_student\.hf_cache"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")

print("HF CACHE:", CACHE_DIR)


# =====================================================
# 1. IMPORTS
# =====================================================

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)

from peft import PeftModel


# =====================================================
# 2. CONFIG
# =====================================================

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

ROCO_PATH = r"D:\nckh_teacher_student\code\teacher_student\processed_data\ROCO"

TEACHER_ADAPTER = r"D:\nckh_teacher_student\code\teacher_student\experiments\VQA-RAD\noniid\alpha_0.1\k2\Qwen2.5-VL-3B-Instruct\server\global_adapter"

OUTPUT_CSV = "roco_kd_dataset.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
TEMPERATURE = 2.0
ALPHA = 0.5


# =====================================================
# 3. LOAD TEACHER (3B + GLOBAL FEDERATED ADAPTER)
# =====================================================

print("Loading teacher...")

teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

teacher = PeftModel.from_pretrained(
    teacher,
    TEACHER_ADAPTER
)

teacher.eval()

processor = AutoProcessor.from_pretrained(BASE_MODEL)


# =====================================================
# 4. LOAD ROCO
# =====================================================

def load_split(split):
    split_path = os.path.join(ROCO_PATH, split)
    csv_path = os.path.join(split_path, "captions.csv")
    img_dir = os.path.join(split_path, "images")

    df = pd.read_csv(csv_path)

    if "image" not in df.columns:
        df.columns = ["image", "caption"]

    df["image_path"] = df["image"].apply(lambda x: os.path.join(img_dir, x))
    return df


train_df = load_split("train")


# =====================================================
# 5. QUESTION TEMPLATES
# =====================================================

QUESTIONS = [
    "What abnormality is present?",
    "Describe the main finding.",
    "Which anatomical region is affected?",
    "Is there any visible pathology?"
]

PROMPT = """
You are an expert radiologist.

Analyze the image and answer the question.

Return strictly:

<rationale>
Medical reasoning based on visual evidence.
</rationale>

<answer>
Final answer.
</answer>
"""


# =====================================================
# 6. GENERATE KD DATASET
# =====================================================

rows = []

print("Generating ROCO KD dataset...")

for i, row in train_df.iterrows():

    image_path = row["image_path"]

    if not os.path.exists(image_path):
        continue

    image = Image.open(image_path).convert("RGB")

    for q in QUESTIONS:

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT + "\nQuestion: " + q}
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
        ).to(teacher.device)

        with torch.no_grad():
            output = teacher.generate(
                **inputs,
                max_new_tokens=256
            )

        response = processor.batch_decode(
            output,
            skip_special_tokens=True
        )[0]

        rows.append({
            "image": image_path,
            "question": q,
            "teacher_response": response
        })

        if len(rows) % 50 == 0:
            print("Generated:", len(rows))

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("Saved KD dataset:", OUTPUT_CSV)


# =====================================================
# 7. LOAD STUDENT (same model OR smaller VSLM)
# =====================================================

print("Loading student...")

student = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

student.train()

optimizer = torch.optim.AdamW(student.parameters(), lr=2e-5)


# =====================================================
# 8. KD LOSS (KL + CE)
# =====================================================

def kl_loss(student_logits, teacher_logits):
    T = TEMPERATURE

    s_log = F.log_softmax(student_logits / T, dim=-1)
    t_prob = F.softmax(teacher_logits / T, dim=-1)

    return F.kl_div(s_log, t_prob, reduction="batchmean") * (T * T)


# =====================================================
# 9. KD TRAINING LOOP
# =====================================================

print("Starting KD training...")

for epoch in range(1):

    for i, row in df.iterrows():

        image = Image.open(row["image"]).convert("RGB")
        question = row["question"]
        target = row["teacher_response"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": target
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False)

        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(student.device)

        # -------------------------
        # Student forward
        # -------------------------
        student_out = student(**inputs, labels=inputs["input_ids"])
        ce_loss = student_out.loss

        # -------------------------
        # Teacher forward (logits)
        # -------------------------
        with torch.no_grad():
            teacher_inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            ).to(teacher.device)

            teacher_out = teacher(**teacher_inputs)

        # -------------------------
        # Align lengths
        # -------------------------
        min_len = min(
            student_out.logits.shape[1],
            teacher_out.logits.shape[1]
        )

        kd = kl_loss(
            student_out.logits[:, :min_len],
            teacher_out.logits[:, :min_len]
        )

        loss = ce_loss + ALPHA * kd

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 20 == 0:
            print(f"Step {i} | CE: {ce_loss.item():.4f} | KD: {kd.item():.4f}")

print("Training done")