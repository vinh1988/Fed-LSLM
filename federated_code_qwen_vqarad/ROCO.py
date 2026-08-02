import os
from pathlib import Path

# =====================================================
# CACHE LOCATION
# =====================================================

CACHE_DIR = r"D:\nckh_teacher_student\code\teacher_student\.hf_cache"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")

# =====================================================
# IMPORT AFTER SETTING CACHE
# =====================================================

from datasets import load_dataset
import pandas as pd

# =====================================================
# OUTPUT LOCATION
# =====================================================

OUT_DIR = Path(
    r"D:\nckh_teacher_student\code\teacher_student\processed_data\ROCO"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# DOWNLOAD DATASET
# =====================================================

dataset = load_dataset(
    "eltorio/ROCOv2-radiology",
    cache_dir=CACHE_DIR
)

print(dataset)

# =====================================================
# SAVE IMAGES + CSV
# =====================================================

for split in dataset.keys():

    print(f"\nProcessing {split}")

    image_dir = OUT_DIR / split / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    total = len(dataset[split])

    for idx, sample in enumerate(dataset[split]):

        image = sample["image"]
        caption = sample["caption"]

        img_name = f"{idx:06d}.jpg"

        image_path = image_dir / img_name

        image.save(image_path)

        rows.append({
            "image_name": img_name,
            "caption": caption
        })

        if idx % 1000 == 0:
            print(f"{idx}/{total}")

    df = pd.DataFrame(rows)

    csv_path = OUT_DIR / split / "captions.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved: {csv_path}")

print("\nFinished.")