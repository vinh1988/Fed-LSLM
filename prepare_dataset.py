#!/usr/bin/env python3
"""
prepare_dataset.py

End-to-end dataset preparation for:
- PathVQA
- VQA-RAD
- SLAKE

Features:
1. Download from Hugging Face
2. Export images + CSV
3. Create centralized datasets
4. Create federated IID splits
5. Create federated Non-IID splits (Dirichlet answer-based)
6. Generate statistics

Dependencies:
    pip install datasets pandas pillow numpy tqdm

Usage:
    # Process all datasets
    python prepare_dataset.py

    # Process specific dataset(s)
    python prepare_dataset.py --datasets PathVQA VQA-RAD
    python prepare_dataset.py --datasets SLAKE
"""

from pathlib import Path
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

SEED = 42
np.random.seed(SEED)

CLIENTS = [2, 3, 4, 5]
NONIID_ALPHAS = [0.5, 0.1]
CLOSED_ANSWERS = {
    "yes",
    "no"
}


def get_question_type(answer):
    answer = str(answer).strip().lower()

    if answer in CLOSED_ANSWERS:
        return "closed"

    return "open"

DATASETS = {
    "PathVQA": "flaviagiammarino/path-vqa",
    "VQA-RAD": "flaviagiammarino/vqa-rad",
    "SLAKE": "BoKelvin/SLAKE",
}

OUTPUT_ROOT = Path("processed_data")

def save_dataset_summary(exported, output_file):
    rows = []

    for split_name, df in exported.items():
        rows.append({
            "split": split_name,
            "samples": len(df),
            "open_questions": int(
                (df["question_type"] == "open").sum()
            ),
            "closed_questions": int(
                (df["question_type"] == "closed").sum()
            )
        })

    pd.DataFrame(rows).to_csv(
        output_file,
        index=False
    )

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def export_split(split_ds, split_name, out_dir):
    images_dir = out_dir / "images" / split_name
    ensure_dir(images_dir)

    rows = []

    for idx, sample in enumerate(tqdm(split_ds, desc=f"Export {split_name}")):
        img_path = images_dir / f"{idx:07d}.jpg"

        sample["image"].save(img_path)

        rows.append({
            "image_path": str(img_path.relative_to(out_dir)),
            "question": sample["question"],
            "answer": sample["answer"],
            "question_type": get_question_type(
                sample["answer"]
            ),
            "split": split_name
        })

    df = pd.DataFrame(rows)

    csv_path = out_dir / f"{split_name}.csv"
    df.to_csv(csv_path, index=False)

    return df


def export_slake_split(split_ds, split_name, out_dir, dataset, images_source_dir):
    """Custom export for SLAKE dataset with extended metadata from local files"""
    images_dir = out_dir / "images" / split_name
    ensure_dir(images_dir)

    rows = []

    # Process all xmlab* directories
    xmlab_dirs = sorted([d for d in images_source_dir.iterdir() if d.is_dir() and d.name.startswith('xmlab')])
    
    img_idx = 0
    for xmlab_dir in tqdm(xmlab_dirs, desc=f"Processing SLAKE {split_name}"):
        question_file = xmlab_dir / "question.json"
        detection_file = xmlab_dir / "detection.json"
        
        if not question_file.exists():
            continue
        
        # Load question data
        with open(question_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        # Load detection data
        detection_data = []
        if detection_file.exists():
            try:
                with open(detection_file, 'r', encoding='utf-8') as f:
                    detection_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Process each question
        for item in questions_data:
            # Filter for English only
            if item.get("q_lang", "en") != "en":
                continue
            
            img_name = item["img_name"]  # e.g., "xmlab0/source.jpg"
            source_img_path = images_source_dir / img_name
            
            # Copy image to target directory with sequential naming
            img_path = images_dir / f"{img_idx:07d}.jpg"
            
            if source_img_path.exists():
                shutil.copy(source_img_path, img_path)
                relative_path = str(img_path.relative_to(out_dir))
            else:
                print(f"Warning: Image not found: {source_img_path}")
                relative_path = ""
            
            # Get object information from detection data
            num_objects = len(detection_data) if detection_data else 0
            objects_str = json.dumps(detection_data) if detection_data else ""
            
            # Map SLAKE fields to VQA-RAD structure
            question_type = item.get("answer_type", "OPEN").lower()
            rows.append({
                "image_path": relative_path,
                "question": item["question"],
                "answer": item["answer"],
                "q_language": item.get("q_lang", "en"),
                "question_type": question_type,
                "body_part": item.get("location", ""),  # location -> body_part
                "modality": item.get("modality", ""),
                "num_objects": num_objects,
                "objects": objects_str,
                "split": split_name
            })
            
            img_idx += 1
    
    df = pd.DataFrame(rows)

    csv_path = out_dir / f"{split_name}.csv"
    df.to_csv(csv_path, index=False)

    return df


def create_stats(client_files, output_file):
    rows = []

    for file in client_files:

        df = pd.read_csv(file)

        open_count = (
            df["question_type"] == "open"
        ).sum()

        closed_count = (
            df["question_type"] == "closed"
        ).sum()

        rows.append({
            "client": file.stem,
            "samples": len(df),
            "open_questions": int(open_count),
            "closed_questions": int(closed_count)
        })

    pd.DataFrame(rows).to_csv(
        output_file,
        index=False
    )


def iid_partition(df, k):
    shuffled = (
        df.sample(frac=1, random_state=SEED)
          .reset_index(drop=True)
    )

    indices = np.array_split(np.arange(len(shuffled)), k)

    splits = [
        shuffled.iloc[idx].reset_index(drop=True)
        for idx in indices
    ]

    return splits

def dirichlet_partition(df, k, alpha):
    labels = (
        df["answer"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    unique_labels = labels.unique()

    client_indices = [[] for _ in range(k)]

    for label in unique_labels:
        idx = np.where(labels == label)[0]

        np.random.shuffle(idx)

        proportions = np.random.dirichlet([alpha] * k)

        split_points = (
            np.cumsum(proportions) * len(idx)
        ).astype(int)[:-1]

        chunks = np.split(idx, split_points)

        for client_id, chunk in enumerate(chunks):
            client_indices[client_id].extend(chunk.tolist())

    result = []

    for client_id in range(k):
        result.append(
            df.iloc[client_indices[client_id]]
            .sample(frac=1, random_state=SEED)
            .reset_index(drop=True)
        )

    return result


def save_client_splits(splits, out_dir):
    ensure_dir(out_dir)

    files = []

    for i, part in enumerate(splits, start=1):
        file = out_dir / f"client_{i}.csv"
        part.to_csv(file, index=False)
        files.append(file)

    create_stats(files, out_dir / "stats.csv")


def process_dataset(dataset_name, hf_name):
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name}")
    print(f"{'='*80}\n")

    base_dir = OUTPUT_ROOT / dataset_name

    raw_dir = base_dir / "raw"
    central_dir = base_dir / "centralized"
    iid_dir = base_dir / "federated_iid"
    noniid_dir = base_dir / "federated_noniid"

    ensure_dir(raw_dir)

    exported = {}

    # Use custom export for SLAKE from local files, standard for others from HF
    if dataset_name == "SLAKE":
        # SLAKE images are in processed_data/Slake/images/imgs
        images_source_dir = base_dir / "images" / "imgs"
        
        # Process all SLAKE data once, then split into train/val/test
        all_data = export_slake_split(
            None,  # No HF dataset needed
            "all",  # Process all data
            raw_dir,
            None,  # No HF dataset needed
            images_source_dir
        )
        
        # Split into train/validation/test (70/15/15)
        train_df = all_data.sample(frac=0.7, random_state=SEED)
        remaining = all_data.drop(train_df.index)
        val_df = remaining.sample(frac=0.5, random_state=SEED)  # 15% of total
        test_df = remaining.drop(val_df.index)  # 15% of total
        
        # Move images to correct split directories and update paths
        all_images_dir = raw_dir / "images" / "all"
        
        for split_name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            split_images_dir = raw_dir / "images" / split_name
            ensure_dir(split_images_dir)
            
            # Update image paths and copy images
            new_image_paths = []
            for idx, row in df.iterrows():
                old_path = row["image_path"]  # e.g., "images/all/0000001.jpg"
                old_filename = Path(old_path).name  # e.g., "0000001.jpg"
                new_path = f"images/{split_name}/{old_filename}"
                new_image_paths.append(new_path)
                
                # Copy image from all to split directory
                src = raw_dir / old_path
                dst = split_images_dir / old_filename
                if src.exists() and not dst.exists():
                    shutil.copy(src, dst)
            
            df["image_path"] = new_image_paths
            df["split"] = split_name  # Update split column to match actual split
            
            # Save split CSV
            csv_path = raw_dir / f"{split_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved {split_name} split: {len(df)} samples")
        
        exported["train"] = train_df
        exported["validation"] = val_df
        exported["test"] = test_df
        
        # Clean up images/all directory
        if all_images_dir.exists():
            shutil.rmtree(all_images_dir)
    else:
        dataset = load_dataset(hf_name)
        for split_name in dataset.keys():
            exported[split_name] = export_split(
                dataset[split_name],
                split_name,
                raw_dir
            )
    
    save_dataset_summary(
        exported,
        base_dir / "dataset_summary.csv"
    )

    ensure_dir(central_dir)

    for split_name, df in exported.items():
        df.to_csv(
            central_dir / f"{split_name}.csv",
            index=False
        )

    if (raw_dir / "images").exists():
        if not (central_dir / "images").exists():
            shutil.copytree(
                raw_dir / "images",
                central_dir / "images"
            )

    train_df = exported["train"]

    for k in CLIENTS:
        splits = iid_partition(train_df, k)

        save_client_splits(
            splits,
            iid_dir / f"k{k}"
        )

    for alpha in NONIID_ALPHAS:
        for k in CLIENTS:
            splits = dirichlet_partition(
                train_df,
                k,
                alpha
            )

            save_client_splits(
                splits,
                noniid_dir / f"alpha_{alpha}" / f"k{k}"
            )

    config = {
        "dataset": dataset_name,
        "seed": SEED,
        "clients": CLIENTS,
        "alphas": NONIID_ALPHAS,
        "partition_method": "answer_dirichlet",
        "question_types": [
            "open",
            "closed"
        ]
    }

    with open(base_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Finished {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for federated learning")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        help="Specific datasets to process (default: all)"
    )
    args = parser.parse_args()

    ensure_dir(OUTPUT_ROOT)

    datasets_to_process = args.datasets if args.datasets else list(DATASETS.keys())

    for dataset_name in datasets_to_process:
        hf_name = DATASETS[dataset_name]
        process_dataset(dataset_name, hf_name)

    print("\nAll datasets prepared successfully.")


if __name__ == "__main__":
    main()
