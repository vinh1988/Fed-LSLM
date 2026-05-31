#!/usr/bin/env python3
"""
prepare_dataset.py

End-to-end dataset preparation for:
- PathVQA
- VQA-RAD

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
    python prepare_dataset.py
"""

from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

SEED = 42
np.random.seed(SEED)

CLIENTS = [2, 3, 4, 5]
NONIID_ALPHAS = [0.5, 0.1]

DATASETS = {
    "PathVQA": "flaviagiammarino/path-vqa",
    "VQA-RAD": "flaviagiammarino/vqa-rad",
}

OUTPUT_ROOT = Path("processed_data")


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
            "answer": sample["answer"]
        })

    df = pd.DataFrame(rows)

    csv_path = out_dir / f"{split_name}.csv"
    df.to_csv(csv_path, index=False)

    return df


def create_stats(client_files, output_file):
    rows = []

    for file in client_files:
        df = pd.read_csv(file)

        rows.append({
            "client": file.stem,
            "samples": len(df)
        })

    pd.DataFrame(rows).to_csv(output_file, index=False)


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
    labels = df["answer"].astype(str)

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

    for i, part in enumerate(splits):
        file = out_dir / f"client_{i}.csv"
        part.to_csv(file, index=False)
        files.append(file)

    create_stats(files, out_dir / "stats.csv")


def process_dataset(dataset_name, hf_name):
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name}")
    print(f"{'='*80}\n")

    dataset = load_dataset(hf_name)

    base_dir = OUTPUT_ROOT / dataset_name

    raw_dir = base_dir / "raw"
    central_dir = base_dir / "centralized"
    iid_dir = base_dir / "federated_iid"
    noniid_dir = base_dir / "federated_noniid"

    ensure_dir(raw_dir)

    exported = {}

    for split_name in dataset.keys():
        exported[split_name] = export_split(
            dataset[split_name],
            split_name,
            raw_dir
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
    }

    with open(base_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Finished {dataset_name}")


def main():
    ensure_dir(OUTPUT_ROOT)

    for dataset_name, hf_name in DATASETS.items():
        process_dataset(dataset_name, hf_name)

    print("\nAll datasets prepared successfully.")


if __name__ == "__main__":
    main()
