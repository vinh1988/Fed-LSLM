#!/usr/bin/env python3
"""
prepare_seed_partitions.py

Generate seed-dependent non-IID (Dirichlet, answer-based) client partitions.

For each (alpha, k, seed) it writes:
    processed_data/{dataset}/federated_noniid/alpha_{alpha}/k{k}/seed_{seed}/client_{i}.csv
    processed_data/{dataset}/federated_noniid/alpha_{alpha}/k{k}/seed_{seed}/stats.csv

Seed 42 partitions already exist at .../k{k}/client_{i}.csv (legacy, no seed dir)
and are NOT regenerated here.

Usage:
    python prepare_seed_partitions.py --dataset Slake --alpha 0.5 --k 2 3 4 5 --seeds 123 2025
    python prepare_seed_partitions.py --dataset Slake --alpha 0.5 --k 2 --seeds 123
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIR_NAMES = {
    "SLAKE": "Slake",
    "VQA-RAD": "VQA-RAD",
}


def dirichlet_partition(df: pd.DataFrame, k: int, alpha: float, seed: int):
    """Same algorithm as prepare_dataset.dirichlet_partition but seeded per-run."""
    rng = np.random.default_rng(seed)

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
        rng.shuffle(idx)
        proportions = rng.dirichlet([alpha] * k)
        split_points = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        chunks = np.split(idx, split_points)

        for client_id, chunk in enumerate(chunks):
            client_indices[client_id].extend(chunk.tolist())

    result = []
    for client_id in range(k):
        part = df.iloc[client_indices[client_id]]
        part = part.sample(frac=1, random_state=seed).reset_index(drop=True)
        result.append(part)
    return result


def create_stats(client_files, output_file):
    rows = []
    for file in client_files:
        df = pd.read_csv(file)
        rows.append({
            "client": file.stem,
            "samples": len(df),
            "open_questions": int((df["question_type"] == "open").sum()),
            "closed_questions": int((df["question_type"] == "closed").sum()),
        })
    pd.DataFrame(rows).to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate seed-dependent non-IID partitions")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_DIR_NAMES.keys()))
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--k", type=int, nargs="+", required=True,
                        help="Client counts, e.g. --k 2 3 4 5")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="Seeds to generate, e.g. --seeds 123 2025")
    args = parser.parse_args()

    dataset_dir = DATASET_DIR_NAMES[args.dataset]
    train_csv = Path("processed_data") / dataset_dir / "centralized" / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(
            f"{train_csv} not found. Run prepare_dataset.py first."
        )

    train_df = pd.read_csv(train_csv)
    print(f"Loaded {len(train_df)} train samples from {train_csv}")

    base_dir = (
        Path("processed_data") / dataset_dir / "federated_noniid"
        / f"alpha_{args.alpha}"
    )

    for seed in args.seeds:
        if seed == 42:
            print(f"Skipping seed 42 (legacy partitions already exist without seed dir)")
            continue
        for k in args.k:
            out_dir = base_dir / f"k{k}" / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)

            splits = dirichlet_partition(train_df, k, args.alpha, seed)

            files = []
            for i, part in enumerate(splits, start=1):
                f = out_dir / f"client_{i}.csv"
                part.to_csv(f, index=False)
                files.append(f)
            create_stats(files, out_dir / "stats.csv")

            sizes = [len(s) for s in splits]
            print(f"  alpha={args.alpha} k={k} seed={seed}: clients={sizes} -> {out_dir}")


if __name__ == "__main__":
    main()
