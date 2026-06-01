#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path("processed_data")
OUT = Path("answer_statistics")

OUT.mkdir(exist_ok=True)


DATASETS = [
    "PathVQA",
    "VQA-RAD"
]


def load_dataset(dataset_name):
    dataset_dir = ROOT / dataset_name / "raw"

    dfs = []

    for split in ["train", "validation", "test"]:
        file = dataset_dir / f"{split}.csv"

        if file.exists():
            dfs.append(pd.read_csv(file))

    return pd.concat(dfs, ignore_index=True)


def analyze_answers(dataset_name):

    print(f"\nProcessing {dataset_name}")

    df = load_dataset(dataset_name)

    stats = (
        df["answer"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .reset_index()
    )

    stats.columns = ["answer", "count"]

    stats["percentage"] = (
        stats["count"] / stats["count"].sum() * 100
    ).round(4)

    stats.to_csv(
        OUT / f"{dataset_name}_answers.csv",
        index=False
    )

    stats.head(50).to_csv(
        OUT / f"{dataset_name}_top50.csv",
        index=False
    )

    print(
        f"Unique answers: {len(stats):,}"
    )

    print(
        f"Saved: {dataset_name}_answers.csv"
    )


def main():

    for dataset in DATASETS:
        analyze_answers(dataset)

    print("\nFinished.")


if __name__ == "__main__":
    main()