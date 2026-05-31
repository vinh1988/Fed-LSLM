#!/usr/bin/env python3
"""
visualize_fl_dataset.py

Creates publication-quality visualizations for:
- PathVQA
- VQA-RAD

Outputs:
visualizations/
├── PathVQA_split_sizes.png
├── PathVQA_client_distribution.png
├── PathVQA_heatmap_alpha_0.5_k5.png
├── PathVQA_heatmap_alpha_0.1_k5.png
├── VQA-RAD_split_sizes.png
├── VQA-RAD_client_distribution.png
├── VQA-RAD_heatmap_alpha_0.5_k5.png
└── VQA-RAD_heatmap_alpha_0.1_k5.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("processed_data")
OUT = Path("visualizations")
OUT.mkdir(exist_ok=True)


def dataset_sizes(dataset_name):
    base = ROOT / dataset_name / "raw"

    rows = []
    for split in ["train", "validation", "test"]:
        f = base / f"{split}.csv"
        if f.exists():
            rows.append([split, len(pd.read_csv(f))])

    df = pd.DataFrame(rows, columns=["split", "samples"])

    plt.figure(figsize=(6, 4))
    plt.bar(df["split"], df["samples"])
    plt.title(f"{dataset_name} Split Sizes")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(OUT / f"{dataset_name}_split_sizes.png", dpi=300)
    plt.close()


def client_distribution(dataset_name):
    base = ROOT / dataset_name

    rows = []

    iid_root = base / "federated_iid"

    if iid_root.exists():
        for kdir in sorted(iid_root.glob("k*")):
            stats = kdir / "stats.csv"
            if stats.exists():
                sdf = pd.read_csv(stats)
                for _, r in sdf.iterrows():
                    rows.append([
                        f"IID-{kdir.name}",
                        r["client"],
                        r["samples"]
                    ])

    df = pd.DataFrame(rows, columns=["setting", "client", "samples"])

    if len(df) == 0:
        return

    pivot = df.pivot(
        index="setting",
        columns="client",
        values="samples"
    ).fillna(0)

    pivot.plot(kind="bar", stacked=True, figsize=(10, 5))

    plt.title(f"{dataset_name} Client Distribution (IID)")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(OUT / f"{dataset_name}_client_distribution.png", dpi=300)
    plt.close()


def answer_heatmap(dataset_name, alpha="0.1", k="k5"):
    root = (
        ROOT / dataset_name /
        "federated_noniid" /
        f"alpha_{alpha}" /
        k
    )

    if not root.exists():
        return

    client_frames = []
    top_answers = {}

    for csv_file in sorted(root.glob("client_*.csv")):
        df = pd.read_csv(csv_file)

        counts = df["answer"].astype(str).value_counts()

        for answer, cnt in counts.items():
            top_answers[answer] = top_answers.get(answer, 0) + cnt

    top_answers = sorted(
        top_answers.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]

    answer_list = [x[0] for x in top_answers]

    matrix = []

    for csv_file in sorted(root.glob("client_*.csv")):
        df = pd.read_csv(csv_file)

        counts = df["answer"].astype(str).value_counts()

        row = [counts.get(a, 0) for a in answer_list]

        matrix.append(row)

    matrix = np.array(matrix)

    plt.figure(figsize=(12, 6))

    plt.imshow(matrix, aspect="auto")

    plt.colorbar(label="Samples")

    plt.yticks(
        range(matrix.shape[0]),
        [f"Client {i}" for i in range(matrix.shape[0])]
    )

    plt.xticks(
        range(len(answer_list)),
        answer_list,
        rotation=90
    )

    plt.title(
        f"{dataset_name} Non-IID Heatmap "
        f"(alpha={alpha}, {k})"
    )

    plt.tight_layout()

    plt.savefig(
        OUT / f"{dataset_name}_heatmap_alpha_{alpha}_{k}.png",
        dpi=300
    )

    plt.close()


def main():
    for dataset in ["PathVQA", "VQA-RAD"]:

        if not (ROOT / dataset).exists():
            continue

        dataset_sizes(dataset)
        client_distribution(dataset)

        answer_heatmap(dataset, "0.5", "k5")
        answer_heatmap(dataset, "0.1", "k5")

    print(f"Saved figures to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
