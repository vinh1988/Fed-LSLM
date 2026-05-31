# teacher_student

# Dataset Structure

This repository prepares two medical Visual Question Answering (VQA) datasets:

* PathVQA
* VQA-RAD

The datasets are converted into a unified format and partitioned for centralized and federated learning experiments.

---

## Directory Layout

```text
processed_data/
│
├── PathVQA/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   ├── test.csv
│   │   └── images/
│   │       ├── train/
│   │       ├── validation/
│   │       └── test/
│   │
│   ├── centralized/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   ├── test.csv
│   │   └── images/
│   │
│   ├── federated_iid/
│   │   ├── k2/
│   │   │   ├── client_0.csv
│   │   │   ├── client_1.csv
│   │   │   └── stats.csv
│   │   │
│   │   ├── k3/
│   │   ├── k4/
│   │   └── k5/
│   │
│   ├── federated_noniid/
│   │   ├── alpha_0.5/
│   │   │   ├── k2/
│   │   │   ├── k3/
│   │   │   ├── k4/
│   │   │   └── k5/
│   │   │
│   │   └── alpha_0.1/
│   │       ├── k2/
│   │       ├── k3/
│   │       ├── k4/
│   │       └── k5/
│   │
│   └── config.json
│
└── VQA-RAD/
    ├── raw/
    ├── centralized/
    ├── federated_iid/
    ├── federated_noniid/
    └── config.json
```

---

## CSV Format

All CSV files use the same schema.

| Column     | Description            |
| ---------- | ---------------------- |
| image_path | Relative path to image |
| question   | Medical VQA question   |
| answer     | Ground-truth answer    |

Example:

```csv
image_path,question,answer
images/train/0000001.jpg,Are regions of the brain infarcted?,yes
images/train/0000002.jpg,Where are liver stem cells located?,in the canals of hering
```

---

## Centralized Setting

The centralized setting uses the entire training dataset.

```text
centralized/
├── train.csv
├── validation.csv
└── test.csv
```

This serves as the upper-bound baseline.

---

## Federated IID Setting

The training data are randomly and uniformly distributed across clients.

Example:

```text
federated_iid/
└── k5/
    ├── client_0.csv
    ├── client_1.csv
    ├── client_2.csv
    ├── client_3.csv
    ├── client_4.csv
    └── stats.csv
```

Client datasets have approximately equal sample counts.

---

## Federated Non-IID Setting

Answer-based Dirichlet partitioning is applied.

Two heterogeneity levels are used:

| Alpha | Heterogeneity    |
| ----- | ---------------- |
| 0.5   | Moderate Non-IID |
| 0.1   | Severe Non-IID   |

Example:

```text
federated_noniid/
└── alpha_0.1/
    └── k5/
        ├── client_0.csv
        ├── client_1.csv
        ├── client_2.csv
        ├── client_3.csv
        ├── client_4.csv
        └── stats.csv
```

Smaller alpha values produce stronger client heterogeneity.

---

## Teacher–Student Federated Learning

The generated partitions are intended for:

* FedAvg
* FedProx
* Teacher–Student Federated Learning

Experimental configurations:

| Setting           | Clients        |
| ----------------- | -------------- |
| IID               | k = 2, 3, 4, 5 |
| Non-IID (α = 0.5) | k = 2, 3, 4, 5 |
| Non-IID (α = 0.1) | k = 2, 3, 4, 5 |

---

## Visualization Files

Generated figures:

```text
visualizations/
├── PathVQA_split_sizes.png
├── PathVQA_client_distribution.png
├── PathVQA_heatmap_alpha_0.5_k5.png
├── PathVQA_heatmap_alpha_0.1_k5.png
├── VQA-RAD_split_sizes.png
├── VQA-RAD_client_distribution.png
├── VQA-RAD_heatmap_alpha_0.5_k5.png
└── VQA-RAD_heatmap_alpha_0.1_k5.png
```

These visualizations characterize dataset size, client distributions, and Non-IID heterogeneity.
