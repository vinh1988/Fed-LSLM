# Fed-LSLM: Federated Large-Small Vision-Language Model Collaboration for Medical Visual Question Answering
## How to Run
uv run prepare_dataset.py

## Dataset Structure

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
│   ├── dataset_summary.csv
│   ├── config.json
│   │
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
│   │   │   ├── client_1.csv
│   │   │   ├── client_2.csv
│   │   │   └── stats.csv
│   │   │
│   │   ├── k3/
│   │   │   ├── client_1.csv
│   │   │   ├── client_2.csv
│   │   │   ├── client_3.csv
│   │   │   └── stats.csv
│   │   │
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
│   └── visualizations/
│
└── VQA-RAD/
    ├── dataset_summary.csv
    ├── config.json
    │
    ├── raw/
    │   ├── train.csv
    │   ├── test.csv
    │   └── images/
    │
    ├── centralized/
    │
    ├── federated_iid/
    │
    ├── federated_noniid/
    │
    └── visualizations/
```

---

## CSV Schema

All generated CSV files use the following schema:

```csv
image_path,question,answer,question_type,split
```

Example:

```csv
image_path,question,answer,question_type,split
images/train/0000001.jpg,Are regions of the brain infarcted?,yes,closed,train
images/train/0000002.jpg,Where are liver stem cells located?,in the canals of hering,open,train
```

---

## Question Types

Questions are automatically categorized into:

| Type   | Description               |
| ------ | ------------------------- |
| closed | Binary answers (yes/no)   |
| open   | Free-text medical answers |

Rule:

```python
if answer.lower() in {"yes", "no"}:
    question_type = "closed"
else:
    question_type = "open"
```

---

## Centralized Learning

The centralized dataset uses the complete training split.

```text
centralized/
├── train.csv
├── validation.csv
└── test.csv
```

This serves as the upper-bound baseline.

---

## Federated IID Setting

Training samples are randomly and uniformly distributed among clients.

Example:

```text
federated_iid/
└── k5/
    ├── client_1.csv
    ├── client_2.csv
    ├── client_3.csv
    ├── client_4.csv
    ├── client_5.csv
    └── stats.csv
```

---

## Federated Non-IID Setting

Answer-based Dirichlet partitioning is used to simulate heterogeneous medical institutions.

Two heterogeneity levels are provided:

| Alpha | Heterogeneity    |
| ----- | ---------------- |
| 0.5   | Moderate Non-IID |
| 0.1   | Severe Non-IID   |

Example:

```text
federated_noniid/
└── alpha_0.1/
    └── k5/
        ├── client_1.csv
        ├── client_2.csv
        ├── client_3.csv
        ├── client_4.csv
        ├── client_5.csv
        └── stats.csv
```

Smaller alpha values produce stronger client heterogeneity.

---

## Client Statistics

Each federated configuration contains:

```text
stats.csv
```

Format:

```csv
client,samples,open_questions,closed_questions
client_1,3921,3012,909
client_2,3855,2951,904
client_3,4012,3097,915
```

---

## Dataset Summary

Each dataset contains:

```text
dataset_summary.csv
```

Example:

```csv
split,samples,open_questions,closed_questions
train,19654,15410,4244
validation,6259,4910,1349
test,6719,5305,1414
```

---

## Configuration

Each dataset contains:

```text
config.json
```

Example:

```json
{
  "dataset": "PathVQA",
  "seed": 42,
  "clients": [2, 3, 4, 5],
  "alphas": [0.5, 0.1],
  "partition_method": "answer_dirichlet",
  "question_types": [
    "open",
    "closed"
  ]
}
```

---

## Visualization Outputs

Generated figures are stored under:

```text
visualizations/
```

Examples:

```text
PathVQA_split_sizes.png
PathVQA_client_distribution.png
PathVQA_heatmap_alpha_0.5_k5.png
PathVQA_heatmap_alpha_0.1_k5.png

VQA-RAD_split_sizes.png
VQA-RAD_client_distribution.png
VQA-RAD_heatmap_alpha_0.5_k5.png
VQA-RAD_heatmap_alpha_0.1_k5.png
```

---

## Intended Research Usage

This dataset organization supports:

* Centralized Training
* Federated Averaging (FedAvg)
* Teacher–Student Federated Learning
* Open vs Closed Question Analysis
* IID vs Non-IID Evaluation
* Medical VQA Benchmarking

The Non-IID partitions are generated using answer-based Dirichlet sampling, which is suitable for studying heterogeneous medical institutions in federated learning environments.
