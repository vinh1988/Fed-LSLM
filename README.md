# Federated Large-Small Language Models with Multimodal Retrieval-Augmented Generation for Medical Visual Question Answering

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

## Experiment Results — PathVQA Centralized Baselines

### Comparison Table

| Method | Closed Acc | Open EM | Open BLEU-1 | Open ROUGE-L | Notes |
|---|---|---|---|---|---|
| Classifier (vocab3000, 15ep) | 0.8415 | 0.3065 | 0.1337 | 0.3208 | Fixed-vocab, Swin+BERT frozen |
| Classifier + no-UNK decoding | 0.8415 | 0.3101 | 0.1591 | 0.3288 | Same checkpoint, skip `<unk>` |
| **mRAG BiomedCLIP+FAISS+Qwen2.5-0.5B** | 0.7072 | 0.2594 | 0.1838 | 0.2899 | Zero-shot, top-5 retrieval |
| **mRAG retrieval-only (top-1)** | TBD | TBD | TBD | TBD | Baseline for RAG quality |
| **V-SLM mRAG Qwen2.5-VL-3B LoRA (text-only)** | TBD | TBD | TBD | TBD | SFT, top-5 retrieved text |
| **V-SLM mRAG Qwen2.5-VL-3B LoRA (text+image)** | TBD | TBD | TBD | TBD | SFT, top-5 retrieved text+image |

### mRAG Pipeline Commands

#### Step 1 — Build FAISS Index
```bash
# Local (CPU, smoke test with 200 samples)
python scripts/build_mrag_index.py \
    --train-csv processed_data/PathVQA/raw/train.csv \
    --data-root processed_data/PathVQA/raw \
    --output-dir outputs/mrag/index \
    --encoder biomedclip \
    --batch-size 8 \
    --device cpu \
    --max-samples 200

# Kaggle (full, GPU)
python scripts/build_mrag_index.py \
    --train-csv /kaggle/input/datasets/lekimquang/pathvqa/raw/train.csv \
    --data-root /kaggle/input/datasets/lekimquang/pathvqa/raw \
    --output-dir outputs/mrag/index \
    --encoder biomedclip \
    --batch-size 32 \
    --device cuda
```

#### Step 2 — Smoke Test (20 samples)
```bash
python scripts/evaluate_mrag.py \
    --eval-csv processed_data/PathVQA/raw/validation.csv \
    --data-root processed_data/PathVQA/raw \
    --index-dir outputs/mrag/index \
    --output-dir outputs/mrag/debug_val20 \
    --top-k 5 \
    --max-samples 20 \
    --mode generation \
    --device cpu
```

#### Step 3 — Retrieval-Only Baseline
```bash
python scripts/evaluate_mrag.py \
    --eval-csv processed_data/PathVQA/raw/test.csv \
    --data-root processed_data/PathVQA/raw \
    --index-dir outputs/mrag/index \
    --output-dir outputs/mrag/retrieval_only_top1/test \
    --top-k 5 \
    --mode retrieval_only \
    --filter-by-question-type \
    --device cuda
```

#### Step 4 — Full Generation Evaluation
```bash
python scripts/evaluate_mrag.py \
    --eval-csv processed_data/PathVQA/raw/test.csv \
    --data-root processed_data/PathVQA/raw \
    --index-dir outputs/mrag/index \
    --output-dir outputs/mrag/centralized_qwen05b_top5/test \
    --top-k 5 \
    --mode generation \
    --generator qwen2.5-0.5b \
    --filter-by-question-type \
    --device cuda
```

---

### V-SLM mRAG Pipeline Commands (Qwen2.5-VL-3B + LoRA)

#### Dependencies (Kaggle)
```bash
pip install peft bitsandbytes accelerate qwen-vl-utils open_clip_torch faiss-cpu
```

#### Step 1 — Build SFT Dataset
```bash
# Smoke test (100 samples)
python scripts/build_vslm_mrag_sft_dataset.py \
    --csv /kaggle/input/datasets/lekimquang/pathvqa/raw/train.csv \
    --data-root /kaggle/input/datasets/lekimquang/pathvqa/raw \
    --index-dir outputs/mrag/index \
    --output-jsonl outputs/vslm_mrag/sft/debug_train_text_only_top5_100.jsonl \
    --top-k 5 --case text_only \
    --exclude-self --exclude-same-image \
    --max-samples 100 --device cuda
```

#### Step 2 — Validate Dataset
```bash
python scripts/check_vslm_sft_dataset.py \
    --jsonl outputs/vslm_mrag/sft/debug_train_text_only_top5_100.jsonl \
    --num-samples 3
```

#### Step 3 — Tiny LoRA Training (smoke test)
```bash
python scripts/train_qwen_vl_lora.py \
    --model-name Qwen/Qwen2.5-VL-3B-Instruct \
    --train-jsonl outputs/vslm_mrag/sft/debug_train_text_only_top5_100.jsonl \
    --output-dir outputs/vslm_mrag/debug_qwen25vl3b_lora_100 \
    --epochs 1 --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --use-4bit --device cuda
```

#### Step 4 — Tiny Eval (smoke test)
```bash
python scripts/evaluate_vslm_mrag.py \
    --model-name Qwen/Qwen2.5-VL-3B-Instruct \
    --adapter-path outputs/vslm_mrag/debug_qwen25vl3b_lora_100 \
    --eval-csv /kaggle/input/datasets/lekimquang/pathvqa/raw/validation.csv \
    --data-root /kaggle/input/datasets/lekimquang/pathvqa/raw \
    --index-dir outputs/mrag/index \
    --output-dir outputs/vslm_mrag/debug_eval_val20 \
    --top-k 5 --case text_only \
    --max-samples 20 --device cuda --use-4bit
```

#### Full Training (after smoke passes)
```bash
# Build full SFT dataset
python scripts/build_vslm_mrag_sft_dataset.py \
    --csv /kaggle/input/datasets/lekimquang/pathvqa/raw/train.csv \
    --data-root /kaggle/input/datasets/lekimquang/pathvqa/raw \
    --index-dir outputs/mrag/index \
    --output-jsonl outputs/vslm_mrag/sft/train_text_only_top5.jsonl \
    --top-k 5 --case text_only \
    --exclude-self --exclude-same-image --device cuda

# Train full LoRA
python scripts/train_qwen_vl_lora.py \
    --model-name Qwen/Qwen2.5-VL-3B-Instruct \
    --train-jsonl outputs/vslm_mrag/sft/train_text_only_top5.jsonl \
    --output-dir outputs/vslm_mrag/qwen25vl3b_lora_text_only_top5 \
    --epochs 1 --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --use-4bit --device cuda

# Evaluate validation
python scripts/evaluate_vslm_mrag.py \
    --model-name Qwen/Qwen2.5-VL-3B-Instruct \
    --adapter-path outputs/vslm_mrag/qwen25vl3b_lora_text_only_top5 \
    --eval-csv /kaggle/input/datasets/lekimquang/pathvqa/raw/validation.csv \
    --data-root /kaggle/input/datasets/lekimquang/pathvqa/raw \
    --index-dir outputs/mrag/index \
    --output-dir outputs/vslm_mrag/qwen25vl3b_lora_text_only_top5_val \
    --top-k 5 --case text_only --device cuda --use-4bit
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
