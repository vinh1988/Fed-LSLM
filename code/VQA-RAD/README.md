# Generative RAG Medical VQA on VQA-RAD (Centralized Split)

This directory contains the code to train a Generative Retrieval-Augmented Generation (RAG) Medical Visual Question Answering (VQA) model on the centralized VQA-RAD dataset.

## Architecture

The system operates as follows:
1. **Multimodal Encoding**: Input medical images and questions are embedded into a shared space using a frozen **BiomedCLIP** model.
2. **FAISS Indexing & Retrieval**: Combined image-question vectors are stored in a FAISS index. For each query (image + question), the top-k most similar cases are retrieved from the training set. Self-leakage is prevented during training by filtering out the query case itself.
3. **Generative RAG**: The retrieved questions and answers serve as the medical knowledge context, formatted as prompt inputs.
4. **Answer Generation**: A **Qwen2.5-0.5B-Instruct** model, fine-tuned with **LoRA (PEFT)**, reads the query and context and generates the free-text answer.

## Requirements

Ensure the virtual environment is synced:
```bash
uv sync
```

## How to Run

To run the training script with default settings (Precomputing embeddings, FAISS indexing, LoRA fine-tuning, and evaluating on the test set):

```bash
teacher_student/code/VQA-RAD$ uv run  code/VQA-RAD/train.py --epochs 10  --batch_size 8
```

### Key Parameters

You can customize the execution using command-line arguments:
* `--data_dir`: Path to the centralized dataset folder (defaults to `processed_data/VQA-RAD/centralized`).
* `--model_name`: HF hub path of Qwen (defaults to `Qwen/Qwen2.5-0.5B-Instruct`).
* `--output_dir`: Output directory for checkpoints and predictions (defaults to `code/VQA-RAD/results`).
* `--k`: Number of retrieved documents to include in the context (defaults to `3`).
* `--epochs`: Number of fine-tuning epochs (defaults to `5`).
* `--batch_size`: Batch size for training (defaults to `4`).
* `--gradient_accumulation_steps`: Gradient accumulation steps (defaults to `2`).
* `--lr`: Learning rate for LoRA (defaults to `2e-4`).

For a full list of options, run:
```bash
.venv/bin/python code/VQA-RAD/train.py --help
```

## Memory Optimization

The code is optimized for hardware configurations with tight VRAM budgets (e.g., 7GB VRAM):
1. **Model Decoupling**: BiomedCLIP is loaded first to extract and cache the image-question embeddings. Once precomputation is done, the model is deleted and the GPU memory is completely cleared before Qwen is loaded.
2. **LoRA Fine-Tuning**: Only low-rank adapter weights are trained, keeping base Qwen2.5 parameters frozen.
3. **bfloat16/float16 Precision**: Automatic mixed precision matches the hardware support.
4. **Batch Size & Accumulation**: Allows setting small batch sizes with gradient accumulation.

## Validation

```bash
uv run metric.py
```
## ================ SUMMARY ================ 
| Setting           | Closed Acc | Open BLEU | Open ROUGE-L | Open BERTScore |
| ----------------- | ---------- | --------- | ------------ | -------------- |
| Without Retrieval | 0.7251     | 0.0733    | 0.3503       | 0.9032         |
| With Retrieval    | 0.7131     | 0.0709    | 0.3659       | 0.9025         |


[RAG-VQA] Epoch 10 Eval Metrics: EM=0.5011, F1=0.5394, BLEU=0.1012, ROUGE-L=0.5490