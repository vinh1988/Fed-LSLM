# Generative RAG Medical VQA

## Goal

Improve open-ended Medical VQA by combining:

* Vision understanding
* Medical knowledge retrieval
* Generative answering

without using a fixed answer vocabulary.

---

## Motivation

Classification-based VQA suffers from OOV answers:

Train:

* yes
* no
* melanoma

Validation:

* glioma

Result:

glioma → <unk>

Generative models can directly generate unseen answers.

---

## Architecture

Image + Question
|
V
BiomedCLIP
|
V
Image-Question Embedding
|
V
FAISS
|
V
Retrieved Medical Knowledge
|
V
Qwen2.5-0.5B
|
V
Answer

---

## Models

### Vision Encoder

* BiomedCLIP (Frozen)

### Retriever

* FAISS

### Generator

* Qwen2.5-0.5B + LoRA

---

## Training

Input:

Question:
{question}

Retrieved Knowledge:
{top-k documents}

Answer:
{ground_truth}

Loss:

Language Modeling Loss

---

## Evaluation

* Exact Match (EM)
* F1 Score
* BLEU
* ROUGE-L

---

## Datasets

* PathVQA
* VQA-RAD

---

## Hardware

GPU:

* 7GB VRAM

Configuration:

* BiomedCLIP (Frozen)
* FAISS on CPU
* Qwen2.5-0.5B + LoRA

---

## Expected Benefits

* No <unk> issue
* Handles rare medical terms
* Supports open-ended answers
* Lightweight enough for a single GPU
