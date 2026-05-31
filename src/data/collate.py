"""Collate functions for PathVQA DataLoader batches."""

from __future__ import annotations

from typing import Any

import torch


def pathvqa_collate_fn(
    batch: list[dict],
    tokenizer: Any | None = None,
    max_length: int = 128,
) -> dict[str, Any]:
    """
    Collate PathVQA samples into a batch dict for training or evaluation.

    Args:
        batch: List of dicts from PathVQADataset.__getitem__.
        tokenizer: Optional Hugging Face tokenizer for questions. If None,
            input_ids and attention_mask are omitted.
        max_length: Max token length when tokenizing questions.

    Returns:
        Dict with stacked images, text fields, question-type masks, and optional
        tokenized questions.
    """
    row_ids = [item["row_id"] for item in batch]
    images = torch.stack([item["image"] for item in batch], dim=0)
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    question_types = [item["question_type"] for item in batch]

    yes_no_mask = torch.tensor(
        [qt == "closed" for qt in question_types],
        dtype=torch.bool,
    )
    open_ended_mask = torch.tensor(
        [qt == "open" for qt in question_types],
        dtype=torch.bool,
    )

    output: dict[str, Any] = {
        "row_ids": row_ids,
        "images": images,
        "questions": questions,
        "answers": answers,
        "question_types": question_types,
        "yes_no_mask": yes_no_mask,
        "open_ended_mask": open_ended_mask,
    }

    if tokenizer is not None:
        encoded = tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        output["input_ids"] = encoded["input_ids"]
        output["attention_mask"] = encoded["attention_mask"]

    return output
