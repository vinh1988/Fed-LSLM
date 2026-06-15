"""PyTorch Dataset for V-SLM SFT training on mRAG JSONL data.

Supports Qwen2.5-VL, Qwen2-VL, and SmolVLM models.
Reads JSONL records produced by ``build_vslm_mrag_sft_dataset.py``,
tokenises them through the VLM processor, and builds ``labels``
that mask everything except the assistant answer (loss on answer only).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


def preprocess_messages_for_vlm(messages: list[dict], model_family: str) -> tuple[list[dict], list[Any]]:
    """Preprocess messages and load images based on model family.

    Returns:
        processed_messages: Messages formatted for the model's apply_chat_template.
        images: List of loaded PIL Image objects (or processor-compatible inputs).
    """
    images = []
    processed_messages = []

    if model_family == "smolvlm":
        # SmolVLM chat template expects list of dicts: [{"type": "text", "text": ...}, {"type": "image"}]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                processed_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}]
                })
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if item.get("type") == "image":
                        img_path = item.get("image")
                        if img_path:
                            try:
                                img = Image.open(img_path).convert("RGB")
                                images.append(img)
                            except Exception as e:
                                logger.error("Failed to load image %s: %s", img_path, e)
                        new_content.append({"type": "image"})
                    else:
                        new_content.append(item)
                processed_messages.append({"role": role, "content": new_content})
        return processed_messages, images
    else:
        # Qwen2-VL / Qwen2.5-VL
        from qwen_vl_utils import process_vision_info  # type: ignore
        image_inputs, _ = process_vision_info(messages)
        return messages, image_inputs


class VLMSFTDataset(Dataset):
    """SFT dataset for VLMs (Qwen-VL, SmolVLM) that loads JSONL records with images.

    Each JSONL record must contain a ``messages`` field in chat format
    (system → user with image+text → assistant).

    Args:
        jsonl_path: Path to the SFT JSONL file.
        processor: VLM processor instance.
        max_length: Maximum token length for truncation.
        max_samples: Optional limit on dataset size (for debugging).
        model_family: Optional model family string ('qwen2_5_vl', 'qwen2_vl', 'smolvlm').
                      If not provided, auto-detected from processor class name.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        processor: Any,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
        model_family: Optional[str] = None,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.records: list[dict] = []

        # Auto-detect model family if not provided
        if model_family is not None:
            self.model_family = model_family
        else:
            processor_class_name = processor.__class__.__name__.lower()
            if "qwen2_5_vl" in processor_class_name:
                self.model_family = "qwen2_5_vl"
            elif "qwen2vl" in processor_class_name:
                self.model_family = "qwen2_vl"
            elif "idefics3" in processor_class_name or "smolvlm" in processor_class_name:
                self.model_family = "smolvlm"
            else:
                logger.warning(
                    "Cannot auto-detect model family from processor class '%s', defaulting to qwen2_5_vl",
                    processor.__class__.__name__
                )
                self.model_family = "qwen2_5_vl"

        logger.info("VLM Dataset family: %s", self.model_family)
        logger.info("Loading SFT JSONL from: %s", jsonl_path)
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))
                if max_samples is not None and len(self.records) >= max_samples:
                    break

        logger.info("Loaded %d SFT records (max_length=%d)", len(self.records), max_length)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Process a single record into model inputs + masked labels."""
        record = self.records[idx]
        messages = record["messages"]

        # Preprocess messages and extract image inputs
        processed_messages, image_inputs = preprocess_messages_for_vlm(messages, self.model_family)

        # --- Build FULL conversation text (with assistant answer) ---
        full_text = self.processor.apply_chat_template(
            processed_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # --- Build PROMPT-ONLY text (without assistant answer) ---
        prompt_messages = [m for m in processed_messages if m["role"] != "assistant"]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # --- Tokenize full conversation ---
        # NOTE: For SmolVLM (Idefics3), passing truncation=True causes a mismatch
        # between image token count in the text template and in input_ids after truncation.
        # Solution: disable truncation in processor, then manually truncate tensors afterwards.
        if self.model_family == "smolvlm":
            full_inputs = self.processor(
                text=[full_text],
                images=image_inputs if image_inputs else None,
                return_tensors="pt",
                padding=False,
            )
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=image_inputs if image_inputs else None,
                return_tensors="pt",
                padding=False,
            )
        else:
            # --- Tokenize full conversation ---
            full_inputs = self.processor(
                text=[full_text],
                images=image_inputs if image_inputs else None,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )

            # --- Tokenize prompt only (to find where answer starts) ---
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=image_inputs if image_inputs else None,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )

        input_ids = full_inputs["input_ids"].squeeze(0)  # (seq_len,)
        attention_mask = full_inputs["attention_mask"].squeeze(0)

        # For SmolVLM: manually truncate to max_length after processing
        if self.model_family == "smolvlm" and input_ids.size(0) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        # --- Build labels: mask prompt tokens with -100 ---
        labels = input_ids.clone()
        prompt_len = min(prompt_inputs["input_ids"].shape[1], self.max_length)
        labels[:prompt_len] = -100

        # Also mask any padding tokens
        labels[attention_mask == 0] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Add vision tensors if present
        for key in ["pixel_values", "pixel_attention_mask", "image_grid_thw"]:
            if key in full_inputs:
                result[key] = full_inputs[key].squeeze(0)

        return result


def vlm_collate_fn(batch: list[dict[str, torch.Tensor]], pad_token_id: int = 0) -> dict[str, torch.Tensor]:
    """Collate function that pads text tensors and stacks/concatenates vision tensors.

    Handles variable-length sequences by padding ``input_ids`` and
    ``labels`` (pad token = pad_token_id, label pad = -100) and ``attention_mask``
    (pad = 0).
    """
    max_len = max(item["input_ids"].size(0) for item in batch)

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        padded_input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        padded_attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_labels.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

    result = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels),
    }

    # If batch size is 1, unsqueeze vision tensors to avoid collation issues
    if len(batch) == 1:
        item = batch[0]
        for key in item.keys():
            if key not in ["input_ids", "attention_mask", "labels"]:
                result[key] = item[key].unsqueeze(0)
    else:
        # Collate vision tensors for batch size > 1
        if "pixel_values" in batch[0]:
            all_pv = [item["pixel_values"] for item in batch if "pixel_values" in item]
            if all_pv:
                if "image_grid_thw" in batch[0]:
                    # Qwen2/2.5-VL style: concatenate along dim 0
                    result["pixel_values"] = torch.cat(all_pv, dim=0)
                else:
                    # SmolVLM / General: try stacking first
                    try:
                        result["pixel_values"] = torch.stack(all_pv, dim=0)
                    except RuntimeError:
                        result["pixel_values"] = torch.cat(all_pv, dim=0)

        if "image_grid_thw" in batch[0]:
            all_thw = [item["image_grid_thw"] for item in batch if "image_grid_thw" in item]
            if all_thw:
                processed_thw = []
                for thw in all_thw:
                    if thw.dim() == 1:
                        thw = thw.unsqueeze(0)
                    processed_thw.append(thw)
                result["image_grid_thw"] = torch.cat(processed_thw, dim=0)

        if "pixel_attention_mask" in batch[0]:
            all_pam = [item["pixel_attention_mask"] for item in batch if "pixel_attention_mask" in item]
            if all_pam:
                try:
                    result["pixel_attention_mask"] = torch.stack(all_pam, dim=0)
                except RuntimeError:
                    result["pixel_attention_mask"] = torch.cat(all_pam, dim=0)

    return result


# Aliases for backward compatibility
QwenVLSFTDataset = VLMSFTDataset
qwen_vl_collate_fn = vlm_collate_fn
