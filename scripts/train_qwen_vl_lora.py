"""Fine-tune Qwen2.5-VL or Qwen2-VL with LoRA/QLoRA on V-SLM mRAG SFT data.

Supported model families:
  - Qwen/Qwen2.5-VL-*  (uses Qwen2_5_VLForConditionalGeneration)
  - Qwen/Qwen2-VL-*    (uses Qwen2VLForConditionalGeneration)

Uses PEFT LoRA adapters with optional 4-bit quantisation (bitsandbytes)
to fit within Kaggle T4 GPU (16GB VRAM).

Training loop is manual (not HF Trainer) for full control over logging,
gradient accumulation, and checkpoint saving.

Usage (Qwen2.5-VL):
    python scripts/train_qwen_vl_lora.py \\
        --model-name Qwen/Qwen2.5-VL-3B-Instruct \\
        --train-jsonl outputs/vslm_mrag/sft/train.jsonl \\
        --output-dir outputs/vslm_mrag/lora_qwen25vl \\
        --epochs 1 --batch-size 1 --use-4bit --device cuda

Usage (Qwen2-VL):
    python scripts/train_qwen_vl_lora.py \\
        --model-name Qwen/Qwen2-VL-2B-Instruct \\
        --train-jsonl outputs/vslm_mrag/sft/train.jsonl \\
        --output-dir outputs/vslm_mrag/lora_qwen2vl \\
        --epochs 1 --batch-size 1 --use-4bit --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_qwen_vl_lora")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL / Qwen2-VL with LoRA on V-SLM mRAG SFT data"
    )
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--train-jsonl", type=str, required=True)
    p.add_argument("--val-jsonl", type=str, default=None)
    p.add_argument("--output-dir", type=str, required=True)

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Quantisation
    p.add_argument(
        "--use-4bit",
        action="store_true",
        default=False,
        help="Load model in 4-bit (QLoRA) via bitsandbytes.",
    )

    # Runtime
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-samples", type=int, default=None,
                    help="(Deprecated, use --max-train-samples) Limit dataset size.")
    p.add_argument("--max-train-samples", type=int, default=None,
                    help="Limit number of training samples (for smoke testing).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-pixels", type=int, default=128 * 28 * 28)
    p.add_argument("--max-pixels", type=int, default=256 * 28 * 28)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------

def detect_model_family(model_name: str) -> str:
    """Detect model family from model name.

    Returns 'qwen2_5_vl', 'qwen2_vl', or 'smolvlm'.
    """
    name_lower = model_name.lower()
    if "smolvlm" in name_lower:
        return "smolvlm"
    elif "qwen2.5-vl" in name_lower or "qwen2_5_vl" in name_lower:
        return "qwen2_5_vl"
    elif "qwen2-vl" in name_lower or "qwen2_vl" in name_lower:
        return "qwen2_vl"
    else:
        # Default to Qwen2.5-VL for backward compatibility
        logger.warning(
            "Cannot detect model family from '%s', defaulting to qwen2_5_vl",
            model_name,
        )
        return "qwen2_5_vl"


def _get_model_class(family: str):
    """Import and return the correct model class for the given family."""
    if family == "smolvlm":
        try:
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq
        except ImportError:
            # Older transformers versions don't have AutoModelForVision2Seq
            from transformers import Idefics3ForConditionalGeneration
            return Idefics3ForConditionalGeneration
    elif family == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(args: argparse.Namespace):
    """Load base model (optionally quantised) and processor.

    Auto-detects model family from --model-name and uses the correct class.
    Returns (model, processor, model_family).
    """
    from transformers import AutoProcessor

    model_family = detect_model_family(args.model_name)
    model_cls = _get_model_class(model_family)
    logger.info("Detected model family: %s -> %s", model_family, model_cls.__name__)

    logger.info("Loading processor: %s", args.model_name)
    if model_family == "smolvlm":
        processor = AutoProcessor.from_pretrained(
            args.model_name,
            trust_remote_code=True,
        )
    else:
        processor = AutoProcessor.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )

    # Build model kwargs
    model_kwargs = {
        "trust_remote_code": True,
    }

    if args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
            logger.info("Using 4-bit quantisation (QLoRA)")
        except ImportError:
            logger.warning(
                "bitsandbytes not available — loading in float16 instead."
            )
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"

    logger.info("Loading model: %s (class=%s)", args.model_name, model_cls.__name__)
    model = model_cls.from_pretrained(
        args.model_name, **model_kwargs
    )

    return model, processor, model_family


def apply_lora(model, args: argparse.Namespace):
    """Apply LoRA adapters to the model.

    Returns the PEFT-wrapped model.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if args.use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    else:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Log trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA applied | trainable: %s (%.2f%%) | total: %s",
        f"{trainable_params:,}",
        100 * trainable_params / total_params,
        f"{total_params:,}",
    )

    return model


# ---------------------------------------------------------------------------
# Learning rate scheduler
# ---------------------------------------------------------------------------

def get_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float) -> float:
    """Linear warmup + cosine decay schedule."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model,
    train_loader: DataLoader,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict]:
    """Run the training loop.

    Returns list of log entries (per-step).
    """
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    logger.info(
        "Training | epochs=%d | batch_size=%d | grad_accum=%d | total_steps=%d | warmup=%d",
        args.epochs, args.batch_size, args.gradient_accumulation_steps,
        total_steps, warmup_steps,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    model.train()
    global_step = 0
    accumulated_loss = 0.0
    log_entries = []

    for epoch in range(args.epochs):
        logger.info("=== Epoch %d/%d ===", epoch + 1, args.epochs)
        epoch_loss = 0.0
        epoch_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move tensors to device
            device = next(model.parameters()).device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Check for valid label tokens (completely masked due to truncation)
            valid_labels = (labels != -100).sum().item()
            if valid_labels == 0:
                logger.warning(
                    "Batch %d has 0 valid label tokens (completely masked due to truncation). Skipping batch.",
                    batch_idx
                )
                # Ensure gradient accumulation steps remain synchronized
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    lr = get_lr(global_step, total_steps, warmup_steps, args.lr)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    accumulated_loss = 0.0
                continue

            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            if "pixel_values" in batch:
                forward_kwargs["pixel_values"] = batch["pixel_values"].to(device)
            if "pixel_attention_mask" in batch:
                forward_kwargs["pixel_attention_mask"] = batch["pixel_attention_mask"].to(device)
            if "image_grid_thw" in batch:
                forward_kwargs["image_grid_thw"] = batch["image_grid_thw"].to(device)

            # Forward pass
            outputs = model(**forward_kwargs)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            accumulated_loss += loss.item()
            epoch_loss += outputs.loss.item()
            epoch_samples += input_ids.shape[0]

            # Gradient accumulation step
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                # Update LR
                lr = get_lr(global_step, total_steps, warmup_steps, args.lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Log
                if global_step % args.log_every == 0 or global_step == 1:
                    avg_loss = accumulated_loss / args.log_every if global_step > 1 else accumulated_loss
                    log_entry = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": round(avg_loss, 4),
                        "lr": round(lr, 8),
                        "samples_seen": epoch_samples + epoch * len(train_loader.dataset),
                    }
                    log_entries.append(log_entry)
                    logger.info(
                        "step=%d | loss=%.4f | lr=%.2e | samples=%d",
                        global_step, avg_loss, lr, log_entry["samples_seen"],
                    )
                    accumulated_loss = 0.0

        avg_epoch_loss = epoch_loss / max(epoch_samples, 1)
        logger.info(
            "Epoch %d done | avg_loss=%.4f | samples=%d",
            epoch + 1, avg_epoch_loss, epoch_samples,
        )

    # Handle remaining gradients (if total batches not divisible by accum steps)
    if (batch_idx + 1) % args.gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    return log_entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve --max-train-samples (prefer over deprecated --max-samples)
    effective_max_samples = args.max_train_samples or args.max_samples

    # 1. Load model and processor
    model, processor, model_family = load_model_and_processor(args)

    # Log configuration summary
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    logger.info("=" * 60)
    logger.info("  Configuration Summary")
    logger.info("=" * 60)
    logger.info("  model_name     : %s", args.model_name)
    logger.info("  model_family   : %s", model_family)
    logger.info("  device         : %s", device)
    logger.info("  dtype          : %s", dtype)
    logger.info("  use_4bit       : %s", args.use_4bit)
    logger.info("  max_length     : %d", args.max_length)
    logger.info("  max_train_samples : %s", effective_max_samples or "all")
    logger.info("=" * 60)

    # 2. Apply LoRA
    model = apply_lora(model, args)

    # 3. Load dataset
    from src.vslm.qwen_vl_dataset import VLMSFTDataset, vlm_collate_fn

    train_dataset = VLMSFTDataset(
        jsonl_path=args.train_jsonl,
        processor=processor,
        max_length=args.max_length,
        max_samples=effective_max_samples,
        model_family=model_family,  # Pass explicitly to avoid auto-detection failure
    )

    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    from functools import partial
    collate_fn = partial(vlm_collate_fn, pad_token_id=pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues with PIL
        pin_memory=True,
    )

    logger.info("Dataset size: %d | Batches: %d", len(train_dataset), len(train_loader))

    # 4. Train
    t0 = time.time()
    log_entries = train(model, train_loader, args, output_dir)
    elapsed = time.time() - t0

    # 5. Save LoRA adapter
    logger.info("Saving LoRA adapter to: %s", output_dir)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # 6. Save training config
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    avg_time_per_step = elapsed / max(total_steps, 1)

    config = {
        "model_name": args.model_name,
        "model_family": model_family,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr": args.lr,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "use_4bit": args.use_4bit,
        "max_train_samples": effective_max_samples,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "dataset_size": len(train_dataset),
        "total_batches": len(train_loader),
        "total_optimizer_steps": total_steps,
        "elapsed_seconds": round(elapsed, 1),
        "avg_seconds_per_step": round(avg_time_per_step, 2),
    }
    config_path = output_dir / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Training config saved: %s", config_path)

    # 7. Save training log
    log_path = output_dir / "training_log.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")
    logger.info("Training log saved: %s", log_path)

    # 8. Summary
    print("\n" + "=" * 65)
    print("  V-SLM LoRA Training Complete")
    print("=" * 65)
    print(f"  Model       : {args.model_name}")
    print(f"  Family      : {model_family}")
    print(f"  LoRA r/α    : {args.lora_r} / {args.lora_alpha}")
    print(f"  4-bit       : {args.use_4bit}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Dataset     : {len(train_dataset)} samples")
    print(f"  Total steps : {total_steps}")
    print(f"  Final loss  : {log_entries[-1]['loss']:.4f}" if log_entries else "  Final loss  : N/A")
    print(f"  Elapsed     : {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Avg/step    : {avg_time_per_step:.1f}s")
    print(f"  Adapter     : {output_dir}")
    print("=" * 65 + "\n")

    logger.info("Done! LoRA adapter saved to: %s", output_dir)


if __name__ == "__main__":
    main()
