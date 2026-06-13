"""Evaluate a saved checkpoint on validation and test splits with optional unk fallback."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pathvqa_dataset import PathVQADataset
from src.data.collate import pathvqa_collate_fn
from src.models.baseline_model import PathVQABaseline
from src.training.trainer import CentralizedTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint with optional unk fallback")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_centralized.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint .pt file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--allow-unk-prediction",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="Allow predicting <unk> (true/false)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (mps, cuda, cpu, auto)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def filter_dataset(dataset: PathVQADataset, split_name: str) -> None:
    from src.evaluation.metrics import normalize_answer
    closed_mask = dataset.df["question_type"] == "closed"
    invalid_mask = closed_mask & (~dataset.df["answer"].astype(str).apply(normalize_answer).isin({"yes", "no"}))
    num_invalid = invalid_mask.sum()

    if num_invalid > 0:
        logger.warning(
            "Found %d closed samples with invalid answers (not 'yes' or 'no') in '%s' split. Skipping them.",
            num_invalid,
            split_name,
        )
        dataset.df = dataset.df[~invalid_mask].reset_index(drop=True)


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate state dict keys between older and newer SwinModel versions in transformers."""
    new_state_dict = {}
    
    # Mapping table for older HuggingFace Swin key format to newer version format.
    # Order matters: more specific suffixes must be checked before less specific ones.
    mapping = {
        # Attention block mappings
        ".attention.self.query.weight": ".attention.q_proj.weight",
        ".attention.self.query.bias": ".attention.q_proj.bias",
        ".attention.self.key.weight": ".attention.k_proj.weight",
        ".attention.self.key.bias": ".attention.k_proj.bias",
        ".attention.self.value.weight": ".attention.v_proj.weight",
        ".attention.self.value.bias": ".attention.v_proj.bias",
        ".attention.output.dense.weight": ".attention.o_proj.weight",
        ".attention.output.dense.bias": ".attention.o_proj.bias",
        ".attention.self.relative_position_bias_table": ".attention.relative_position_bias.relative_position_bias_table",
        ".attention.self.relative_position_index": ".attention.relative_position_bias.relative_position_index",
        
        # MLP block mappings (these have shorter suffixes, so they are checked after attention mappings)
        ".intermediate.dense.weight": ".mlp.fc1.weight",
        ".intermediate.dense.bias": ".mlp.fc1.bias",
        ".output.dense.weight": ".mlp.fc2.weight",
        ".output.dense.bias": ".mlp.fc2.bias",
    }
    
    mapped_count = 0
    for k, v in state_dict.items():
        mapped_key = k
        # ONLY apply translation to image_encoder keys to avoid breaking text_encoder (BERT)
        if k.startswith("image_encoder."):
            for old_suffix, new_suffix in mapping.items():
                if k.endswith(old_suffix):
                    mapped_key = k[:-len(old_suffix)] + new_suffix
                    mapped_count += 1
                    break
        new_state_dict[mapped_key] = v
        
    if mapped_count > 0:
        logger.info("Mapped %d Swin keys from old to new HF transformers format.", mapped_count)
        
    return new_state_dict


def main() -> None:
    args = parse_args()

    # 1. Load config
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.device:
        config["device"] = args.device

    # Path fallback for Kaggle environment
    kaggle_raw_dir = Path("/kaggle/input/datasets/lekimquang/pathvqa/raw")
    if not Path(config["train_csv"]).exists() and kaggle_raw_dir.exists():
        logger.info("Config dataset paths not found. Falling back to Kaggle dataset paths.")
        config["data_root"] = str(kaggle_raw_dir)
        config["train_csv"] = str(kaggle_raw_dir / "train.csv")
        config["val_csv"] = str(kaggle_raw_dir / "validation.csv")
        config["test_csv"] = str(kaggle_raw_dir / "test.csv")

    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Load checkpoint metadata first to align configurations
    logger.info("Pre-loading checkpoint from: %s to verify config", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})

    # Sync critical configuration items from checkpoint to ensure matching structure
    for key in ["open_vocab_size", "image_encoder_name", "text_encoder_name"]:
        if key in checkpoint_config:
            if config.get(key) != checkpoint_config[key]:
                logger.info(
                    "Syncing critical config parameter '%s' from checkpoint: %s -> %s",
                    key,
                    config.get(key),
                    checkpoint_config[key],
                )
                config[key] = checkpoint_config[key]

    # 2. Build Datasets
    logger.info("Loading train dataset to build vocabulary...")
    train_dataset = PathVQADataset(csv_path=config["train_csv"], data_root=config["data_root"])
    filter_dataset(train_dataset, "train")

    logger.info("Loading validation dataset...")
    val_dataset = PathVQADataset(csv_path=config["val_csv"], data_root=config["data_root"])
    filter_dataset(val_dataset, "validation")

    logger.info("Loading test dataset...")
    test_dataset = PathVQADataset(csv_path=config["test_csv"], data_root=config["data_root"])
    filter_dataset(test_dataset, "test")

    # 3. Build Vocab from train dataset (open questions only)
    logger.info("Building open-ended answer vocabulary...")
    open_vocab = train_dataset.get_answer_vocab(max_size=config["open_vocab_size"])

    # Fallback to checkpoint vocabulary if size mismatch persists
    checkpoint_vocab = checkpoint.get("open_vocab", None)
    if checkpoint_vocab is not None and len(open_vocab) != len(checkpoint_vocab):
        logger.warning(
            "Vocab length mismatch between rebuilt vocab (%d) and checkpoint vocab (%d). "
            "Using checkpoint vocab list directly to avoid weight mismatches.",
            len(open_vocab),
            len(checkpoint_vocab),
        )
        open_vocab = checkpoint_vocab

    # 4. Initialize baseline model
    model = PathVQABaseline(
        image_encoder_name=config["image_encoder_name"],
        text_encoder_name=config["text_encoder_name"],
        freeze_encoders=config["freeze_encoders"],
        dropout=config["dropout"],
        open_vocab=open_vocab,
        allow_unk_prediction=args.allow_unk_prediction,
    )

    # 5. Build Tokenizer
    logger.info("Loading tokenizer matching text encoder model: %s", model.text_encoder.model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model.text_encoder.model_name)
    except Exception as e:
        logger.warning(
            "Failed to load tokenizer for %s online: %s. Retrying with local_files_only=True...",
            model.text_encoder.model_name,
            e,
        )
        tokenizer = AutoTokenizer.from_pretrained(model.text_encoder.model_name, local_files_only=True)

    # 6. Build DataLoaders
    collate_fn = lambda batch: pathvqa_collate_fn(batch, tokenizer=tokenizer)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
    )

    # 7. Setup dummy optimizer and Trainer
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    trainer = CentralizedTrainer(
        model=model,
        train_loader=None,
        val_loader=val_loader,
        optimizer=optimizer,
        config=config,
    )

    # 8. Load checkpoint state
    cleaned_state = clean_state_dict(checkpoint["model_state_dict"])
    # strict=False allows matching buffers like relative_position_index to be handled gracefully
    trainer.model.load_state_dict(cleaned_state, strict=False)
    logger.info("Successfully loaded model state dict from checkpoint.")

    # 9. Evaluate
    logger.info("Evaluating on validation split...")
    val_metrics = trainer.evaluate(val_loader, split="validation")
    with open(output_path / "results" / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    logger.info("Evaluating on test split...")
    test_metrics = trainer.evaluate(test_loader, split="test")
    with open(output_path / "results" / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # 10. Save Predictions CSVs
    logger.info("Saving predictions...")
    trainer.save_predictions(val_loader, split="val")
    trainer.save_predictions(test_loader, split="test")

    logger.info("Evaluation completed successfully! Results are saved in %s", output_path)


if __name__ == "__main__":
    main()
