"""CLI entry point for centralized baseline training and evaluation on PathVQA."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import yaml
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pathvqa_dataset import PathVQADataset
from src.data.collate import pathvqa_collate_fn
from src.evaluation.metrics import normalize_answer
from src.models.baseline_model import PathVQABaseline
from src.training.trainer import CentralizedTrainer

# Set up logging to both console and log file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_centralized")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train centralized baseline model on PathVQA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_centralized.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (mps, cuda, cpu, auto)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--open-vocab-size",
        type=int,
        default=None,
        help="Override open vocab size",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick smoke test with subset of data and 1 epoch",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def filter_dataset(dataset: PathVQADataset, split_name: str) -> None:
    """Filter out invalid closed answers and log skipped samples."""
    closed_mask = dataset.df["question_type"] == "closed"
    invalid_mask = closed_mask & (~dataset.df["answer"].astype(str).apply(normalize_answer).isin({"yes", "no"}))
    num_invalid = invalid_mask.sum()

    if num_invalid > 0:
        invalid_samples = dataset.df[invalid_mask][["row_id", "answer"]].head(5).to_dict(orient="records")
        logger.warning(
            "Found %d closed samples with invalid answers (not 'yes' or 'no') in '%s' split. "
            "These samples will be skipped. Examples: %s",
            num_invalid,
            split_name,
            invalid_samples,
        )
        dataset.df = dataset.df[~invalid_mask].reset_index(drop=True)
    else:
        logger.info("Checked closed answers in '%s' split: all valid.", split_name)


def main() -> None:
    """Main training routine."""
    args = parse_args()

    # 1. Load config
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.device:
        config["device"] = args.device
    if args.num_epochs is not None:
        config["num_epochs"] = args.num_epochs
    if args.open_vocab_size is not None:
        config["open_vocab_size"] = args.open_vocab_size
    if args.smoke_test:
        config["output_dir"] = str(Path(config["output_dir"]) / "smoke_test")
        config["num_epochs"] = 1
        config["warmup_steps"] = 2
        config["log_every"] = 1

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

    # Save active config for reproducibility
    with open(output_path / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    # 2. Build Datasets
    logger.info("Loading train dataset...")
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

    # Log vocabulary coverage & unk ratio on training set
    open_df = train_dataset.df[train_dataset.df["question_type"] == "open"]
    all_open_answers = open_df["answer"].astype(str).str.lower().str.strip().tolist()
    unique_answers = set(all_open_answers)
    covered_unique = [ans for ans in unique_answers if ans in open_vocab]
    vocab_coverage = len(covered_unique) / len(unique_answers) if unique_answers else 0.0

    unk_count = sum(1 for ans in all_open_answers if ans not in open_vocab)
    unk_ratio = unk_count / len(all_open_answers) if all_open_answers else 0.0

    logger.info("=== Vocabulary Stats ===")
    logger.info("  Total open samples: %d", len(all_open_answers))
    logger.info("  Unique open answers: %d", len(unique_answers))
    logger.info("  Vocab unique answers covered: %d (%.2f%%)", len(covered_unique), vocab_coverage * 100)
    logger.info("  Vocab unk ratio (samples mapped to <unk>): %d (%.2f%%)", unk_count, unk_ratio * 100)

    # If in smoke test, downsample dataset to keep execution fast
    if args.smoke_test:
        logger.info("Running in SMOKE-TEST mode. Subsetting datasets to minimal sizes.")
        train_dataset = Subset(train_dataset, range(min(8, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(4, len(val_dataset))))
        test_dataset = Subset(test_dataset, range(min(4, len(test_dataset))))

    # 4. Initialize baseline model
    model = PathVQABaseline(
        image_encoder_name=config["image_encoder_name"],
        text_encoder_name=config["text_encoder_name"],
        freeze_encoders=config["freeze_encoders"],
        dropout=config["dropout"],
        open_vocab=open_vocab,
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
    )
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

    # 7. Setup optimizer & scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    total_training_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_training_steps,
    )

    # 8. Setup Trainer
    trainer = CentralizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )

    # 9. Train baseline model
    logger.info("Starting training...")
    best_metrics = trainer.train()

    # 10. Test evaluation
    logger.info("Evaluating best model on test split...")
    # Load best checkpoint before final test evaluation
    best_model_path = Path(config["output_dir"]) / "checkpoints" / "best_model.pt"
    trainer.load_checkpoint(best_model_path)

    test_metrics = trainer.evaluate(test_loader, split="test")

    with open(Path(config["output_dir"]) / "results" / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # 11. Save predictions CSVs
    logger.info("Saving predictions for val and test splits...")
    trainer.save_predictions(val_loader, split="val")
    trainer.save_predictions(test_loader, split="test")

    logger.info("==========================================================")
    if args.smoke_test:
        logger.info("✓ Smoke test passed successfully!")
    else:
        logger.info("✓ Centralized training pipeline completed successfully!")
    logger.info("==========================================================")


if __name__ == "__main__":
    main()
