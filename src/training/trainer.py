"""Trainer for centralized baseline model training and evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics, normalize_answer

logger = logging.getLogger(__name__)


class CentralizedTrainer:
    """
    Handles training, evaluation, checkpointing, and prediction saving
    for the centralized PathVQA model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Config setup
        self.config = config or {}
        self.num_epochs = self.config.get("num_epochs", 20)
        self.gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        self.gradient_clip = self.config.get("gradient_clip", 1.0)
        self.log_every = self.config.get("log_every", 50)
        self.output_dir = Path(self.config.get("output_dir", "outputs/centralized_no_rag"))
        self.patience = self.config.get("patience", 5)

        # Output Directories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.result_dir = self.output_dir / "results"
        self.prediction_dir = self.output_dir / "predictions"
        self.log_dir = self.output_dir / "logs"

        for d in [self.checkpoint_dir, self.result_dir, self.prediction_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Device detection
        device_str = self.config.get("device", "auto")
        if device_str == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_str)

        logger.info("Using device: %s", self.device)
        self.model.to(self.device)

    def train(self) -> dict[str, Any]:
        """
        Run the complete training loop.
        Returns:
            Dict containing best evaluation metrics.
        """
        best_val_score = -1.0
        best_epoch = -1
        best_metrics = {}
        epochs_no_improve = 0

        train_log_file = self.log_dir / "train_log.jsonl"
        # Write/clear log file
        with open(train_log_file, "w", encoding="utf-8") as f:
            f.write("")

        for epoch in range(1, self.num_epochs + 1):
            logger.info("=== Epoch %d / %d ===", epoch, self.num_epochs)
            start_time = time.time()

            # Train phase
            train_loss = self._train_one_epoch(epoch)

            # Validation phase
            logger.info("Evaluating on validation split...")
            val_metrics = self.evaluate(self.val_loader, split="validation")

            # Composite metric: 0.5 * closed_accuracy + 0.5 * open_rouge_l
            closed_acc = val_metrics.get("closed_accuracy", 0.0)
            open_rouge = val_metrics.get("open_rouge_l", 0.0)
            val_score = 0.5 * closed_acc + 0.5 * open_rouge

            epoch_time = time.time() - start_time
            logger.info(
                "Epoch %d summary - Time: %.2fs - Train Loss: %.4f - "
                "Val Score: %.4f (Closed Acc: %.4f, Open ROUGE-L: %.4f)",
                epoch,
                epoch_time,
                train_loss,
                val_score,
                closed_acc,
                open_rouge,
            )

            # Log to JSONL
            log_entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_score": val_score,
                "val_metrics": val_metrics,
                "time_seconds": epoch_time,
            }
            with open(train_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Checkpoint saving
            is_best = val_score > best_val_score
            if is_best:
                best_val_score = val_score
                best_epoch = epoch
                best_metrics = val_metrics
                epochs_no_improve = 0
                self.save_checkpoint(epoch, val_metrics, filename="best_model.pt")
                logger.info("✓ New best model saved (score: %.4f)", val_score)
            else:
                epochs_no_improve += 1
                logger.info(
                    "No improvement in validation score for %d epoch(s). Best score: %.4f at epoch %d.",
                    epochs_no_improve,
                    best_val_score,
                    best_epoch,
                )

            # Save last checkpoint
            self.save_checkpoint(epoch, val_metrics, filename="last_model.pt")

            # Early stopping
            if epochs_no_improve >= self.patience:
                logger.warning(
                    "Early stopping triggered! No improvement for %d epochs.",
                    self.patience,
                )
                break

        logger.info(
            "Training completed. Best Val Score: %.4f at Epoch %d",
            best_val_score,
            best_epoch,
        )

        # Save final val metrics to results
        with open(self.result_dir / "val_metrics.json", "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2)

        return best_metrics

    def _train_one_epoch(self, epoch: int) -> float:
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        step_count = 0

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}", leave=False)
        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            # Move tensors to device
            device_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    device_batch[k] = v.to(self.device)
                else:
                    device_batch[k] = v

            # Forward
            outputs = self.model(
                images=device_batch["images"],
                input_ids=device_batch["input_ids"],
                attention_mask=device_batch["attention_mask"],
                yes_no_mask=device_batch["yes_no_mask"],
                open_ended_mask=device_batch["open_ended_mask"],
            )

            # Loss computation
            try:
                losses = self.model.compute_loss(device_batch, outputs)
            except ValueError as e:
                logger.error("ValueError during loss computation at step %d: %s", step, e)
                raise e

            loss = losses["loss"]

            if torch.isnan(loss):
                logger.error("Loss is NaN at step %d!", step)
                raise RuntimeError("Training loss became NaN.")

            # Backward with gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item() * self.gradient_accumulation_steps
            step_count += 1

            if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            # Progress bar update
            if step % self.log_every == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                        "closed_loss": f"{losses['closed_loss'].item():.4f}",
                        "open_loss": f"{losses['open_loss'].item():.4f}",
                    }
                )

        epoch_avg_loss = total_loss / max(step_count, 1)
        return epoch_avg_loss

    def evaluate(self, loader: torch.utils.data.DataLoader, split: str = "validation") -> dict[str, float | int]:
        """
        Evaluate the model on the given dataloader.
        """
        self.model.eval()
        all_preds = []
        all_refs = []
        all_qtypes = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {split}", leave=False):
                device_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        device_batch[k] = v.to(self.device)
                    else:
                        device_batch[k] = v

                # Forward
                outputs = self.model(
                    images=device_batch["images"],
                    input_ids=device_batch["input_ids"],
                    attention_mask=device_batch["attention_mask"],
                    yes_no_mask=device_batch["yes_no_mask"],
                    open_ended_mask=device_batch["open_ended_mask"],
                )

                all_preds.extend(outputs["predictions"])
                all_refs.extend(batch["answers"])
                all_qtypes.extend(batch["question_types"])

        metrics = compute_metrics(
            predictions=all_preds,
            references=all_refs,
            question_types=all_qtypes,
        )

        logger.info("Evaluation results on split '%s':", split)
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info("  %s: %.4f", k, v)
            else:
                logger.info("  %s: %d", k, v)

        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict[str, Any], filename: str = "model.pt") -> None:
        """
        Save training checkpoint.
        """
        filepath = self.checkpoint_dir / filename
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "open_vocab": self.model.open_vocab,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(state, filepath)

    def load_checkpoint(self, filepath: str | Path) -> None:
        """
        Load training checkpoint.
        """
        logger.info("Loading checkpoint from %s", filepath)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Successfully loaded checkpoint (epoch %d)", checkpoint["epoch"])

    def save_predictions(self, loader: torch.utils.data.DataLoader, split: str) -> None:
        """
        Generate predictions and save them to a CSV file.
        Columns: row_id, question, reference, prediction, question_type, correct
        """
        self.model.eval()
        rows = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Predicting {split}", leave=False):
                device_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        device_batch[k] = v.to(self.device)
                    else:
                        device_batch[k] = v

                # Forward
                outputs = self.model(
                    images=device_batch["images"],
                    input_ids=device_batch["input_ids"],
                    attention_mask=device_batch["attention_mask"],
                    yes_no_mask=device_batch["yes_no_mask"],
                    open_ended_mask=device_batch["open_ended_mask"],
                )

                preds = outputs["predictions"]
                raw_preds = outputs.get("raw_predictions", preds)
                used_unk_fallbacks = outputs.get("used_unk_fallback", [False] * len(preds))
                refs = batch["answers"]
                qtypes = batch["question_types"]
                questions = batch["questions"]
                row_ids = batch["row_ids"]

                for i in range(len(preds)):
                    correct = normalize_answer(preds[i]) == normalize_answer(refs[i])
                    rows.append(
                        {
                            "row_id": row_ids[i],
                            "question": questions[i],
                            "reference": refs[i],
                            "prediction": preds[i],
                            "raw_prediction": raw_preds[i],
                            "final_prediction": preds[i],
                            "question_type": qtypes[i],
                            "used_unk_fallback": int(used_unk_fallbacks[i]),
                            "correct": int(correct),
                        }
                    )

        df = pd.DataFrame(rows)
        csv_filepath = self.prediction_dir / f"{split}_predictions.csv"
        df.to_csv(csv_filepath, index=False)
        logger.info("Predictions saved to %s", csv_filepath)
