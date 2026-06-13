"""PathVQA baseline model combining image & text encoders and classification heads."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from src.evaluation.metrics import normalize_answer
from src.models.encoders import ImageEncoder, TextEncoder
from src.models.heads import OpenEndedHead, YesNoHead

logger = logging.getLogger(__name__)


class PathVQABaseline(nn.Module):
    """
    Baseline model for PathVQA.
    Fuses Swin/ViT image features with BERT text features, and uses
    routing by question type (closed vs open) to separate classification heads.
    """

    def __init__(
        self,
        image_encoder_name: str = "microsoft/swin-tiny-patch4-window7-224",
        text_encoder_name: str = "bert-base-uncased",
        freeze_encoders: bool = True,
        dropout: float = 0.1,
        open_vocab: list[str] | None = None,
        allow_unk_prediction: bool = True,
    ) -> None:
        super().__init__()

        self.allow_unk_prediction = allow_unk_prediction

        # Encoders
        self.image_encoder = ImageEncoder(model_name=image_encoder_name, freeze=freeze_encoders)
        self.text_encoder = TextEncoder(model_name=text_encoder_name, freeze=freeze_encoders)

        # Output dimensions
        img_dim = self.image_encoder.hidden_dim
        txt_dim = self.text_encoder.hidden_dim
        fused_dim = img_dim + txt_dim

        # Vocab for open-ended questions
        if open_vocab is None:
            self.open_vocab = ["<unk>"]
        else:
            self.open_vocab = list(open_vocab)
            if "<unk>" not in self.open_vocab:
                self.open_vocab.append("<unk>")

        self.unk_index = self.open_vocab.index("<unk>")

        # Classification Heads
        self.yes_no_head = YesNoHead(input_dim=fused_dim, dropout_prob=dropout)
        self.open_ended_head = OpenEndedHead(
            input_dim=fused_dim,
            vocab_size=len(self.open_vocab),
            dropout_prob=dropout,
        )

        logger.info(
            "PathVQABaseline initialized. Image Encoder: %s (%d dim), "
            "Text Encoder: %s (%d dim), Open-ended vocab size: %d",
            self.image_encoder.model_name,
            img_dim,
            self.text_encoder.model_name,
            txt_dim,
            len(self.open_vocab),
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        yes_no_mask: torch.Tensor,
        open_ended_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Forward pass.
        Args:
            images: Tensor of shape [B, 3, 224, 224]
            input_ids: Tensor of shape [B, L]
            attention_mask: Tensor of shape [B, L]
            yes_no_mask: Boolean tensor of shape [B] indicating yes/no (closed) questions
            open_ended_mask: Boolean tensor of shape [B] indicating open-ended questions
        Returns:
            Dict containing closed_logits, open_logits, predictions list.
        """
        batch_size = images.size(0)

        # 1. Extract features
        img_feat = self.image_encoder(images)  # [B, img_dim]
        txt_feat = self.text_encoder(input_ids, attention_mask)  # [B, txt_dim]

        # 2. Fuse features (simple concatenation)
        fused = torch.cat([img_feat, txt_feat], dim=-1)  # [B, img_dim + txt_dim]

        # 3. Routing & head prediction
        closed_logits = None
        open_logits = None

        if yes_no_mask.any():
            closed_logits = self.yes_no_head(fused[yes_no_mask])  # [N_closed, 2]

        if open_ended_mask.any():
            open_logits = self.open_ended_head(fused[open_ended_mask])  # [N_open, vocab_size]

        # 4. Generate final text predictions in batch order
        predictions: list[str] = [""] * batch_size
        raw_predictions: list[str] = [""] * batch_size
        used_unk_fallback: list[bool] = [False] * batch_size
        closed_idx = 0
        open_idx = 0

        for i in range(batch_size):
            if yes_no_mask[i]:
                # closed answer argmax
                pred_label = closed_logits[closed_idx].argmax(dim=-1).item()
                val = "yes" if pred_label == 0 else "no"
                predictions[i] = val
                raw_predictions[i] = val
                closed_idx += 1
            elif open_ended_mask[i]:
                # open answer argmax
                logits = open_logits[open_idx]
                raw_pred_label = logits.argmax(dim=-1).item()
                raw_pred_str = self.open_vocab[raw_pred_label]
                raw_predictions[i] = raw_pred_str

                if not self.allow_unk_prediction and raw_pred_str == "<unk>":
                    # Mask <unk> index with a very small logit
                    masked_logits = logits.clone()
                    masked_logits[self.unk_index] = -1e9
                    final_pred_label = masked_logits.argmax(dim=-1).item()
                    predictions[i] = self.open_vocab[final_pred_label]
                    used_unk_fallback[i] = True
                else:
                    predictions[i] = raw_pred_str
                open_idx += 1
            else:
                predictions[i] = "<unk>"
                raw_predictions[i] = "<unk>"

        return {
            "closed_logits": closed_logits,
            "open_logits": open_logits,
            "predictions": predictions,
            "raw_predictions": raw_predictions,
            "used_unk_fallback": used_unk_fallback,
        }

    def compute_loss(
        self,
        batch: dict[str, Any],
        forward_output: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """
        Compute training loss.
        Args:
            batch: Collated batch dictionary
            forward_output: Output from self.forward
        Returns:
            Dict containing overall 'loss', and optional 'closed_loss' and 'open_loss'
        """
        yes_no_mask = batch["yes_no_mask"]
        open_ended_mask = batch["open_ended_mask"]
        answers = batch["answers"]

        closed_logits = forward_output["closed_logits"]
        open_logits = forward_output["open_logits"]

        device = next(self.parameters()).device

        closed_loss = torch.tensor(0.0, device=device)
        open_loss = torch.tensor(0.0, device=device)
        n_closed = 0
        n_open = 0

        # Loss for closed-ended questions
        if yes_no_mask.any() and closed_logits is not None:
            closed_targets = []
            for i, is_closed in enumerate(yes_no_mask):
                if is_closed:
                    ans = answers[i]
                    norm = normalize_answer(ans)
                    if norm == "yes":
                        closed_targets.append(0)
                    elif norm == "no":
                        closed_targets.append(1)
                    else:
                        raise ValueError(
                            f"Invalid closed answer: '{ans}' (normalized: '{norm}'). "
                            "Closed answers must normalize to 'yes' or 'no'."
                        )
            closed_targets_tensor = torch.tensor(closed_targets, dtype=torch.long, device=device)
            closed_loss = nn.functional.cross_entropy(closed_logits, closed_targets_tensor)
            n_closed = len(closed_targets)

        # Loss for open-ended questions
        if open_ended_mask.any() and open_logits is not None:
            open_targets = []
            for i, is_open in enumerate(open_ended_mask):
                if is_open:
                    ans = answers[i]
                    norm = normalize_answer(ans)
                    if norm in self.open_vocab:
                        open_targets.append(self.open_vocab.index(norm))
                    else:
                        open_targets.append(self.unk_index)
            open_targets_tensor = torch.tensor(open_targets, dtype=torch.long, device=device)
            open_loss = nn.functional.cross_entropy(open_logits, open_targets_tensor)
            n_open = len(open_targets)

        # Calculate combined loss
        if n_closed > 0 and n_open > 0:
            loss = closed_loss + open_loss
        elif n_closed > 0:
            loss = closed_loss
        elif n_open > 0:
            loss = open_loss
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        return {
            "loss": loss,
            "closed_loss": closed_loss,
            "open_loss": open_loss,
        }
