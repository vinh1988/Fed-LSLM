"""Pretrained frozen image and text encoders with robust fallbacks."""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    """
    Image encoder wrapping a pretrained Hugging Face vision model.
    Loads models with robust fallback logic: Swin -> ViT -> error.
    """

    def __init__(
        self,
        model_name: str = "microsoft/swin-tiny-patch4-window7-224",
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name

        logger.info("Initializing ImageEncoder with model: %s", model_name)
        self.model = None

        # Fallback list starting from the requested model
        fallbacks = [model_name]
        if "swin" in model_name.lower() and "vit-base-patch16-224" not in model_name:
            fallbacks.append("google/vit-base-patch16-224")

        for name in fallbacks:
            try:
                self.model = AutoModel.from_pretrained(name)
                self.model_name = name
                logger.info("Successfully loaded image encoder: %s", name)
                break
            except Exception as e:
                logger.warning(
                    "Failed to load image encoder %s online: %s. Retrying with local_files_only=True...",
                    name,
                    e,
                )
                try:
                    self.model = AutoModel.from_pretrained(name, local_files_only=True)
                    self.model_name = name
                    logger.info("Successfully loaded image encoder from local cache: %s", name)
                    break
                except Exception as local_e:
                    logger.warning(
                        "Failed to load image encoder %s from local cache: %s. Trying next option...",
                        name,
                        local_e,
                    )

        if self.model is None:
            raise RuntimeError(
                f"Could not load any image encoder from request '{model_name}' "
                f"or fallbacks {fallbacks[1:]}."
            )

        # Freeze parameter weights
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            logger.info("ImageEncoder weights are frozen.")

        # Resolve hidden dimension size
        if hasattr(self.model.config, "hidden_size"):
            self.hidden_dim = self.model.config.hidden_size
        elif hasattr(self.model.config, "embed_dim"):
            self.hidden_dim = self.model.config.embed_dim
        else:
            # Fallback default
            self.hidden_dim = 768
            logger.warning(
                "Could not detect hidden_size from model config. Defaulting to %d",
                self.hidden_dim,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input image tensor of shape [B, 3, 224, 224]
        Returns:
            Pooled feature tensor of shape [B, hidden_dim]
        """
        # If model is frozen, ensure it is in eval mode and gradients are not tracked
        if not next(self.model.parameters()).requires_grad:
            self.model.eval()

        outputs = self.model(x)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            # Mean pool over spatial/sequence dimension
            return outputs.last_hidden_state.mean(dim=1)
        else:
            raise RuntimeError(
                "Vision model output does not have pooler_output or last_hidden_state."
            )


class TextEncoder(nn.Module):
    """
    Text encoder wrapping a pretrained Hugging Face transformer model.
    Loads models with robust fallback logic: requested -> BioBERT variants -> error.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name

        logger.info("Initializing TextEncoder with model: %s", model_name)
        self.model = None

        # Fallback list starting from the requested model
        fallbacks = [
            model_name,
            "dmis-lab/biobert-base-cased-v1.2",
            "dmis-lab/biobert-base-cased-v1.1",
        ]
        # Remove duplicates while keeping order
        seen = set()
        unique_fallbacks = [x for x in fallbacks if not (x in seen or seen.add(x))]

        for name in unique_fallbacks:
            try:
                self.model = AutoModel.from_pretrained(name)
                self.model_name = name
                logger.info("Successfully loaded text encoder: %s", name)
                break
            except Exception as e:
                logger.warning(
                    "Failed to load text encoder %s online: %s. Retrying with local_files_only=True...",
                    name,
                    e,
                )
                try:
                    self.model = AutoModel.from_pretrained(name, local_files_only=True)
                    self.model_name = name
                    logger.info("Successfully loaded text encoder from local cache: %s", name)
                    break
                except Exception as local_e:
                    logger.warning(
                        "Failed to load text encoder %s from local cache: %s. Trying next option...",
                        name,
                        local_e,
                    )

        if self.model is None:
            raise RuntimeError(
                f"Could not load any text encoder from request '{model_name}' "
                f"or fallbacks {unique_fallbacks[1:]}."
            )

        # Freeze parameter weights
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            logger.info("TextEncoder weights are frozen.")

        # Resolve hidden dimension size
        if hasattr(self.model.config, "hidden_size"):
            self.hidden_dim = self.model.config.hidden_size
        else:
            self.hidden_dim = 768
            logger.warning(
                "Could not detect hidden_size from model config. Defaulting to %d",
                self.hidden_dim,
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            input_ids: Tokenized question IDs of shape [B, L]
            attention_mask: Attention mask of shape [B, L]
        Returns:
            Sentence representation of shape [B, hidden_dim] (typically CLS token)
        """
        if not next(self.model.parameters()).requires_grad:
            self.model.eval()

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract CLS token representation (first token)
        return outputs.last_hidden_state[:, 0]
