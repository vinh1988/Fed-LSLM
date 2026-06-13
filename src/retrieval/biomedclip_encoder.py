"""BiomedCLIP encoder for multimodal retrieval in mRAG pipeline.

Priority:
  1. microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (via open_clip)
  2. openai/clip-vit-base-patch32 (fallback via transformers CLIPModel)

When fallback occurs, `encoder_name_actual` reflects the model that was
actually loaded so callers can persist this in index_config.json.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Canonical model identifiers
BIOMEDCLIP_OPEN_CLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BIOMEDCLIP_HF_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CLIP_FALLBACK_HF_ID = "openai/clip-vit-base-patch32"


class BiomedCLIPEncoder:
    """Multimodal encoder that embeds images and text questions.

    Attempts to load BiomedCLIP via open_clip first. Falls back to
    OpenAI CLIP via transformers if open_clip is not installed or if
    the BiomedCLIP weights cannot be fetched.

    Attributes:
        encoder_name_requested: The encoder name requested by the caller.
        encoder_name_actual: The encoder that was *actually* loaded
            (may differ if fallback occurred).
        embedding_dim: Output dimension of the combined embedding.
        device: Torch device used for inference.
    """

    def __init__(
        self,
        encoder_name: str = "biomedclip",
        device: str = "cpu",
    ) -> None:
        """Load encoder.

        Args:
            encoder_name: ``"biomedclip"`` or ``"clip"``. Anything other
                than ``"clip"`` triggers a BiomedCLIP load attempt first.
            device: ``"cuda"``, ``"mps"``, or ``"cpu"``.
        """
        self.encoder_name_requested = encoder_name
        self.device = torch.device(device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        ))
        self._backend: str  # "open_clip" | "transformers"
        self._model = None
        self._preprocess = None
        self._tokenizer = None

        if encoder_name == "clip":
            logger.info("encoder_name='clip' requested — loading CLIP fallback directly.")
            self._load_clip_fallback()
        else:
            self._try_load_biomedclip()

        logger.info(
            "Encoder ready | requested=%s | actual=%s | backend=%s | dim=%d | device=%s",
            self.encoder_name_requested,
            self.encoder_name_actual,
            self._backend,
            self.embedding_dim,
            self.device,
        )

    def _extract_tensor(self, output) -> torch.Tensor:
        """Extract a raw torch.Tensor from a model output (e.g., BaseModelOutputWithPooling)."""
        if isinstance(output, torch.Tensor):
            return output
        
        # Try common HF model output attributes
        for attr in ("pooler_output", "last_hidden_state", "image_embeds", "text_embeds"):
            val = getattr(output, attr, None)
            if isinstance(val, torch.Tensor):
                return val
                
        # If it is dict-like
        if hasattr(output, "keys") and hasattr(output, "values"):
            for key in ("pooler_output", "last_hidden_state", "image_embeds", "text_embeds"):
                if key in output and isinstance(output[key], torch.Tensor):
                    return output[key]
            # fallback to the first tensor value
            for val in output.values():
                if isinstance(val, torch.Tensor):
                    return val
                    
        # If it is indexable (tuple/list)
        if isinstance(output, (list, tuple)) and len(output) > 0:
            if isinstance(output[0], torch.Tensor):
                return output[0]
                
        raise TypeError(
            f"Cannot extract torch.Tensor from output of type {type(output)}. "
            f"Available attributes: {dir(output)}"
        )

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _try_load_biomedclip(self) -> None:
        """Attempt to load BiomedCLIP via open_clip; fallback on failure."""
        try:
            import open_clip  # type: ignore
        except ImportError:
            logger.warning(
                "open_clip not installed. Falling back to OpenAI CLIP. "
                "Install with: pip install open_clip_torch"
            )
            self._load_clip_fallback()
            return

        try:
            logger.info("Loading BiomedCLIP via open_clip: %s", BIOMEDCLIP_OPEN_CLIP_ID)
            model, _, preprocess = open_clip.create_model_and_transforms(
                BIOMEDCLIP_OPEN_CLIP_ID
            )
            tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_OPEN_CLIP_ID)
            model = model.to(self.device).eval()

            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer
            self._backend = "open_clip"
            self.encoder_name_actual = BIOMEDCLIP_OPEN_CLIP_ID

            # Probe embedding dim
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            with torch.no_grad():
                feat = self._model.encode_image(dummy)
                feat = self._extract_tensor(feat)
            self.embedding_dim = int(feat.shape[-1])

        except Exception as exc:
            logger.warning(
                "BiomedCLIP load failed (%s). Falling back to OpenAI CLIP.", exc
            )
            self._load_clip_fallback()

    def _load_clip_fallback(self) -> None:
        """Load OpenAI CLIP via transformers as fallback."""
        from transformers import CLIPModel, CLIPProcessor  # type: ignore

        logger.info("Loading CLIP fallback via transformers: %s", CLIP_FALLBACK_HF_ID)
        model = CLIPModel.from_pretrained(CLIP_FALLBACK_HF_ID)
        processor = CLIPProcessor.from_pretrained(CLIP_FALLBACK_HF_ID)
        model = model.to(self.device).eval()

        self._model = model
        self._processor = processor
        self._backend = "transformers"
        self.encoder_name_actual = CLIP_FALLBACK_HF_ID

        # Probe embedding dim via vision config
        self.embedding_dim = model.config.projection_dim

    # ------------------------------------------------------------------
    # Public encoding API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a single PIL image into a unit-norm embedding vector.

        Args:
            image: RGB PIL image.

        Returns:
            1-D float32 numpy array of shape (embedding_dim,).
        """
        if self._backend == "open_clip":
            tensor = self._preprocess(image).unsqueeze(0).to(self.device)
            feat = self._model.encode_image(tensor)
            feat = self._extract_tensor(feat)
        else:
            inputs = self._processor(images=image, return_tensors="pt").to(self.device)
            feat = self._model.get_image_features(**inputs)
            feat = self._extract_tensor(feat)

        feat = feat.float()
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_text(self, question: str) -> np.ndarray:
        """Encode a question string into a unit-norm embedding vector.

        Args:
            question: Raw question text.

        Returns:
            1-D float32 numpy array of shape (embedding_dim,).
        """
        if self._backend == "open_clip":
            tokens = self._tokenizer([question]).to(self.device)
            feat = self._model.encode_text(tokens)
            feat = self._extract_tensor(feat)
        else:
            inputs = self._processor(
                text=[question], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            feat = self._model.get_text_features(**inputs)
            feat = self._extract_tensor(feat)

        feat = feat.float()
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_image_question(
        self, image: Image.Image, question: str
    ) -> np.ndarray:
        """Compute combined multimodal embedding.

        ``combined = normalize(img_emb) + normalize(txt_emb)``
        then re-normalize the sum to unit norm.

        Args:
            image: RGB PIL image.
            question: Raw question text.

        Returns:
            1-D float32 numpy array of shape (embedding_dim,).
        """
        img_emb = self.encode_image(image)   # already unit-norm
        txt_emb = self.encode_text(question)  # already unit-norm
        combined = img_emb + txt_emb
        norm = np.linalg.norm(combined) + 1e-8
        return (combined / norm).astype(np.float32)

    # ------------------------------------------------------------------
    # Batch helpers (used by build_mrag_index.py)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Batch-encode a list of PIL images.

        Returns:
            Float32 array of shape (N, embedding_dim), unit-norm rows.
        """
        if self._backend == "open_clip":
            tensors = torch.stack(
                [self._preprocess(img) for img in images], dim=0
            ).to(self.device)
            feats = self._model.encode_image(tensors)
            feats = self._extract_tensor(feats).float()
        else:
            inputs = self._processor(
                images=images, return_tensors="pt"
            ).to(self.device)
            feats = self._model.get_image_features(**inputs)
            feats = self._extract_tensor(feats).float()

        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_text_batch(self, questions: list[str]) -> np.ndarray:
        """Batch-encode a list of question strings.

        Returns:
            Float32 array of shape (N, embedding_dim), unit-norm rows.
        """
        if self._backend == "open_clip":
            tokens = self._tokenizer(questions).to(self.device)
            feats = self._model.encode_text(tokens)
            feats = self._extract_tensor(feats).float()
        else:
            inputs = self._processor(
                text=questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)
            feats = self._model.get_text_features(**inputs)
            feats = self._extract_tensor(feats).float()

        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats.cpu().numpy().astype(np.float32)

    def encode_image_question_batch(
        self, images: list[Image.Image], questions: list[str]
    ) -> np.ndarray:
        """Batch combined multimodal embeddings.

        Returns:
            Float32 array of shape (N, embedding_dim), unit-norm rows.
        """
        img_embs = self.encode_image_batch(images)    # (N, D) unit-norm
        txt_embs = self.encode_text_batch(questions)  # (N, D) unit-norm
        combined = img_embs + txt_embs
        norms = np.linalg.norm(combined, axis=-1, keepdims=True) + 1e-8
        return (combined / norms).astype(np.float32)
