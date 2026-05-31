"""PyTorch Dataset for PathVQA manifests with on-disk images."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

from src.data.pathvqa_utils import MANIFEST_COLUMNS, adapt_mentor_csv, is_mentor_csv

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_image_transform() -> transforms.Compose:
    """Default ImageNet-normalized transform for 224x224 inputs."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class PathVQADataset(TorchDataset):
    """PathVQA dataset backed by a manifest CSV and images on disk."""

    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        image_transform: Callable | None = None,
        max_question_length: int = 128,
    ) -> None:
        """
        Load a PathVQA manifest and resolve images under data_root.

        Args:
            csv_path: Path to manifest CSV (train, val, test, or federated client).
            data_root: Root directory for image_path (mentor: .../PathVQA/centralized).
            image_transform: Optional torchvision transform. Uses ImageNet default if None.
            max_question_length: Character-level truncation for question text.
        """
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.max_question_length = max_question_length
        self.image_transform = (
            image_transform if image_transform is not None else default_image_transform()
        )

        self.df = pd.read_csv(self.csv_path)
        if is_mentor_csv(self.df):
            logger.info(
                "Adapting mentor CSV layout for %s (columns: %s)",
                self.csv_path,
                list(self.df.columns),
            )
            self.df = adapt_mentor_csv(self.df, self.csv_path)
        else:
            missing_cols = [c for c in MANIFEST_COLUMNS if c not in self.df.columns]
            if missing_cols:
                raise ValueError(
                    f"Manifest {self.csv_path} missing columns: {missing_cols}. "
                    f"Expected: {MANIFEST_COLUMNS}"
                )
        self.df = self.df[MANIFEST_COLUMNS]

        self._yes_no_indices: list[int] | None = None
        self._open_ended_indices: list[int] | None = None

        logger.info(
            "Loaded PathVQADataset from %s: %d samples",
            self.csv_path,
            len(self.df),
        )

    def __len__(self) -> int:
        """Return the number of QA samples in the manifest."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, str | torch.Tensor]:
        """Load one sample: image tensor and raw question/answer text."""
        row = self.df.iloc[idx]
        row_id = str(row["row_id"])
        image_path = str(row["image_path"]).strip()
        question = str(row["question"])
        if self.max_question_length > 0:
            question = question[: self.max_question_length]

        if not image_path:
            raise RuntimeError(
                f"Image path is empty for row_id={row_id}. "
                f"Expected a relative path under {self.data_root}."
            )

        full_path = self.data_root / image_path
        if not full_path.is_file():
            raise RuntimeError(
                f"Image file not found for row_id={row_id}. "
                f"Expected path: {full_path}"
            )

        with Image.open(full_path) as img:
            image = img.convert("RGB")
        image_tensor = self.image_transform(image)

        return {
            "row_id": row_id,
            "image": image_tensor,
            "question": question,
            "answer": str(row["answer"]),
            "question_type": str(row["question_type"]),
            "split": str(row["split"]),
        }

    def get_yes_no_indices(self) -> list[int]:
        """Return dataset indices where question_type is closed."""
        if self._yes_no_indices is None:
            self._yes_no_indices = self.df.index[
                self.df["question_type"] == "closed"
            ].tolist()
        return self._yes_no_indices

    def get_open_ended_indices(self) -> list[int]:
        """Return dataset indices where question_type is open."""
        if self._open_ended_indices is None:
            self._open_ended_indices = self.df.index[
                self.df["question_type"] == "open"
            ].tolist()
        return self._open_ended_indices

    def get_answer_vocab(self, max_size: int = 500) -> list[str]:
        """
        Build answer vocabulary from open-ended questions only.
        """
        open_df = self.df[self.df["question_type"] == "open"]
        normalized = open_df["answer"].astype(str).str.lower().str.strip()
        counts = Counter(normalized.tolist())
        frequent = [ans for ans, _ in counts.most_common(max_size)]
        return frequent
