"""PathVQA manifest columns and mentor CSV adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS: list[str] = [
    "row_id",
    "split",
    "image_path",
    "question",
    "answer",
    "question_type",
]

MANIFEST_FILE_NAMES: dict[str, str] = {
    "train": "train_manifest.csv",
    "validation": "val_manifest.csv",
    "test": "test_manifest.csv",
}

MENTOR_CSV_COLUMNS: tuple[str, ...] = ("image_path", "question", "answer")


def is_mentor_csv(df: pd.DataFrame) -> bool:
    """True when CSV is missing one or more of the full manifest columns."""
    return set(MENTOR_CSV_COLUMNS).issubset(df.columns) and not set(
        MANIFEST_COLUMNS
    ).issubset(df.columns)


def infer_split_from_csv(csv_path: Path) -> str:
    """Infer split name from mentor CSV filename or image_path prefix."""
    name = csv_path.stem.lower()
    if name in {"train", "validation", "val", "test"}:
        return "validation" if name == "val" else name
    if name.startswith("client"):
        return "train"
    return "train"


def infer_split_from_image_path(image_path: str) -> str | None:
    """Infer split from paths like images/train/0000000.jpg."""
    parts = Path(image_path).parts
    if len(parts) >= 2 and parts[0] == "images":
        split = parts[1]
        if split in {"train", "validation", "test"}:
            return split
    return None


def adapt_mentor_csv(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """
    Convert mentor CSV to full manifest columns.

    Derives row_id, split, and question_type in-memory;
    does not write files or duplicate data on disk.
    """
    split_default = infer_split_from_csv(csv_path)
    rows: list[dict[str, str]] = []

    for idx, row in df.iterrows():
        image_path = str(row["image_path"]).strip()
        question = str(row["question"])
        answer = str(row["answer"])

        # Keep split from CSV if present, otherwise infer it
        if "split" in row and pd.notna(row["split"]):
            split = str(row["split"]).strip()
        else:
            split = infer_split_from_image_path(image_path) or split_default

        # Keep question_type directly from CSV if present
        if "question_type" in row and pd.notna(row["question_type"]):
            q_type = str(row["question_type"]).lower().strip()
        else:
            # Fallback for legacy CSV missing question_type
            a_lower = answer.lower().strip()
            q_type = "closed" if a_lower in {"yes", "no"} else "open"

        # Validate question_type is either "open" or "closed"
        if q_type not in {"open", "closed"}:
            raise ValueError(
                f"Invalid question_type '{q_type}' at row {idx} in {csv_path}. "
                "Must be either 'open' or 'closed'."
            )

        # Generate row_id if missing using split + index
        if "row_id" in row and pd.notna(row["row_id"]):
            row_id = str(row["row_id"]).strip()
        else:
            row_id = f"{split}_{idx:06d}"

        rows.append(
            {
                "row_id": row_id,
                "split": split,
                "image_path": image_path,
                "question": question,
                "answer": answer,
                "question_type": q_type,
            }
        )

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
