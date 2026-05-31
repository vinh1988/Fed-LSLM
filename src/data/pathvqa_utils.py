"""PathVQA manifest columns and mentor CSV adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS: list[str] = [
    "row_id",
    "split",
    "image_id",
    "image_path",
    "question",
    "answer",
    "question_type",
    "answer_type",
]

MANIFEST_FILE_NAMES: dict[str, str] = {
    "train": "train_manifest.csv",
    "validation": "val_manifest.csv",
    "test": "test_manifest.csv",
}

YES_NO_QUESTION_PREFIXES: tuple[str, ...] = (
    "is",
    "are",
    "was",
    "were",
    "does",
    "do",
    "did",
    "can",
    "could",
    "has",
    "have",
)

MENTOR_CSV_COLUMNS: tuple[str, ...] = ("image_path", "question", "answer")


def classify_question_type(question: str) -> str:
    """Classify a question into a question_type category."""
    q = question.lower().strip()
    for prefix in YES_NO_QUESTION_PREFIXES:
        if q.startswith(prefix):
            return "yes_no"
    if q.startswith("what"):
        return "what"
    if q.startswith("where"):
        return "where"
    if q.startswith("why"):
        return "why"
    if q.startswith("how many") or q.startswith("how much"):
        return "counting"
    if q.startswith("how"):
        return "how"
    return "others"


def classify_answer_type(answer: str) -> str:
    """Classify an answer as yes_no or open_ended."""
    a = answer.lower().strip()
    if a in {"yes", "no"}:
        return "yes_no"
    return "open_ended"


def is_mentor_csv(df: pd.DataFrame) -> bool:
    """True when CSV has mentor layout (image_path, question, answer) only."""
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
    Convert mentor CSV (image_path, question, answer) to full manifest columns.

    Derives row_id, split, image_id, question_type, and answer_type in-memory;
    does not write files or duplicate data on disk.
    """
    split_default = infer_split_from_csv(csv_path)
    rows: list[dict[str, str]] = []

    for idx, row in df.iterrows():
        image_path = str(row["image_path"]).strip()
        question = str(row["question"])
        answer = str(row["answer"])
        split = infer_split_from_image_path(image_path) or split_default
        image_id = Path(image_path).stem
        rows.append(
            {
                "row_id": f"{split}_{idx:06d}",
                "split": split,
                "image_id": image_id,
                "image_path": image_path,
                "question": question,
                "answer": answer,
                "question_type": classify_question_type(question),
                "answer_type": classify_answer_type(answer),
            }
        )

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
