#!/usr/bin/env python3
"""Smoke test for PathVQA Dataset and DataLoader."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.collate import pathvqa_collate_fn
from src.data.pathvqa_dataset import PathVQADataset

# Default paths: mentor layout inside this repo
PATHVQA_DIR = ROOT / "processed_data" / "PathVQA"
DEFAULT_DATA_ROOT = PATHVQA_DIR / "centralized"
DEFAULT_TRAIN_CSV = DEFAULT_DATA_ROOT / "train.csv"
DEFAULT_VAL_CSV = DEFAULT_DATA_ROOT / "validation.csv"
DEFAULT_CLIENT_CSV = PATHVQA_DIR / "federated_iid" / "k5" / "client_0.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Smoke test PathVQA Dataset and DataLoader.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Legacy root for roadmap layout: processed/train_manifest.csv, "
            "federated/iid/k_5/client_0.csv"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Image root for PathVQADataset (default: processed_data/PathVQA/centralized)",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="Centralized train CSV (default: centralized/train.csv)",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=None,
        help="Optional validation CSV (default: centralized/validation.csv)",
    )
    parser.add_argument(
        "--client-csv",
        type=Path,
        default=None,
        help="Federated client CSV (default: federated_iid/k5/client_0.csv)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for DataLoader test (default: 4)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers (default: 0)",
    )
    return parser.parse_args()


def _resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path | None]:
    """Resolve data_root, train_csv, client_csv, and optional val_csv."""
    if args.data_root is not None and args.train_csv is not None:
        data_root = args.data_root
        train_csv = args.train_csv
        fed_csv = args.client_csv or (data_root.parent / "federated_iid" / "k5" / "client_0.csv")
        val_csv = args.val_csv
        return data_root, train_csv, fed_csv, val_csv

    if args.data_dir is not None:
        data_dir = args.data_dir
        train_csv = data_dir / "processed" / "train_manifest.csv"
        fed_csv = data_dir / "federated" / "iid" / "k_5" / "client_0.csv"
        val_manifest = data_dir / "processed" / "val_manifest.csv"
        val_csv = args.val_csv or (val_manifest if val_manifest.is_file() else None)
        return data_dir, train_csv, fed_csv, val_csv

    data_root = args.data_root or DEFAULT_DATA_ROOT
    train_csv = args.train_csv or DEFAULT_TRAIN_CSV
    fed_csv = args.client_csv or DEFAULT_CLIENT_CSV
    val_csv = args.val_csv if args.val_csv is not None else DEFAULT_VAL_CSV
    return data_root, train_csv, fed_csv, val_csv


def _format_count(n: int) -> str:
    return f"{n:,}"


def _print_dataset_stats(title: str, csv_path: Path, data_root: Path) -> PathVQADataset:
    print(title)
    print(f"    CSV           : {csv_path}")
    print(f"    data_root     : {data_root}")
    dataset = PathVQADataset(csv_path=csv_path, data_root=data_root)
    n_yes_no = len(dataset.get_yes_no_indices())
    n_open = len(dataset.get_open_ended_indices())
    print(f"    Total samples : {_format_count(len(dataset))}")
    print(f"    Yes/No        : {_format_count(n_yes_no)}")
    print(f"    Open-ended    : {_format_count(n_open)}")
    print()
    return dataset


def main() -> None:
    """Run all DataLoader smoke checks."""
    args = parse_args()
    data_root, train_csv, fed_csv, val_csv = _resolve_paths(args)

    print("=" * 60)
    print("PathVQA DataLoader Smoke Test")
    print("=" * 60)
    print()

    if not train_csv.is_file():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not fed_csv.is_file():
        raise FileNotFoundError(f"Client CSV not found: {fed_csv}")

    print("[1] Centralized train dataset")
    train_ds = _print_dataset_stats("", train_csv, data_root)

    if val_csv is not None and val_csv.is_file():
        print("[1b] Centralized validation dataset")
        _print_dataset_stats("", val_csv, data_root)

    print("[2] Federated client dataset")
    fed_ds = _print_dataset_stats("", fed_csv, data_root)

    collate = partial(pathvqa_collate_fn, tokenizer=None, max_length=128)

    print("[3] Batch shape check")
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    batch = next(iter(loader))

    expected_shape = torch.Size([args.batch_size, 3, 224, 224])
    assert batch["images"].shape == expected_shape, (
        f"images shape {batch['images'].shape} != {expected_shape}"
    )
    assert len(batch["questions"]) == args.batch_size
    assert len(batch["answers"]) == args.batch_size
    assert batch["yes_no_mask"].shape == torch.Size([args.batch_size])
    assert batch["open_ended_mask"].shape == torch.Size([args.batch_size])

    print(f"    images          : {batch['images'].shape} ✓")
    print(f"    questions       : list[{len(batch['questions'])}] ✓")
    print(f"    answers         : list[{len(batch['answers'])}] ✓")
    print(f"    yes_no_mask     : {batch['yes_no_mask'].shape} ✓")
    print(f"    open_ended_mask : {batch['open_ended_mask'].shape} ✓")
    print()

    print("[4] Sample preview")
    sample = train_ds[0]
    img = sample["image"]
    print(f"    row_id        : {sample['row_id']}")
    print(f"    question      : {sample['question']!r}")
    print(f"    answer        : {sample['answer']!r}")
    print(f"    question_type : {sample['question_type']}")
    print(f"    answer_type   : {sample['answer_type']}")
    print(f"    image shape   : {img.shape}")
    print(f"    image range   : [min={img.min().item():.4f}, max={img.max().item():.4f}]")
    print()

    print("[5] Image path check", end="")
    n_check = min(5, len(train_ds.df))
    for i in range(n_check):
        row = train_ds.df.iloc[i]
        full_path = data_root / str(row["image_path"])
        if not full_path.is_file():
            raise RuntimeError(f"Missing image at {full_path} for row {row['row_id']}")
    print(" ✓")
    print()

    print("[6] Federated batch check", end="")
    fed_loader = DataLoader(
        fed_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    _ = next(iter(fed_loader))
    print(" ✓")
    print()

    print("=" * 60)
    print("All checks passed. DataLoader ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
