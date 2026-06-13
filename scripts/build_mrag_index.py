"""Build a FAISS index from PathVQA train.csv for the mRAG pipeline.

Usage (local CPU smoke test):
    python scripts/build_mrag_index.py \\
        --train-csv processed_data/PathVQA/raw/train.csv \\
        --data-root processed_data/PathVQA/raw \\
        --output-dir outputs/mrag/index \\
        --encoder biomedclip \\
        --batch-size 8 \\
        --device cpu

Usage (Kaggle GPU):
    python scripts/build_mrag_index.py \\
        --train-csv /kaggle/input/datasets/lekimquang/pathvqa/raw/train.csv \\
        --data-root /kaggle/input/datasets/lekimquang/pathvqa/raw \\
        --output-dir outputs/mrag/index \\
        --encoder biomedclip \\
        --batch-size 32 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.biomedclip_encoder import BiomedCLIPEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("build_mrag_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS index from PathVQA train.csv"
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="Path to train.csv",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory for resolving image_path columns",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/mrag/index",
        help="Directory to save FAISS index, metadata, and config",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="biomedclip",
        choices=["biomedclip", "clip"],
        help="Encoder backend: 'biomedclip' (preferred) or 'clip' (fallback)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (reduce if OOM)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device: cuda, mps, or cpu",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional: limit number of train samples (for debugging)",
    )
    return parser.parse_args()


def resolve_kaggle_paths(train_csv: str, data_root: str) -> tuple[str, str]:
    """Auto-detect Kaggle dataset paths if local paths don't exist."""
    kaggle_raw = Path("/kaggle/input/datasets/lekimquang/pathvqa/raw")
    if not Path(train_csv).exists() and kaggle_raw.exists():
        logger.info("Local paths not found — falling back to Kaggle dataset paths.")
        train_csv = str(kaggle_raw / "train.csv")
        data_root = str(kaggle_raw)
    return train_csv, data_root


def load_train_df(train_csv: str, max_samples: int | None) -> pd.DataFrame:
    """Load and validate train CSV."""
    df = pd.read_csv(train_csv)
    logger.info("Loaded train.csv: %d rows, columns: %s", len(df), list(df.columns))

    required = {"image_path", "question", "answer", "question_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv missing required columns: {missing}")

    # Add row_id if not present
    if "row_id" not in df.columns:
        split_col = df["split"] if "split" in df.columns else "train"
        df["row_id"] = [f"train_{i:06d}" for i in range(len(df))]

    if max_samples is not None:
        df = df.head(max_samples).reset_index(drop=True)
        logger.info("Limiting to %d samples (--max-samples)", max_samples)

    return df


def encode_all(
    df: pd.DataFrame,
    data_root: Path,
    encoder: BiomedCLIPEncoder,
    batch_size: int,
) -> np.ndarray:
    """Encode all samples and return embedding matrix (N, D)."""
    n = len(df)
    embeddings = []
    failed_indices = []

    logger.info("Encoding %d samples with batch_size=%d ...", n, batch_size)
    start = time.time()

    for batch_start in tqdm(range(0, n, batch_size), desc="Encoding batches"):
        batch_df = df.iloc[batch_start : batch_start + batch_size]
        images = []
        questions = []
        valid_indices = []

        for local_idx, (_, row) in enumerate(batch_df.iterrows()):
            img_path = data_root / str(row["image_path"]).strip()
            try:
                with Image.open(img_path) as img:
                    images.append(img.convert("RGB").copy())
                questions.append(str(row["question"]))
                valid_indices.append(batch_start + local_idx)
            except Exception as exc:
                logger.warning(
                    "Skipping sample %d — cannot load image %s: %s",
                    batch_start + local_idx,
                    img_path,
                    exc,
                )
                failed_indices.append(batch_start + local_idx)

        if not images:
            continue

        try:
            batch_emb = encoder.encode_image_question_batch(images, questions)
            embeddings.append(batch_emb)
        except Exception as exc:
            logger.error(
                "Batch encoding failed at offset %d: %s — skipping batch.", batch_start, exc
            )
            failed_indices.extend(valid_indices)

    elapsed = time.time() - start
    logger.info("Encoding done in %.1fs. Failed samples: %d", elapsed, len(failed_indices))

    if not embeddings:
        raise RuntimeError("All batches failed — no embeddings produced.")

    return np.vstack(embeddings).astype(np.float32), failed_indices


def build_faiss_index(embeddings: np.ndarray, embedding_dim: int):
    """Build and return a FAISS IndexFlatIP (cosine sim via normalized vecs)."""
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "faiss is required. Install with: pip install faiss-cpu"
        ) from exc

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    logger.info("FAISS IndexFlatIP built: %d vectors, dim=%d", index.ntotal, embedding_dim)
    return index


def save_outputs(
    output_dir: Path,
    index,
    df: pd.DataFrame,
    failed_indices: list[int],
    encoder: BiomedCLIPEncoder,
    args: argparse.Namespace,
    embedding_dim: int,
) -> None:
    """Save FAISS index, metadata.jsonl, and index_config.json."""
    import faiss  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out failed rows from metadata (align with FAISS IDs)
    failed_set = set(failed_indices)
    valid_rows = [
        row for i, row in enumerate(df.itertuples(index=False))
        if i not in failed_set
    ]

    # Save FAISS index
    index_path = output_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index saved: %s", index_path)

    # Save metadata.jsonl (one JSON object per line, aligned with FAISS vector IDs)
    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for row in valid_rows:
            record = {
                "row_id": str(getattr(row, "row_id", "")),
                "image_path": str(row.image_path).strip(),
                "question": str(row.question),
                "answer": str(row.answer),
                "question_type": str(row.question_type),
                "split": str(getattr(row, "split", "train")),
            }
            f.write(json.dumps(record) + "\n")
    logger.info("Metadata saved: %s (%d entries)", metadata_path, len(valid_rows))

    # Save index_config.json
    config = {
        "encoder_requested": args.encoder,
        "encoder_actual": encoder.encoder_name_actual,
        "encoder_backend": encoder._backend,
        "embedding_dim": embedding_dim,
        "n_samples": index.ntotal,
        "n_failed": len(failed_indices),
        "index_type": "IndexFlatIP",
        "train_csv": str(args.train_csv),
        "data_root": str(args.data_root),
        "batch_size": args.batch_size,
        "device": args.device,
        "max_samples": args.max_samples,
    }
    config_path = output_dir / "index_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Index config saved: %s", config_path)

    # Print summary
    print("\n" + "=" * 60)
    print("  mRAG Index Build Summary")
    print("=" * 60)
    print(f"  Train samples processed : {len(df)}")
    print(f"  Vectors in FAISS index  : {index.ntotal}")
    print(f"  Failed / skipped        : {len(failed_indices)}")
    print(f"  Embedding dimension     : {embedding_dim}")
    print(f"  Index type              : IndexFlatIP (cosine via IP)")
    print(f"  Encoder requested       : {args.encoder}")
    print(f"  Encoder actual          : {encoder.encoder_name_actual}")
    if args.encoder != "clip" and "openai/clip" in encoder.encoder_name_actual:
        print("  ⚠  FALLBACK: BiomedCLIP unavailable, used OpenAI CLIP instead")
    print(f"  Output directory        : {output_dir}")
    print("=" * 60 + "\n")


def main() -> None:
    args = parse_args()

    # Resolve Kaggle paths
    args.train_csv, args.data_root = resolve_kaggle_paths(args.train_csv, args.data_root)

    output_dir = Path(args.output_dir)
    data_root = Path(args.data_root)

    # 1. Load train data
    df = load_train_df(args.train_csv, args.max_samples)

    # 2. Load encoder
    logger.info("Initializing encoder: %s on %s", args.encoder, args.device)
    encoder = BiomedCLIPEncoder(encoder_name=args.encoder, device=args.device)
    embedding_dim = encoder.embedding_dim

    # 3. Encode all samples
    embeddings, failed_indices = encode_all(df, data_root, encoder, args.batch_size)

    # 4. Build FAISS index
    index = build_faiss_index(embeddings, embedding_dim)

    # 5. Save outputs
    save_outputs(output_dir, index, df, failed_indices, encoder, args, embedding_dim)

    logger.info("Done! Index ready at: %s", output_dir)


if __name__ == "__main__":
    main()
