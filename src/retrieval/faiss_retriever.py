"""FAISS-based retriever for the mRAG pipeline.

Loads a pre-built FAISS index and metadata file, then retrieves the
top-k most similar training examples for a given image+question query.

Supports optional ``filter_question_type`` to restrict retrieval to
examples of the same question type (closed-only or open-only).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FAISSRetriever:
    """Retrieve similar PathVQA training examples using a FAISS index.

    Attributes:
        index_path: Path to the ``faiss.index`` file.
        metadata_path: Path to the ``metadata.jsonl`` file.
        index_config: Dict loaded from ``index_config.json``.
        encoder: Loaded :class:`BiomedCLIPEncoder` instance.
        n_samples: Number of vectors in the index.
        embedding_dim: Embedding dimensionality.
    """

    def __init__(self) -> None:
        self._index = None
        self._metadata: list[dict] = []
        self._closed_ids: list[int] = []  # FAISS vector IDs for closed samples
        self._open_ids: list[int] = []    # FAISS vector IDs for open samples
        self.index_config: dict = {}
        self.encoder = None
        self.n_samples: int = 0
        self.embedding_dim: int = 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, index_dir: str | Path) -> "FAISSRetriever":
        """Load FAISS index, metadata, config, and encoder from *index_dir*.

        Args:
            index_dir: Directory containing ``faiss.index``,
                ``metadata.jsonl``, and ``index_config.json``.

        Returns:
            self (for chaining).
        """
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "faiss is required. Install with: pip install faiss-cpu"
            ) from exc

        index_dir = Path(index_dir)

        # Load config
        config_path = index_dir / "index_config.json"
        with open(config_path, encoding="utf-8") as f:
            self.index_config = json.load(f)

        self.embedding_dim = self.index_config["embedding_dim"]
        encoder_actual = self.index_config.get("encoder_actual", "unknown")

        logger.info(
            "Loading FAISS index from %s | encoder_actual=%s | dim=%d",
            index_dir,
            encoder_actual,
            self.embedding_dim,
        )

        # Load FAISS index
        index_path = index_dir / "faiss.index"
        self._index = faiss.read_index(str(index_path))
        self.n_samples = self._index.ntotal
        logger.info("FAISS index loaded: %d vectors", self.n_samples)

        # Load metadata
        metadata_path = index_dir / "metadata.jsonl"
        with open(metadata_path, encoding="utf-8") as f:
            self._metadata = [json.loads(line) for line in f if line.strip()]
        logger.info("Metadata loaded: %d entries", len(self._metadata))

        if len(self._metadata) != self.n_samples:
            logger.warning(
                "Metadata count (%d) != FAISS vector count (%d). "
                "Retrieval may return incorrect samples.",
                len(self._metadata),
                self.n_samples,
            )

        # Pre-compute per-type ID lists for fast filtering
        self._closed_ids = [
            i for i, m in enumerate(self._metadata) if m.get("question_type") == "closed"
        ]
        self._open_ids = [
            i for i, m in enumerate(self._metadata) if m.get("question_type") == "open"
        ]
        logger.info(
            "Type split: %d closed, %d open in index",
            len(self._closed_ids),
            len(self._open_ids),
        )

        # Load encoder (same config as used during build)
        encoder_requested = self.index_config.get("encoder_requested", "biomedclip")
        device = self.index_config.get("device", "cpu")
        self._load_encoder(encoder_requested, encoder_actual, device)

        return self

    def _load_encoder(
        self, encoder_requested: str, encoder_actual: str, device: str
    ) -> None:
        """Instantiate the encoder matching what was used to build the index."""
        from src.retrieval.biomedclip_encoder import BiomedCLIPEncoder

        # If the index was built with the CLIP fallback, request clip directly
        # to avoid a wasted BiomedCLIP load attempt.
        if "openai/clip" in encoder_actual:
            effective_request = "clip"
        else:
            effective_request = encoder_requested

        self.encoder = BiomedCLIPEncoder(encoder_name=effective_request, device=device)

        if self.encoder.encoder_name_actual != encoder_actual:
            logger.warning(
                "Encoder mismatch! Index was built with '%s' but loaded '%s'. "
                "Retrieval quality may be degraded.",
                encoder_actual,
                self.encoder.encoder_name_actual,
            )

    # ------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        image: Image.Image,
        question: str,
        top_k: int = 5,
        filter_question_type: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve top-k similar training examples for image+question.

        Args:
            image: Query PIL image.
            question: Query question text.
            top_k: Number of results to return.
            filter_question_type: ``"closed"``, ``"open"``, or ``None``
                (no filter). When set, only examples of the matching
                question type are returned.

        Returns:
            List of result dicts sorted by descending score.
        """
        query_emb = self.encoder.encode_image_question(image, question)
        return self.retrieve_by_embedding(
            query_emb, top_k=top_k, filter_question_type=filter_question_type
        )

    def retrieve_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_question_type: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve top-k results given a pre-computed query embedding.

        Args:
            query_embedding: Unit-norm float32 array of shape (D,).
            top_k: Number of results to return.
            filter_question_type: Optional question-type filter.

        Returns:
            List of result dicts sorted by descending score.
        """
        import faiss  # type: ignore

        query = query_embedding.reshape(1, -1).astype(np.float32)

        if filter_question_type is None:
            # Standard search over full index
            scores, indices = self._index.search(query, top_k)
            scores = scores[0].tolist()
            indices = indices[0].tolist()
        else:
            # Search a larger pool then filter by type
            candidate_pool = (
                self._closed_ids
                if filter_question_type == "closed"
                else self._open_ids
            )
            if not candidate_pool:
                logger.warning(
                    "No samples of type '%s' in index — returning empty.",
                    filter_question_type,
                )
                return []

            # Build an in-memory sub-index for the requested type
            sub_vecs = self._get_sub_vectors(candidate_pool)
            sub_index = faiss.IndexFlatIP(self.embedding_dim)
            sub_index.add(sub_vecs)

            k_search = min(top_k, len(candidate_pool))
            sub_scores, sub_idx = sub_index.search(query, k_search)
            scores = sub_scores[0].tolist()
            # Map sub-index positions back to global metadata indices
            indices = [candidate_pool[i] for i in sub_idx[0].tolist()]

        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "question": meta.get("question", ""),
                    "answer": meta.get("answer", ""),
                    "question_type": meta.get("question_type", ""),
                    "image_path": meta.get("image_path", ""),
                    "row_id": meta.get("row_id", ""),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sub_vectors(self, global_ids: list[int]) -> np.ndarray:
        """Reconstruct vectors for given global FAISS IDs.

        Note: This requires the index to support ``reconstruct`` (e.g.,
        ``IndexFlatIP`` does). For large filtered sets this is fast enough
        because PathVQA only has ~19k train samples.
        """
        vecs = np.zeros((len(global_ids), self.embedding_dim), dtype=np.float32)
        for i, gid in enumerate(global_ids):
            self._index.reconstruct(gid, vecs[i])
        return vecs
