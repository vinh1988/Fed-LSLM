"""Evaluation metrics for PathVQA — closed (yes/no) and open-ended subsets.

Metrics follow the standard VQA evaluation protocol used in:
  "Federated Large-Small Language Models with Multimodal
   Retrieval-Augmented Generation for Medical Visual Question Answering"

Public API
----------
compute_metrics          — core evaluation over predictions / references
compute_metrics_from_batch — convenience wrapper around collate output
aggregate_metrics        — weighted average across multiple batch dicts
"""

from __future__ import annotations

from collections import Counter
import logging
import re
import string
from typing import Any

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text normalisation (shared by EM, Accuracy, token-level F1)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize_answer(s: str) -> str:
    """Lowercase, strip, remove punctuation, collapse whitespace.

    Used internally for Exact Match, Closed Accuracy, and token splitting
    for F1 / BLEU.
    """
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = " ".join(s.split())  # collapse multiple spaces
    return s


# ---------------------------------------------------------------------------
# Per-sample helpers
# ---------------------------------------------------------------------------


def _token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between two normalised strings.

    F1 = 2 * precision * recall / (precision + recall)

    Returns 0.0 when either side has zero tokens.
    """
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def compute_metrics(
    predictions: list[str],
    references: list[str],
    question_types: list[str],
) -> dict[str, float | int]:
    """Compute evaluation metrics split by question type.

    Args:
        predictions: Model-generated answers.
        references: Ground-truth answers.
        question_types: ``"closed"`` or ``"open"`` per sample.

    Returns:
        Dict with overall, closed-subset, open-subset metrics and counts.
        See module docstring for the full key list.
    """
    # --- validation ---
    if not (len(predictions) == len(references) == len(question_types)):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, "
            f"references={len(references)}, "
            f"question_types={len(question_types)}"
        )

    valid_types = {"open", "closed"}
    for i, qt in enumerate(question_types):
        if qt not in valid_types:
            raise ValueError(
                f"Invalid question_type '{qt}' at index {i}. "
                "Must be 'open' or 'closed'."
            )

    # --- partition into subsets ---
    closed_preds: list[str] = []
    closed_refs: list[str] = []
    open_preds: list[str] = []
    open_refs: list[str] = []

    for pred, ref, qt in zip(predictions, references, question_types):
        if qt == "closed":
            closed_preds.append(pred)
            closed_refs.append(ref)
        else:
            open_preds.append(pred)
            open_refs.append(ref)

    n_total = len(predictions)
    n_closed = len(closed_preds)
    n_open = len(open_preds)

    # --- overall exact match ---
    overall_em = _exact_match(predictions, references)

    # --- closed metrics ---
    closed_accuracy = _exact_match(closed_preds, closed_refs)
    closed_f1 = _mean_token_f1(closed_preds, closed_refs)

    # --- open metrics ---
    open_em = _exact_match(open_preds, open_refs)
    open_bleu = _corpus_bleu1(open_preds, open_refs)
    open_rouge = _mean_rouge_l(open_preds, open_refs)

    return {
        "overall_exact_match": overall_em,
        "closed_accuracy": closed_accuracy,
        "closed_f1": closed_f1,
        "open_exact_match": open_em,
        "open_bleu": open_bleu,
        "open_rouge_l": open_rouge,
        "n_total": n_total,
        "n_closed": n_closed,
        "n_open": n_open,
    }


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def compute_metrics_from_batch(
    predictions: list[str],
    batch: dict[str, Any],
) -> dict[str, float | int]:
    """Wrapper that extracts references and question_types from a batch dict.

    Args:
        predictions: Model-generated answers for this batch.
        batch: Output dict from ``pathvqa_collate_fn``.

    Returns:
        Same dict as :func:`compute_metrics`.
    """
    references = batch["answers"]
    question_types = batch["question_types"]
    return compute_metrics(predictions, references, question_types)


def aggregate_metrics(
    list_of_metrics: list[dict[str, float | int]],
) -> dict[str, float | int]:
    """Weighted average of metric dicts from multiple batches.

    Closed metrics are weighted by ``n_closed``, open metrics by ``n_open``,
    and overall metrics by ``n_total``.  Counts are summed.

    .. note::

        Corpus-level BLEU is not perfectly aggregatable via weighted
        averaging.  For final validation / test evaluation, call
        :func:`compute_metrics` once over **all** predictions and references
        instead of aggregating batch results.
    """
    if not list_of_metrics:
        return {
            "overall_exact_match": 0.0,
            "closed_accuracy": 0.0,
            "closed_f1": 0.0,
            "open_exact_match": 0.0,
            "open_bleu": 0.0,
            "open_rouge_l": 0.0,
            "n_total": 0,
            "n_closed": 0,
            "n_open": 0,
        }

    total_n = sum(m["n_total"] for m in list_of_metrics)
    total_closed = sum(m["n_closed"] for m in list_of_metrics)
    total_open = sum(m["n_open"] for m in list_of_metrics)

    def _wavg(key: str, weight_key: str, total_weight: int) -> float:
        if total_weight == 0:
            return 0.0
        return sum(m[key] * m[weight_key] for m in list_of_metrics) / total_weight

    return {
        "overall_exact_match": _wavg("overall_exact_match", "n_total", total_n),
        "closed_accuracy": _wavg("closed_accuracy", "n_closed", total_closed),
        "closed_f1": _wavg("closed_f1", "n_closed", total_closed),
        "open_exact_match": _wavg("open_exact_match", "n_open", total_open),
        "open_bleu": _wavg("open_bleu", "n_open", total_open),
        "open_rouge_l": _wavg("open_rouge_l", "n_open", total_open),
        "n_total": total_n,
        "n_closed": total_closed,
        "n_open": total_open,
    }


# ---------------------------------------------------------------------------
# Internal metric implementations
# ---------------------------------------------------------------------------


def _exact_match(predictions: list[str], references: list[str]) -> float:
    """Fraction of samples where normalised pred == normalised ref."""
    if not predictions:
        return 0.0
    matches = sum(
        1
        for p, r in zip(predictions, references)
        if normalize_answer(p) == normalize_answer(r)
    )
    return matches / len(predictions)


def _mean_token_f1(predictions: list[str], references: list[str]) -> float:
    """Mean token-level F1 over all pairs."""
    if not predictions:
        return 0.0
    return sum(_token_f1(p, r) for p, r in zip(predictions, references)) / len(
        predictions
    )


def _corpus_bleu1(predictions: list[str], references: list[str]) -> float:
    """Corpus-level BLEU-1 using normalised token lists.

    Uses SmoothingFunction.method1 for robustness on short answers.
    """
    if not predictions:
        return 0.0

    refs_tokenized = [[normalize_answer(r).split()] for r in references]
    preds_tokenized = [normalize_answer(p).split() for p in predictions]

    # Filter out empty hypotheses (edge case: prediction is purely punctuation)
    filtered_refs = []
    filtered_preds = []
    for ref, pred in zip(refs_tokenized, preds_tokenized):
        if pred and ref[0]:
            filtered_refs.append(ref)
            filtered_preds.append(pred)

    if not filtered_preds:
        return 0.0

    smooth = SmoothingFunction().method1
    return corpus_bleu(
        filtered_refs,
        filtered_preds,
        weights=(1.0, 0.0, 0.0, 0.0),
        smoothing_function=smooth,
    )


def _mean_rouge_l(predictions: list[str], references: list[str]) -> float:
    """Mean ROUGE-L F-measure over all pairs."""
    if not predictions:
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    total = 0.0
    for pred, ref in zip(predictions, references):
        score = scorer.score(normalize_answer(ref), normalize_answer(pred))
        total += score["rougeL"].fmeasure
    return total / len(predictions)
