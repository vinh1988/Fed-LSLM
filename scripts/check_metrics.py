#!/usr/bin/env python3
"""Smoke test for PathVQA evaluation metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import aggregate_metrics, compute_metrics


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def _check(label: str, value: float, expected: float, tol: float = 1e-3) -> bool:
    ok = abs(value - expected) < tol
    mark = "✓" if ok else "✗"
    print(f"    {label:20s}: {_fmt(value)} {mark}")
    if not ok:
        print(f"      EXPECTED {_fmt(expected)}")
    return ok


def main() -> None:
    all_ok = True

    print("=" * 60)
    print("PathVQA Evaluation Metrics Smoke Test")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Case 1: All-closed perfect
    # ------------------------------------------------------------------
    print("[1] All-closed perfect")
    m = compute_metrics(
        predictions=["yes", "no", "yes"],
        references=["yes", "no", "yes"],
        question_types=["closed", "closed", "closed"],
    )
    all_ok &= _check("closed_accuracy", m["closed_accuracy"], 1.0)
    all_ok &= _check("closed_f1", m["closed_f1"], 1.0)
    assert m["n_closed"] == 3
    assert m["n_open"] == 0
    assert m["open_exact_match"] == 0.0
    assert m["open_bleu"] == 0.0
    assert m["open_rouge_l"] == 0.0
    print()

    # ------------------------------------------------------------------
    # Case 2: All-open perfect
    # ------------------------------------------------------------------
    print("[2] All-open perfect")
    m = compute_metrics(
        predictions=["lymphocyte", "necrosis", "carcinoma"],
        references=["lymphocyte", "necrosis", "carcinoma"],
        question_types=["open", "open", "open"],
    )
    all_ok &= _check("open_exact_match", m["open_exact_match"], 1.0)
    all_ok &= _check("open_bleu", m["open_bleu"], 1.0)
    all_ok &= _check("open_rouge_l", m["open_rouge_l"], 1.0)
    assert m["n_open"] == 3
    assert m["n_closed"] == 0
    assert m["closed_accuracy"] == 0.0
    assert m["closed_f1"] == 0.0
    print()

    # ------------------------------------------------------------------
    # Case 3: Mixed, partial correct
    # ------------------------------------------------------------------
    print("[3] Mixed partial")
    m = compute_metrics(
        predictions=["yes", "lymphocyte", "no", "adenocarcinoma"],
        references=["yes", "necrosis", "yes", "adenocarcinoma"],
        question_types=["closed", "open", "closed", "open"],
    )
    all_ok &= _check("closed_accuracy", m["closed_accuracy"], 0.5)
    all_ok &= _check("open_exact_match", m["open_exact_match"], 0.5)
    assert m["n_closed"] == 2
    assert m["n_open"] == 2
    print()

    # ------------------------------------------------------------------
    # Case 4: Edge case — empty open subset
    # ------------------------------------------------------------------
    print("[4] Edge case empty open subset")
    m = compute_metrics(
        predictions=["yes", "no"],
        references=["yes", "no"],
        question_types=["closed", "closed"],
    )
    all_ok &= _check("open_exact_match", m["open_exact_match"], 0.0)
    assert m["n_open"] == 0
    print("    (no open samples, no crash)")
    print()

    # ------------------------------------------------------------------
    # Case 5: Normalization
    # ------------------------------------------------------------------
    print("[5] Normalization")
    m = compute_metrics(
        predictions=["Yes ", "LYMPHOCYTE"],
        references=["yes", "lymphocyte"],
        question_types=["closed", "open"],
    )
    all_ok &= _check("closed_accuracy", m["closed_accuracy"], 1.0)
    all_ok &= _check("open_exact_match", m["open_exact_match"], 1.0)
    print()

    # ------------------------------------------------------------------
    # Case 6: Aggregation
    # ------------------------------------------------------------------
    print("[6] Aggregation")
    batch1 = compute_metrics(
        predictions=["yes", "no", "lymphocyte"],
        references=["yes", "no", "lymphocyte"],
        question_types=["closed", "closed", "open"],
    )
    batch2 = compute_metrics(
        predictions=["yes", "carcinoma", "necrosis"],
        references=["no", "carcinoma", "adenoma"],
        question_types=["closed", "open", "open"],
    )
    agg = aggregate_metrics([batch1, batch2])

    # n_closed: 2 + 1 = 3, n_open: 1 + 2 = 3, n_total: 3 + 3 = 6
    assert agg["n_total"] == 6
    assert agg["n_closed"] == 3
    assert agg["n_open"] == 3

    # closed_accuracy: batch1 = 1.0 * 2, batch2 = 0.0 * 1 → 2/3 ≈ 0.667
    all_ok &= _check("agg closed_acc", agg["closed_accuracy"], 2.0 / 3.0)

    # open_exact_match: batch1 = 1.0 * 1, batch2 = 0.5 * 2 → 2/3 ≈ 0.667
    all_ok &= _check("agg open_em", agg["open_exact_match"], 2.0 / 3.0)

    print("    aggregate_metrics([batch1, batch2]) → ok ✓")
    print()

    # ------------------------------------------------------------------
    # Validation: invalid question_type should raise
    # ------------------------------------------------------------------
    try:
        compute_metrics(["yes"], ["yes"], ["invalid_type"])
        all_ok = False
        print("    ERROR: should have raised ValueError for invalid question_type")
    except ValueError:
        pass  # expected

    try:
        compute_metrics(["yes", "no"], ["yes"], ["closed"])
        all_ok = False
        print("    ERROR: should have raised ValueError for length mismatch")
    except ValueError:
        pass  # expected

    # ------------------------------------------------------------------
    print("=" * 60)
    if all_ok:
        print("All checks passed. Metrics ready.")
    else:
        print("SOME CHECKS FAILED.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
