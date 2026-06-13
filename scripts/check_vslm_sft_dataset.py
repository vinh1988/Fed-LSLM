"""Inspect and validate a V-SLM mRAG SFT JSONL dataset.

Checks:
  - Image path exists on disk.
  - Answer is not empty.
  - Retrieved list has expected entries.
  - Current row_id is NOT in retrieved items (leakage check).
  - Same image_path is not in retrieved items (if configured).
  - Messages follow Qwen2.5-VL format.
  - Prints formatted sample prompts for visual inspection.

Usage:
    python scripts/check_vslm_sft_dataset.py \\
        --jsonl outputs/vslm_mrag/sft/train_text_only_top5.jsonl \\
        --num-samples 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate V-SLM mRAG SFT JSONL dataset")
    p.add_argument("--jsonl", type=str, required=True, help="Path to SFT JSONL file")
    p.add_argument(
        "--num-samples", type=int, default=3,
        help="Number of sample records to print in detail",
    )
    p.add_argument(
        "--data-root", type=str, default=None,
        help="Override data root for image existence checks",
    )
    p.add_argument(
        "--check-leakage", action="store_true", default=True,
        help="Check that current row_id is not in retrieved items",
    )
    p.add_argument(
        "--check-same-image", action="store_true", default=False,
        help="Check that same image_path is not in retrieved items",
    )
    p.add_argument(
        "--check-question-type-consistency", action="store_true", default=False,
        help="Check that all retrieved items have the same question_type as the current sample",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_record(
    record: dict,
    idx: int,
    check_leakage: bool,
    check_same_image: bool,
    check_question_type_consistency: bool = False,
) -> list[str]:
    """Validate a single JSONL record. Returns list of error strings."""
    errors = []
    record_id = record.get("id", f"record_{idx}")

    # 1. Check image path exists
    image_path = record.get("image", "")
    if not image_path:
        errors.append(f"[{record_id}] Missing 'image' field")
    elif not Path(image_path).exists():
        errors.append(f"[{record_id}] Image not found: {image_path}")

    # 2. Check answer is not empty
    answer = record.get("answer", "")
    if not answer or not answer.strip():
        errors.append(f"[{record_id}] Empty answer")

    # 3. Check retrieved list
    retrieved = record.get("retrieved", [])
    if not retrieved:
        errors.append(f"[{record_id}] Empty retrieved list")

    # 4. Leakage check: current row_id should not be in retrieved
    if check_leakage:
        current_id = record.get("id", "")
        for r in retrieved:
            if r.get("row_id", "") == current_id:
                errors.append(
                    f"[{record_id}] LEAKAGE: row_id '{current_id}' found in retrieved items!"
                )
                break

    # 5. Same image_path check
    if check_same_image:
        # Extract current image_path (relative) from the full path
        current_img = record.get("image", "")
        for r in retrieved:
            r_img = r.get("image_path", "")
            if r_img and current_img.endswith(r_img):
                errors.append(
                    f"[{record_id}] LEAKAGE: same image_path '{r_img}' in retrieved items!"
                )
                break

    # 6. Question-type consistency check
    if check_question_type_consistency:
        current_qt = record.get("question_type", "").strip()
        for r in retrieved:
            r_qt = r.get("question_type", "").strip()
            if r_qt and r_qt != current_qt:
                errors.append(
                    f"[{record_id}] Q-TYPE MISMATCH: sample is '{current_qt}' "
                    f"but retrieved rank={r.get('rank', '?')} is '{r_qt}'"
                )
                break  # one error per record is enough

    # 7. Check messages format
    messages = record.get("messages", [])
    if not messages:
        errors.append(f"[{record_id}] Missing 'messages' field")
    else:
        roles = [m.get("role", "") for m in messages]
        if roles[0] != "system":
            errors.append(f"[{record_id}] First message should be 'system', got '{roles[0]}'")
        if "user" not in roles:
            errors.append(f"[{record_id}] No 'user' message found")
        if "assistant" not in roles:
            errors.append(f"[{record_id}] No 'assistant' message found")

        # Check user message has image content
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if user_msgs:
            user_content = user_msgs[0].get("content", "")
            if isinstance(user_content, list):
                types = [c.get("type", "") for c in user_content]
                if "image" not in types:
                    errors.append(f"[{record_id}] User message missing image content")
                if "text" not in types:
                    errors.append(f"[{record_id}] User message missing text content")

        # Check assistant message has content
        asst_msgs = [m for m in messages if m.get("role") == "assistant"]
        if asst_msgs:
            asst_content = asst_msgs[0].get("content", "")
            if not asst_content or not str(asst_content).strip():
                errors.append(f"[{record_id}] Assistant message is empty")

    return errors


def print_sample(record: dict, idx: int) -> None:
    """Print a formatted sample record for visual inspection."""
    print(f"\n{'='*70}")
    print(f"  SAMPLE #{idx + 1} | id={record.get('id', 'N/A')}")
    print(f"{'='*70}")
    print(f"  Image       : {record.get('image', 'N/A')}")
    print(f"  Question    : {record.get('question', 'N/A')}")
    print(f"  Q-Type      : {record.get('question_type', 'N/A')}")
    print(f"  Answer (GT) : {record.get('answer', 'N/A')}")
    print(f"  # Retrieved : {len(record.get('retrieved', []))}")

    # Print retrieved items
    for r in record.get("retrieved", []):
        r_qt = r.get('question_type', 'N/A')
        print(f"    [{r.get('rank', '?')}] score={r.get('score', 0):.4f} "
              f"| type={r_qt} "
              f"| Q: {r.get('question', '')[:55]}... "
              f"| A: {r.get('answer', '')[:30]}")

    # Print formatted user prompt
    messages = record.get("messages", [])
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            print(f"\n  --- USER PROMPT ---")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image":
                        img = part.get("image", "")
                        # Show just the filename
                        print(f"  [IMAGE: .../{Path(img).name}]")
                    elif part.get("type") == "text":
                        # Strip leading/trailing newlines to avoid excessive double spacing in console
                        text_val = part.get("text", "").strip("\n")
                        for line in text_val.splitlines():
                            print(f"  {line}")
            else:
                for line in str(content).splitlines():
                    print(f"  {line}")
        elif role == "assistant":
            print(f"\n  --- ASSISTANT (target) ---")
            print(f"  {content}")

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"ERROR: JSONL file not found: {jsonl_path}")
        sys.exit(1)

    # Load all records
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"\nLoaded {len(records)} records from: {jsonl_path}\n")

    # --- Validate all records ---
    all_errors = []
    for i, record in enumerate(records):
        errs = validate_record(
            record, i,
            check_leakage=args.check_leakage,
            check_same_image=args.check_same_image,
            check_question_type_consistency=args.check_question_type_consistency,
        )
        all_errors.extend(errs)

    # --- Print stats ---
    n_total = len(records)
    q_types = [r.get("question_type", "unknown") for r in records]
    n_closed = sum(1 for qt in q_types if qt == "closed")
    n_open = sum(1 for qt in q_types if qt == "open")

    ret_counts = [len(r.get("retrieved", [])) for r in records]
    avg_ret = sum(ret_counts) / max(len(ret_counts), 1)

    answers = [r.get("answer", "") for r in records]
    avg_answer_len = sum(len(a.split()) for a in answers) / max(len(answers), 1)

    print("-" * 50)
    print(f"  Total records    : {n_total}")
    print(f"  Closed questions : {n_closed}")
    print(f"  Open questions   : {n_open}")
    print(f"  Avg retrieved/sample : {avg_ret:.1f}")
    print(f"  Avg answer length    : {avg_answer_len:.1f} words")

    # Retrieved question_type distribution analysis
    n_qt_match = 0
    n_qt_mismatch = 0
    for r_record in records:
        current_qt = r_record.get("question_type", "")
        for ret_item in r_record.get("retrieved", []):
            ret_qt = ret_item.get("question_type", "")
            if ret_qt == current_qt:
                n_qt_match += 1
            else:
                n_qt_mismatch += 1
    n_qt_total = n_qt_match + n_qt_mismatch
    if n_qt_total > 0:
        print(f"  Retrieved Q-type match : {n_qt_match}/{n_qt_total} "
              f"({100 * n_qt_match / n_qt_total:.1f}%)")
        if n_qt_mismatch > 0:
            print(f"  ⚠ Q-type mismatches   : {n_qt_mismatch}")
        else:
            print(f"  ✓ All retrieved match  : 100% same question_type")

    print("-" * 50)

    if all_errors:
        print(f"\n⚠ Found {len(all_errors)} errors:")
        for err in all_errors[:20]:
            print(f"  {err}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more errors")
    else:
        print("\n✓ All records passed validation!")

    # --- Print sample records ---
    n_print = min(args.num_samples, len(records))
    print(f"\n--- Showing {n_print} sample(s) ---")
    for i in range(n_print):
        print_sample(records[i], i)

    # Final status
    if all_errors:
        print(f"\n❌ Validation FAILED with {len(all_errors)} errors.")
        sys.exit(1)
    else:
        print(f"\n✅ Validation PASSED. {n_total} records OK.")


if __name__ == "__main__":
    main()
