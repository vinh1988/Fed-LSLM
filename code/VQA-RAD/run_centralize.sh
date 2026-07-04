#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_centralize.sh
#
# Automated centralized training & validation pipeline for VQA-RAD.
# Supports: nomrag (zero-shot) and mrag variants.
#
# Features:
#   • Resume mechanism — skips training if adapter_config.json already exists,
#     skips validation if metrics.json already exists.
#   • Step tracking — each phase prints a clear banner with step number and name.
#   • Configurable — change MODEL, DATASET, EPOCHS, etc. via environment vars.
#
# Usage:
#   chmod +x run_centralize.sh
#   ./run_centralize.sh                  # run all steps
#   SKIP_NOMRAG=1 ./run_centralize.sh    # skip nomrag, run mrag only
#   SKIP_MRAG=1 ./run_centralize.sh      # skip mrag, run nomrag only
#   FORCE_RERUN=1 ./run_centralize.sh    # ignore existing outputs, re-run all
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Resolve paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CODE_DIR="${SCRIPT_DIR}"

# ── Configurable parameters (override via env vars) ─────────────────────────
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"
MODEL_SHORT="${MODEL_SHORT:-Qwen2.5-VL-3B-Instruct}"
DATASET="${DATASET:-VQA-RAD}"
SETTING="${SETTING:-centralized}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/processed_data/${DATASET}/${SETTING}}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
SEED="${SEED:-42}"
K="${K:-3}"

# Skip flags
SKIP_NOMRAG="${SKIP_NOMRAG:-0}"
SKIP_MRAG="${SKIP_MRAG:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

# ── Output directories ──────────────────────────────────────────────────────
NOMRAG_OUTPUT="${PROJECT_ROOT}/experiments/${DATASET}/${SETTING}/${MODEL_SHORT}/nomrag"
MRAG_OUTPUT="${PROJECT_ROOT}/experiments/${DATASET}/${SETTING}/${MODEL_SHORT}/mrag"

# ── Colors & formatting ─────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

TOTAL_STEPS=4

# ── Helper functions ─────────────────────────────────────────────────────────
print_banner() {
    local step_num=$1
    local step_name=$2
    local step_desc=$3
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  STEP ${step_num}/${TOTAL_STEPS}: ${step_name}${NC}"
    echo -e "${BLUE}───────────────────────────────────────────────────────────────────${NC}"
    echo -e "  ${step_desc}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_status() {
    local status=$1
    local msg=$2
    case "${status}" in
        OK)      echo -e "  ${GREEN}✔ ${msg}${NC}" ;;
        SKIP)    echo -e "  ${YELLOW}⏭ ${msg}${NC}" ;;
        FAIL)    echo -e "  ${RED}✘ ${msg}${NC}" ;;
        INFO)    echo -e "  ${CYAN}ℹ ${msg}${NC}" ;;
        RUN)     echo -e "  ${BOLD}▶ ${msg}${NC}" ;;
    esac
}

elapsed_fmt() {
    local secs=$1
    printf '%02dh:%02dm:%02ds' $((secs/3600)) $((secs%3600/60)) $((secs%60))
}

# Check if training output exists (adapter_config.json is created by model.save_pretrained)
training_exists() {
    local out_dir=$1
    [[ "${FORCE_RERUN}" == "0" ]] && [[ -f "${out_dir}/adapter_config.json" ]]
}

# Check if validation output exists (metrics.json is created at end of validation)
validation_exists() {
    local out_dir=$1
    [[ "${FORCE_RERUN}" == "0" ]] && [[ -f "${out_dir}/val/metrics.json" ]]
}

# Print metrics summary from a metrics.json file
print_metrics() {
    local metrics_file=$1
    if [[ -f "${metrics_file}" ]]; then
        echo ""
        python3 -c "
import json
with open('${metrics_file}') as f:
    m = json.load(f)
print(f\"    Closed Accuracy : {m['closed-ended']['Accuracy']:.4f} (n={m['closed-ended']['n_samples']})\")
print(f\"    Open Accuracy   : {m['open-ended']['Accuracy']:.4f} (n={m['open-ended']['n_samples']})\")
print(f\"    Open ROUGE-L    : {m['open-ended']['ROUGE-L']:.4f}\")
print(f\"    Open BERTScore  : {m['open-ended']['BERTScore']:.4f}\")
print(f\"    Hallucination   : {m['hallucination']['Hallucination Rate']:.4f}\")
"
    fi
}

# ── Print pipeline overview ─────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${BLUE}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║       CENTRALIZED VQA TRAINING & VALIDATION PIPELINE             ║${NC}"
echo -e "${BOLD}${BLUE}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
print_status INFO "Project Root : ${PROJECT_ROOT}"
print_status INFO "Model        : ${MODEL_NAME}"
print_status INFO "Dataset      : ${DATASET} / ${SETTING}"
print_status INFO "Data Dir     : ${DATA_DIR}"
print_status INFO "Epochs       : ${EPOCHS}"
print_status INFO "Batch/Accum  : ${BATCH_SIZE} / ${GRAD_ACCUM}"
print_status INFO "Learning Rate: ${LR}"
print_status INFO "Seed         : ${SEED}"
print_status INFO "RAG k        : ${K}"
echo ""

if [[ "${SKIP_NOMRAG}" == "1" ]]; then
    print_status SKIP "NOMRAG steps will be skipped (SKIP_NOMRAG=1)"
fi
if [[ "${SKIP_MRAG}" == "1" ]]; then
    print_status SKIP "MRAG steps will be skipped (SKIP_MRAG=1)"
fi
if [[ "${FORCE_RERUN}" == "1" ]]; then
    print_status INFO "Force rerun enabled — ignoring existing outputs"
fi

echo ""
echo -e "${CYAN}Pipeline steps:${NC}"
echo "  [1/4] Train    NoMRAG  (zero-shot fine-tuning)"
echo "  [2/4] Validate NoMRAG"
echo "  [3/4] Train    mRAG    (retrieval-augmented fine-tuning)"
echo "  [4/4] Validate mRAG"
echo ""

PIPELINE_START=$(date +%s)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Train NoMRAG
# ═════════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_NOMRAG}" == "1" ]]; then
    print_banner 1 "Train NoMRAG" "Skipped by user (SKIP_NOMRAG=1)"
    print_status SKIP "Skipping NoMRAG training"
elif training_exists "${NOMRAG_OUTPUT}"; then
    print_banner 1 "Train NoMRAG" "adapter_config.json found — training already done"
    print_status SKIP "Exists: ${NOMRAG_OUTPUT}/adapter_config.json"
else
    print_banner 1 "Train NoMRAG" "Zero-shot fine-tuning without retrieval augmentation"
    print_status RUN "Output: ${NOMRAG_OUTPUT}"
    mkdir -p "${NOMRAG_OUTPUT}"

    STEP_START=$(date +%s)
    uv run "${CODE_DIR}/centralize_train_nomrag.py" \
        --data_dir "${DATA_DIR}" \
        --model_name "${MODEL_NAME}" \
        --output_dir "${NOMRAG_OUTPUT}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --lr "${LR}" \
        --seed "${SEED}" \
        2>&1 | tee "${NOMRAG_OUTPUT}/train.log"
    STEP_END=$(date +%s)

    print_status OK "Training completed in $(elapsed_fmt $((STEP_END - STEP_START)))"
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Validate NoMRAG
# ═════════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_NOMRAG}" == "1" ]]; then
    print_banner 2 "Validate NoMRAG" "Skipped by user (SKIP_NOMRAG=1)"
    print_status SKIP "Skipping NoMRAG validation"
elif ! [[ -f "${NOMRAG_OUTPUT}/adapter_config.json" ]]; then
    print_banner 2 "Validate NoMRAG" "No adapter found — cannot validate"
    print_status FAIL "Missing: ${NOMRAG_OUTPUT}/adapter_config.json"
elif validation_exists "${NOMRAG_OUTPUT}"; then
    print_banner 2 "Validate NoMRAG" "metrics.json found — validation already done"
    print_status SKIP "Exists: ${NOMRAG_OUTPUT}/val/metrics.json"
    print_metrics "${NOMRAG_OUTPUT}/val/metrics.json"
else
    print_banner 2 "Validate NoMRAG" "Evaluating zero-shot fine-tuned adapter on test set"
    print_status RUN "Adapter: ${NOMRAG_OUTPUT}"
    mkdir -p "${NOMRAG_OUTPUT}/val"

    STEP_START=$(date +%s)
    uv run "${CODE_DIR}/centralize_val_nomrag.py" \
        --adapter_dir "${NOMRAG_OUTPUT}" \
        --test_csv "${DATA_DIR}/test.csv" \
        --image_root "${DATA_DIR}" \
        2>&1 | tee "${NOMRAG_OUTPUT}/val/val.log"
    STEP_END=$(date +%s)

    print_status OK "Validation completed in $(elapsed_fmt $((STEP_END - STEP_START)))"
    print_metrics "${NOMRAG_OUTPUT}/val/metrics.json"
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: Train mRAG
# ═════════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_MRAG}" == "1" ]]; then
    print_banner 3 "Train mRAG" "Skipped by user (SKIP_MRAG=1)"
    print_status SKIP "Skipping mRAG training"
elif training_exists "${MRAG_OUTPUT}"; then
    print_banner 3 "Train mRAG" "adapter_config.json found — training already done"
    print_status SKIP "Exists: ${MRAG_OUTPUT}/adapter_config.json"
else
    print_banner 3 "Train mRAG" "Retrieval-augmented fine-tuning with BiomedCLIP + FAISS"
    print_status RUN "Output: ${MRAG_OUTPUT}"
    mkdir -p "${MRAG_OUTPUT}"

    STEP_START=$(date +%s)
    uv run "${CODE_DIR}/centralize_train_mrag.py" \
        --data_dir "${DATA_DIR}" \
        --model_name "${MODEL_NAME}" \
        --output_dir "${MRAG_OUTPUT}" \
        --k "${K}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --lr "${LR}" \
        --seed "${SEED}" \
        2>&1 | tee "${MRAG_OUTPUT}/train.log"
    STEP_END=$(date +%s)

    print_status OK "Training completed in $(elapsed_fmt $((STEP_END - STEP_START)))"
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: Validate mRAG
# ═════════════════════════════════════════════════════════════════════════════
if [[ "${SKIP_MRAG}" == "1" ]]; then
    print_banner 4 "Validate mRAG" "Skipped by user (SKIP_MRAG=1)"
    print_status SKIP "Skipping mRAG validation"
elif ! [[ -f "${MRAG_OUTPUT}/adapter_config.json" ]]; then
    print_banner 4 "Validate mRAG" "No adapter found — cannot validate"
    print_status FAIL "Missing: ${MRAG_OUTPUT}/adapter_config.json"
elif validation_exists "${MRAG_OUTPUT}"; then
    print_banner 4 "Validate mRAG" "metrics.json found — validation already done"
    print_status SKIP "Exists: ${MRAG_OUTPUT}/val/metrics.json"
    print_metrics "${MRAG_OUTPUT}/val/metrics.json"
else
    print_banner 4 "Validate mRAG" "Evaluating mRAG adapter on test set"
    print_status RUN "Adapter: ${MRAG_OUTPUT}"
    mkdir -p "${MRAG_OUTPUT}/val"

    STEP_START=$(date +%s)
    uv run "${CODE_DIR}/centralize_val_mrag.py" \
        --adapter_dir "${MRAG_OUTPUT}" \
        --train_csv "${DATA_DIR}/train.csv" \
        --test_csv "${DATA_DIR}/test.csv" \
        --image_root "${DATA_DIR}" \
        --k_examples "${K}" \
        2>&1 | tee "${MRAG_OUTPUT}/val/val.log"
    STEP_END=$(date +%s)

    print_status OK "Validation completed in $(elapsed_fmt $((STEP_END - STEP_START)))"
    print_metrics "${MRAG_OUTPUT}/val/metrics.json"
fi

# ═════════════════════════════════════════════════════════════════════════════
# Final Summary
# ═════════════════════════════════════════════════════════════════════════════
PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$((PIPELINE_END - PIPELINE_START))

echo ""
echo -e "${BOLD}${GREEN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║                    PIPELINE COMPLETE                              ║${NC}"
echo -e "${BOLD}${GREEN}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
print_status OK "Total pipeline time: $(elapsed_fmt ${TOTAL_ELAPSED})"
echo ""
echo -e "  ${CYAN}Results:${NC}"
echo "    NoMRAG : ${NOMRAG_OUTPUT}/val/"
echo "    mRAG   : ${MRAG_OUTPUT}/val/"
echo ""
echo -e "  ${YELLOW}To force re-run everything:${NC}"
echo "    FORCE_RERUN=1 ./run_centralize.sh"
echo ""
