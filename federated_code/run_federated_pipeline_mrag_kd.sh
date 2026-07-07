#!/bin/bash
# ==============================================================================
# run_federated_pipeline_mrag_kd.sh
# Automated federated training pipeline with resume capability
# Pipeline: mRAG + Knowledge Distillation
# Steps: 01-06 (client training → fedavg → KD → continue → final fedavg → validation)
#
# Usage:
#   bash federated_code/run_federated_pipeline_mrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2
#
# The script checks for existing outputs before running each step, allowing
# resumption from any point if interrupted (e.g., power failure).
# ==============================================================================

set -e  # Exit on error

# Default arguments
DATASET="VQA-RAD"
SETTING="noniid"
ALPHA=0.1
K=2
MODE="04_train_client_continue_mrag_kd"
ROCO_SLAKE_MODE="01_train_client_mrag_kd"  # Mode for ROCO+Slake adapter from step 03

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --setting)
            SETTING="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --roco_slake_mode)
            ROCO_SLAKE_MODE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --dataset DATASET       Dataset name (default: VQA-RAD)"
            echo "  --setting SETTING       Setting: iid or noniid (default: noniid)"
            echo "  --alpha ALPHA           Alpha for noniid (default: 0.1)"
            echo "  --k K                   Number of clients (default: 2)"
            echo "  --mode MODE              Script name suffix (default: 04_train_client_continue_mrag_kd)"
            echo "  --roco_slake_mode MODE   ROCO+Slake adapter mode from step 03 (default: 01_train_client_mrag_kd)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Base paths
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FEDERATED_CODE="$PROJECT_ROOT/federated_code"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments/$DATASET/$SETTING"
if [ "$SETTING" = "noniid" ]; then
    EXPERIMENTS_DIR="$EXPERIMENTS_DIR/alpha_$ALPHA"
fi
EXPERIMENTS_DIR="$EXPERIMENTS_DIR/k${K}/Qwen2.5-VL-3B-Instruct"
SERVER_DIR="$EXPERIMENTS_DIR/server"
CLIENTS_DIR="$EXPERIMENTS_DIR/clients"

# Log file
LOG_FILE="$PROJECT_ROOT/federated_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================================================="
echo "FEDERATED TRAINING PIPELINE"
echo "=============================================================================="
echo "Dataset: $DATASET"
echo "Setting: $SETTING"
echo "Alpha: $ALPHA"
echo "K: $K"
echo "Mode: $MODE"
echo "Log file: $LOG_FILE"
echo "=============================================================================="
echo ""

# Function to check if a step is already completed
check_step_done() {
    local step_name=$1
    local output_path=$2
    
    if [ -d "$output_path" ] || [ -f "$output_path" ]; then
        echo "[RESUME] $step_name already completed at $output_path"
        echo "[RESUME] Skipping $step_name (use --force to override)"
        return 0
    else
        return 1
    fi
}

# Function to run a step with error handling
run_step() {
    local step_num=$1
    local step_name=$2
    local command=$3
    local output_path=$4
    
    echo "=============================================================================="
    echo "STEP $step_num: $step_name"
    echo "=============================================================================="
    
    if check_step_done "$step_name" "$output_path"; then
        return 0
    fi
    
    echo "[START] Running $step_name"
    echo ""
    
    cd "$PROJECT_ROOT"
    if eval "$command"; then
        echo "[SUCCESS] $step_name completed"
        echo ""
        return 0
    else
        echo "[ERROR] $step_name failed"
        echo "[ERROR] Check log: $LOG_FILE"
        exit 1
    fi
}

# ==============================================================================
# STEP 01: Federated Client Training (1 epoch, mRAG with KD)
# ==============================================================================
echo "=============================================================================="
echo "STEP 01: Federated Client Training (mRAG with KD, 1 epoch)"
echo "=============================================================================="

# Check for specific client directories from step 01
STEP01_CHECK_PATTERN="$CLIENTS_DIR/client_*_01_train_client_mrag_kd"
STEP01_HAS_OUTPUT=$(ls -d $STEP01_CHECK_PATTERN 2>/dev/null | wc -l)
if [ "$STEP01_HAS_OUTPUT" -gt 0 ]; then
    echo "[RESUME] Federated Client Training (mRAG with KD, 1 epoch) already completed"
    echo "[RESUME] Found $STEP01_HAS_OUTPUT client directories matching pattern: $STEP01_CHECK_PATTERN"
    echo "[RESUME] Skipping Federated Client Training (mRAG with KD, 1 epoch) (use --force to override)"
else
    echo "[START] Running Federated Client Training (mRAG with KD, 1 epoch)"
    echo ""
    
    cd "$PROJECT_ROOT"
    if eval "uv run $FEDERATED_CODE/01_train_client_mrag_kd.py --dataset $DATASET --setting $SETTING --alpha $ALPHA --k $K"; then
        echo "[SUCCESS] Federated Client Training (mRAG with KD, 1 epoch) completed"
        echo ""
    else
        echo "[ERROR] Federated Client Training (mRAG with KD, 1 epoch) failed"
        echo "[ERROR] Check log: $LOG_FILE"
        exit 1
    fi
fi

# ==============================================================================
# STEP 02: Federated Averaging
# ==============================================================================
STEP02_OUTPUT="$SERVER_DIR/global_adapter_01_train_client_mrag_kd"
STEP02_CMD="uv run $FEDERATED_CODE/02_fedavg.py --dataset $DATASET --setting $SETTING --alpha $ALPHA --k $K --mode 01_train_client_mrag_kd"

run_step "02" "Federated Averaging" "$STEP02_CMD" "$STEP02_OUTPUT"

# ==============================================================================
# STEP 03: Knowledge Distillation (ROCO + Slake)
# ==============================================================================
STEP03_OUTPUT="$SERVER_DIR/global_adapter_roco_slake_01_train_client_mrag_kd"
STEP03_CMD="uv run $FEDERATED_CODE/03_train_kd_roco_slake.py --mode 01_train_client_mrag_kd --dataset $DATASET --setting $SETTING --alpha $ALPHA --k $K"

run_step "03" "Knowledge Distillation (ROCO + Slake)" "$STEP03_CMD" "$STEP03_OUTPUT"

# ==============================================================================
# STEP 04: Continue Training on VQA-RAD
# ==============================================================================
echo "=============================================================================="
echo "STEP 04: Continue Training on VQA-RAD"
echo "=============================================================================="

# Check for specific client directories from step 04
STEP04_CHECK_PATTERN="$CLIENTS_DIR/client_*_${MODE}"
STEP04_HAS_OUTPUT=$(ls -d $STEP04_CHECK_PATTERN 2>/dev/null | wc -l)
if [ "$STEP04_HAS_OUTPUT" -gt 0 ]; then
    echo "[RESUME] Continue Training on VQA-RAD already completed"
    echo "[RESUME] Found $STEP04_HAS_OUTPUT client directories matching pattern: $STEP04_CHECK_PATTERN"
    echo "[RESUME] Skipping Continue Training on VQA-RAD (use --force to override)"
else
    echo "[START] Running Continue Training on VQA-RAD"
    echo ""
    
    cd "$PROJECT_ROOT"
    if eval "uv run $FEDERATED_CODE/04_train_client_continue_mrag_kd.py --mode $MODE --roco_slake_mode $ROCO_SLAKE_MODE --dataset $DATASET --setting $SETTING --alpha $ALPHA --k $K"; then
        echo "[SUCCESS] Continue Training on VQA-RAD completed"
        echo ""
    else
        echo "[ERROR] Continue Training on VQA-RAD failed"
        echo "[ERROR] Check log: $LOG_FILE"
        exit 1
    fi
fi

# ==============================================================================
# STEP 05: Final Federated Averaging
# ==============================================================================
STEP05_OUTPUT="$SERVER_DIR/global_adapter_${MODE}"
STEP05_CMD="uv run $FEDERATED_CODE/05_fedavg_final_mrag_kd.py --mode $MODE --dataset $DATASET --setting $SETTING --alpha $ALPHA --k $K"

run_step "05" "Final Federated Averaging" "$STEP05_CMD" "$STEP05_OUTPUT"

# ==============================================================================
# STEP 06: Final Validation
# ==============================================================================
STEP06_OUTPUT="$EXPERIMENTS_DIR/final_val_${MODE}"
STEP06_CMD="uv run $FEDERATED_CODE/06_val_final_mrag_kd.py --mode $MODE --dataset $DATASET --setting $SETTING --alpha $ALPHA --k $K --rag_k 3"

run_step "06" "Final Validation" "$STEP06_CMD" "$STEP06_OUTPUT"

# ==============================================================================
# PIPELINE COMPLETED
# ==============================================================================
echo "=============================================================================="
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "=============================================================================="
echo "All steps completed. Results saved to:"
echo "  - Client adapters: $CLIENTS_DIR"
echo "  - Server adapters: $SERVER_DIR"
echo "  - Validation results: $EXPERIMENTS_DIR/final_val_${MODE}"
echo "=============================================================================="
echo "Log file: $LOG_FILE"
