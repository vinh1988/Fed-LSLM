#!/bin/bash
# ==============================================================================
# run_federated_pipeline_nomrag_nokd.sh
# Automated federated training pipeline with resume capability
# Pipeline: No-RAG + No-KD (simple federated learning)
# Steps: 01-03 (client training → fedavg → validation)
#
# Usage:
#   bash federated_code_smolvlm_slake/run_federated_pipeline_nomrag_nokd.sh --dataset Slake --setting noniid --alpha 0.1 --k 2
#
# The script checks for existing outputs before running each step, allowing
# resumption from any point if interrupted (e.g., power failure).
# ==============================================================================

set -e  # Exit on error

# Default arguments
DATASET="Slake"
SETTING="noniid"
ALPHA=0.1
K=2
MODE="01_train_client_nomrag_nokd"  # Mode for client training
MODEL_VERSION="smolvlm500"  # Options: "2", "2.5", or "smolvlm500"

# Set model based on version
if [ "$MODEL_VERSION" = "2" ]; then
    MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
    MODEL_SHORT="Qwen2-VL-2B-Instruct"
elif [ "$MODEL_VERSION" = "2.5" ]; then
    MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
    MODEL_SHORT="Qwen2.5-VL-3B-Instruct"
else
    MODEL_NAME="HuggingFaceTB/SmolVLM-500M-Instruct"
    MODEL_SHORT="SmolVLM-500M-Instruct"
fi

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
        --model_version)
            MODEL_VERSION="$2"
            # Update model based on version
            if [ "$MODEL_VERSION" = "2" ]; then
                MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
                MODEL_SHORT="Qwen2-VL-2B-Instruct"
            elif [ "$MODEL_VERSION" = "2.5" ]; then
                MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
                MODEL_SHORT="Qwen2.5-VL-3B-Instruct"
            else
                MODEL_NAME="HuggingFaceTB/SmolVLM-500M-Instruct"
                MODEL_SHORT="SmolVLM-500M-Instruct"
            fi
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model_short)
            MODEL_SHORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --dataset DATASET       Dataset name (default: VQA-RAD)"
            echo "  --setting SETTING       Setting: iid or noniid (default: noniid)"
            echo "  --alpha ALPHA           Alpha for noniid (default: 0.1)"
            echo "  --k K                   Number of clients (default: 2)"
            echo "  --mode MODE              Script name suffix (default: 01_train_client_nomrag_nokd)"
            echo "  --model_version VERSION  Model version: 2, 2.5, or smolvlm500 (default: smolvlm500)"
            echo "  --model_name MODEL_NAME  VLM model name from HuggingFace (auto-set by model_version by default)"
            echo "  --model_short MODEL_SHORT Short model name for directory naming (auto-set by model_version by default)"
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
FEDERATED_CODE="$PROJECT_ROOT/federated_code_smolvlm_slake"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments/$DATASET/$SETTING"
if [ "$SETTING" = "noniid" ]; then
    EXPERIMENTS_DIR="$EXPERIMENTS_DIR/alpha_$ALPHA"
fi
EXPERIMENTS_DIR="$EXPERIMENTS_DIR/k${K}/$MODEL_SHORT"
SERVER_DIR="$EXPERIMENTS_DIR/server"
CLIENTS_DIR="$EXPERIMENTS_DIR/clients"

# Log file
LOG_FILE="$PROJECT_ROOT/federated_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================================================="
echo "FEDERATED TRAINING PIPELINE (No-RAG, No-KD)"
echo "=============================================================================="
echo "Dataset: $DATASET"
echo "Setting: $SETTING"
echo "Alpha: $ALPHA"
echo "K: $K"
echo "Mode: $MODE"
echo "Model: $MODEL_NAME"
echo "Model Short: $MODEL_SHORT"
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
# STEP 01: Federated Client Training (No-RAG, No-KD)
# ==============================================================================
echo "=============================================================================="
echo "STEP 01: Federated Client Training (No-RAG, No-KD, 1 epoch)"
echo "=============================================================================="

# Check for specific client directories from step 01
STEP01_CHECK_PATTERN="$CLIENTS_DIR/client_*_${MODE}"
STEP01_HAS_OUTPUT=$(ls -d $STEP01_CHECK_PATTERN 2>/dev/null | wc -l)
if [ "$STEP01_HAS_OUTPUT" -gt 0 ]; then
    echo "[RESUME] Federated Client Training (No-RAG, No-KD, 1 epoch) already completed"
    echo "[RESUME] Found $STEP01_HAS_OUTPUT client directories matching pattern: $STEP01_CHECK_PATTERN"
    echo "[RESUME] Skipping Federated Client Training (No-RAG, No-KD, 1 epoch) (use --force to override)"
else
    echo "[START] Running Federated Client Training (No-RAG, No-KD, 1 epoch)"
    echo ""
    
    cd "$PROJECT_ROOT"
    if eval "uv run $FEDERATED_CODE/01_train_client_nomrag_nokd.py --dataset $DATASET --setting $SETTING --alpha $ALPHA --k $K --model_name $MODEL_NAME --model_short $MODEL_SHORT"; then
        echo "[SUCCESS] Federated Client Training (No-RAG, No-KD, 1 epoch) completed"
        echo ""
    else
        echo "[ERROR] Federated Client Training (No-RAG, No-KD, 1 epoch) failed"
        echo "[ERROR] Check log: $LOG_FILE"
        exit 1
    fi
fi

# ==============================================================================
# STEP 02: Federated Averaging
# ==============================================================================
STEP02_OUTPUT="$SERVER_DIR/global_adapter_${MODE}"
STEP02_CMD="uv run $FEDERATED_CODE/02_fedavg.py --dataset $DATASET --setting $SETTING --alpha $ALPHA --k $K --mode $MODE --model_short $MODEL_SHORT"

run_step "02" "Federated Averaging" "$STEP02_CMD" "$STEP02_OUTPUT"

# ==============================================================================
# STEP 03: Final Validation
# ==============================================================================
STEP03_OUTPUT="$EXPERIMENTS_DIR/final_val_${MODE}"
STEP03_CMD="uv run $FEDERATED_CODE/06_val_final_nomrag_nokd.py --mode $MODE --dataset $DATASET --setting $SETTING --alpha $ALPHA --k $K --model_name $MODEL_NAME --model_short $MODEL_SHORT"

run_step "03" "Final Validation" "$STEP03_CMD" "$STEP03_OUTPUT"

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
