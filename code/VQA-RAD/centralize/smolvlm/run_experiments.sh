#!/bin/bash
# Run mRAG and NoRAG experiments for Qwen 2 and 2.5 with resume capability

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Configuration
DATA_DIR="$PROJECT_ROOT/processed_data/VQA-RAD/centralized"
MODELS=(
    "HuggingFaceM4/SmolVLM-500M"
)
RAG_TYPES=("mrag" "nomrag")
EPOCHS=5
K=3
BATCH_SIZE=1
GRAD_ACCUM=8
LR=2e-4

echo "=========================================="
echo "mRAG & NoRAG Experiments for SmolVLM"
echo "=========================================="
echo "Data Dir: $DATA_DIR"
echo "Epochs: $EPOCHS"
echo "K (retrieval): $K"
echo "=========================================="

for RAG_TYPE in "${RAG_TYPES[@]}"; do
    for MODEL_NAME in "${MODELS[@]}"; do
        MODEL_SHORT=$(basename "$MODEL_NAME")
        OUTPUT_DIR="$PROJECT_ROOT/experiments/VQA-RAD/centralized/$MODEL_SHORT/$RAG_TYPE"
        ADAPTER_CONFIG="$OUTPUT_DIR/best_lora_adapter/adapter_config.json"
        VAL_METRICS="$OUTPUT_DIR/best_lora_adapter/val/metrics.json"

        echo ""
        echo "=========================================="
        echo "Model: $MODEL_NAME | RAG: $RAG_TYPE"
        echo "Output: $OUTPUT_DIR"
        echo "=========================================="

        # Training phase
        if [ -f "$ADAPTER_CONFIG" ] && [ "$FORCE_RERUN" != "1" ]; then
            echo "⏭ Training already done (adapter_config.json exists)"
            echo "   Set FORCE_RERUN=1 to retrain"
        else
            echo "▶ Starting training..."
            uv run python "$SCRIPT_DIR/train_${RAG_TYPE}.py" \
                --data_dir "$DATA_DIR" \
                --model_name "$MODEL_NAME" \
                --output_dir "$OUTPUT_DIR" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --gradient_accumulation_steps "$GRAD_ACCUM" \
                --lr "$LR" \
                --seed 42 \
                --k "$K"
            echo "✔ Training completed"
        fi

        # Validation phase
        if [ -f "$VAL_METRICS" ] && [ "$FORCE_RERUN" != "1" ]; then
            echo "⏭ Validation already done (metrics.json exists)"
            echo "   Set FORCE_RERUN=1 to re-validate"
        else
            echo "▶ Starting validation..."
            uv run python "$SCRIPT_DIR/val_${RAG_TYPE}.py" \
                --data_dir "$DATA_DIR" \
                --adapter_dir "$OUTPUT_DIR/best_lora_adapter" \
                --model_name "$MODEL_NAME" \
                --k_examples "$K"
            echo "✔ Validation completed"
        fi

        echo "✓ Model $MODEL_SHORT ($RAG_TYPE) completed"
    done
done

echo ""
echo "=========================================="
echo "All experiments completed successfully!"
echo "=========================================="
echo "Results:"
for RAG_TYPE in "${RAG_TYPES[@]}"; do
    for MODEL_NAME in "${MODELS[@]}"; do
        MODEL_SHORT=$(basename "$MODEL_NAME")
        OUTPUT_DIR="$PROJECT_ROOT/experiments/VQA-RAD/centralized/$MODEL_SHORT/$RAG_TYPE"
        VAL_METRICS="$OUTPUT_DIR/val/metrics.json"
        if [ -f "$VAL_METRICS" ]; then
            echo "  $MODEL_SHORT ($RAG_TYPE): $VAL_METRICS"
        fi
    done
done
