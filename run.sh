#!/usr/bin/env bash
# Runner script for centralized VQA training & validation pipeline
# Usage: ./run.sh
set -euo pipefail

# bash code/VQA-RAD/run_centralize.sh "$@"
# bash federated_code/run_federated_pipeline_mrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 5
# bash federated_code/run_federated_pipeline_mrag_nokd.sh \
#   --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2 \
#   --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
#   --model_short "Qwen2.5-VL-3B-Instruct"
# bash federated_code/run_federated_pipeline_mrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 5 --model_version 2.5

# bash federated_code/run_federated_pipeline_mrag_nokd.sh \
#   --dataset VQA-RAD --setting noniid --alpha 0.1 --k 3 \
#   --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
#   --model_short "Qwen2.5-VL-3B-Instruct"

# bash federated_code/run_federated_pipeline_mrag_nokd.sh \
#   --dataset VQA-RAD --setting noniid --alpha 0.1 --k 4 \
#   --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
#   --model_short "Qwen2.5-VL-3B-Instruct"

# bash federated_code/run_federated_pipeline_mrag_nokd.sh \
#   --dataset VQA-RAD --setting noniid --alpha 0.1 --k 5 \
#   --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
#   --model_short "Qwen2.5-VL-3B-Instruct"


# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2
# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 3
# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 4
# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 5

# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2 --model_version 2
# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 3 --model_version 2
# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 4 --model_version 2
# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 5 --model_version 2

# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2 --model_version 2.5
# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 3 --model_version 2.5
# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 4 --model_version 2.5
# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 5 --model_version 2.5

# bash federated_code/run_federated_pipeline_mrag_kd.sh \
#   --dataset VQA-RAD --setting noniid --alpha 0.1 --k 3 \
#   --model_version 2

# bash federated_code/run_federated_pipeline_nomrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2 --model_version 2.5
# bash federated_code/run_federated_pipeline_nomrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 3 --model_version 2.5
# bash federated_code/run_federated_pipeline_nomrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 4 --model_version 2.5
# bash federated_code/run_federated_pipeline_nomrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 5 --model_version 2.5

# bash federated_code/run_federated_pipeline_nomrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2 --model_version 2
# bash federated_code/run_federated_pipeline_nomrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 3 --model_version 2
# bash federated_code/run_federated_pipeline_nomrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 4 --model_version 2
# bash federated_code/run_federated_pipeline_nomrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 5 --model_version 2

# bash federated_code/run_federated_pipeline_mrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2 --model_version 2
# bash federated_code/run_federated_pipeline_mrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 3 --model_version 2
# bash federated_code/run_federated_pipeline_mrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 4 --model_version 2
# bash federated_code/run_federated_pipeline_mrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 5 --model_version 2

# bash federated_code/run_federated_pipeline_mrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2 --model_version 2
# bash federated_code/run_federated_pipeline_mrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 3 --model_version 2
# bash federated_code/run_federated_pipeline_mrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 4 --model_version 2
# bash federated_code/run_federated_pipeline_mrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 5 --model_version 2

# ##=====alpha 0.5=====##1
# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 3 --model_version 2.5
# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 2 --model_version 2.5
# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 4 --model_version 2.5
# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 5 --model_version 2.5

# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 3 --model_version 2
# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 2 --model_version 2
# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 4 --model_version 2
# bash federated_code/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 5 --model_version 2

# ##=====alpha 0.5=====##2
# bash federated_code/run_federated_pipeline_mrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 3 --model_version 2.5
# bash federated_code/run_federated_pipeline_mrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 2 --model_version 2.5
# bash federated_code/run_federated_pipeline_mrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 4 --model_version 2.5
# bash federated_code/run_federated_pipeline_mrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 5 --model_version 2.5

# bash federated_code/run_federated_pipeline_mrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 3 --model_version 2
# bash federated_code/run_federated_pipeline_mrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 2 --model_version 2
# bash federated_code/run_federated_pipeline_mrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 4 --model_version 2
# bash federated_code/run_federated_pipeline_mrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 5 --model_version 2

# ##=====alpha 0.5=====##3
# bash federated_code/run_federated_pipeline_nomrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 3 --model_version 2.5
# bash federated_code/run_federated_pipeline_nomrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 2 --model_version 2.5
# bash federated_code/run_federated_pipeline_nomrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 4 --model_version 2.5
# bash federated_code/run_federated_pipeline_nomrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 5 --model_version 2.5

# bash federated_code/run_federated_pipeline_nomrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 3 --model_version 2
# bash federated_code/run_federated_pipeline_nomrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 2 --model_version 2
# bash federated_code/run_federated_pipeline_nomrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 4 --model_version 2
# bash federated_code/run_federated_pipeline_nomrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 5 --model_version 2

# ##=====alpha 0.5=====##4
# bash federated_code/run_federated_pipeline_mrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 3 --model_version 2
# bash federated_code/run_federated_pipeline_mrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 2 --model_version 2
# bash federated_code/run_federated_pipeline_mrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 4 --model_version 2
# bash federated_code/run_federated_pipeline_mrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 5 --model_version 2

# bash federated_code/run_federated_pipeline_mrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 3 --model_version 2.5
# bash federated_code/run_federated_pipeline_mrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 2 --model_version 2.5
# bash federated_code/run_federated_pipeline_mrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 4 --model_version 2.5
# bash federated_code/run_federated_pipeline_mrag_kd.sh --dataset VQA-RAD --setting noniid --alpha 0.5 --k 5 --model_version 2.5

# ##=====alpha 1.0=====##
# bash federated_code_smolvlm/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2 --model_version smolvlm500
# bash federated_code_smolvlm/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 3 --model_version smolvlm500
# bash federated_code_smolvlm/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 4 --model_version smolvlm500
# bash federated_code_smolvlm/run_federated_pipeline_nomrag_nokd.sh --dataset VQA-RAD --setting noniid --alpha 0.1 --k 5 --model_version smolvlm500