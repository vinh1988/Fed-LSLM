#!/usr/bin/env bash
# Runner script for centralized VQA training & validation pipeline
# Usage: ./run.sh
set -euo pipefail

bash code/VQA-RAD/run_centralize.sh "$@"
