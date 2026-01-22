#!/bin/bash
# 예측 스크립트 예제
#
# 사용법:
#   bash scripts/predict.sh
#   bash scripts/predict.sh checkpoints/ldm/best_ldm.pt
#   bash scripts/predict.sh checkpoints/ldm/best_ldm.pt --output my_predictions.csv

cd "$(dirname "$0")/.."

CHECKPOINT="${1:-checkpoints/ldm/best_ldm.pt}"
OUTPUT="${OUTPUT:-predictions.csv}"

echo "========================================"
echo "LDM-LINCS Prediction"
echo "========================================"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT"
echo "========================================"

python main.py predict \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    "${@:2}"
