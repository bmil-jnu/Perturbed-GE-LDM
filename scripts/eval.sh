#!/bin/bash
# 평가 스크립트 예제
#
# 사용법:
#   bash scripts/eval.sh
#   bash scripts/eval.sh checkpoints/ldm/best_ldm.pt

cd "$(dirname "$0")/.."

CHECKPOINT="${1:-checkpoints/ldm/best_ldm.pt}"

echo "========================================"
echo "LDM-LINCS Evaluation"
echo "========================================"
echo "Checkpoint: $CHECKPOINT"
echo "========================================"

python main.py eval \
    --checkpoint "$CHECKPOINT" \
    "${@:2}"
