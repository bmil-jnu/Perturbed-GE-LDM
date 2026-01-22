#!/bin/bash
# 훈련 스크립트 예제
#
# 사용법:
#   bash scripts/train.sh
#   bash scripts/train.sh --parallel
#   bash scripts/train.sh --config configs/my_config.yaml

cd "$(dirname "$0")/.."

# 기본 설정
CONFIG="${CONFIG:-configs/base.yaml}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"

# 환경 변수
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "========================================"
echo "LDM-LINCS Training"
echo "========================================"
echo "Config: $CONFIG"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "========================================"

python main.py train \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    "$@"
