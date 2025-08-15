#!/bin/bash

MODE=$1         # 'train' or 'evaluate'
DATASET=$2      # Dataset name: 'NASA', 'Oxford', 'CALCE', 'SNL', or 'ISU'

if [ "$MODE" = "train" ]; then
    echo "Starting training on $DATASET dataset..."
    python scripts/train.py \
        --config configs/battery_gpt.yaml \
        --dataset "$DATASET" \
        --save "checkpoints/battery_gpt_${DATASET}.pth"

elif [ "$MODE" = "evaluate" ]; then
    echo "Starting evaluation on $DATASET dataset..."
    python scripts/evaluate.py \
        --config configs/battery_gpt.yaml \
        --dataset "$DATASET" \
        --checkpoint "checkpoints/battery_gpt_${DATASET}.pth" \
        --output "results/${DATASET}_eval_results.txt"

else
    echo "Invalid mode specified. Use 'train' or 'evaluate'."
fi
