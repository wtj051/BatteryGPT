BatteryGPT: Generative Pre-training for Battery SOH and RUL Estimation
Overview
This repository contains the complete implementation of BatteryGPT, a generative pre-training transformer-based model designed for accurate and efficient battery State-of-Health (SOH) and Remaining Useful Life (RUL) estimation. BatteryGPT leverages multi-modal fusion, dynamic masking, and windowed multi-scale Transformer mechanisms to achieve state-of-the-art prediction performance.

To address reproducibility concerns, this repository now includes complete and fully functional training and evaluation pipelines. These pipelines enable users to reproduce the exact results reported in our manuscript using publicly available battery datasets, such as those from NASA, Oxford, CALCE, SNL, and ISU.

Note:
This repository includes only the training and evaluation pipelines for the proposed BatteryGPT model. Comparison models (e.g., LSTM, Transformer, PINN) are not part of this repository.

Getting Started
Requirements
Ensure you have Python 3.8+ installed. Install required packages with:

bash

pip install -r requirements.txt
Data Preparation
This code uses publicly available battery degradation datasets mentioned in our manuscript (NASA, Oxford, CALCE, SNL, ISU). Please download these datasets separately and place them in the specified directories:

kotlin

BatteryGPT/
├── data/
│   ├── NASA/
│   ├── Oxford/
│   ├── CALCE/
│   ├── SNL/
│   └── ISU/
You may find dataset sources in our manuscript.

Training and Evaluation Pipelines
The repository provides a unified, easy-to-use script (run_pipeline.sh) to train and evaluate the BatteryGPT model.

Single Entry-Point Script
To simplify usage, we provide a single bash script that handles both training and evaluation workflows.

Usage example:

bash

bash run_pipeline.sh [train|evaluate] [dataset_name]
[train|evaluate]: Choose whether to run the training or evaluation pipeline.

[dataset_name]: Choose the dataset to use (NASA, Oxford, CALCE, SNL, or ISU).

Example: Training BatteryGPT on NASA Dataset
To train BatteryGPT from scratch on the NASA dataset, run:

bash

bash run_pipeline.sh train NASA
This command:

Loads and preprocesses the NASA dataset.

Initializes the BatteryGPT model with exact hyperparameters used in the manuscript.

Runs the complete training pipeline.

Saves the trained model checkpoint (battery_gpt_NASA.pth) to checkpoints/.

Example: Evaluating BatteryGPT on NASA Dataset
After training (or using a provided checkpoint), evaluate the BatteryGPT model:

bash

bash run_pipeline.sh evaluate NASA
This command:

Loads the trained model checkpoint (battery_gpt_NASA.pth).

Evaluates performance on the specified NASA test dataset.

Prints evaluation metrics (e.g., MAE, RMSE for SOH and RUL) to the terminal.

Saves detailed evaluation results in results/NASA_eval_results.txt.

Detailed Pipeline Explanation
Training Pipeline (scripts/train.py)
The training pipeline performs the following steps:

Data loading and preprocessing:

Interpolates missing values.

Normalizes multi-modal inputs (voltage, current, temperature).

Model initialization:

Initializes BatteryGPT with the parameters from configs/battery_gpt.yaml.

Training loop:

Utilizes Adam optimizer.

Applies dynamic masking strategy during generative pre-training.

Saves model checkpoints at regular intervals.

Evaluation Pipeline (scripts/evaluate.py)
The evaluation pipeline performs:

Loading test dataset:

Processes test data using the same preprocessing steps as training.

Loading trained BatteryGPT model:

Loads the model checkpoint from the training step.

Evaluation:

Calculates SOH and RUL predictions on test data.

Computes evaluation metrics (Mean Absolute Error, Relative Error).

Saves detailed predictions and metrics.

Directory Structure
graphql

BatteryGPT/
├── checkpoints/            # Saved model checkpoints after training
├── configs/                # Configuration files with hyperparameters
│   └── battery_gpt.yaml
├── data/                   # Datasets (user-provided)
├── models/                 # BatteryGPT model definition code
├── results/                # Evaluation results
├── scripts/                # Core training and evaluation scripts
│   ├── train.py
│   └── evaluate.py
├── utils/                  # Utility functions (data processing, metrics calculation)
├── run_pipeline.sh         # Unified script for training and evaluation (single entry-point)
├── requirements.txt
└── README.md
Unified Pipeline Script (run_pipeline.sh)
This unified bash script streamlines the training and evaluation processes:

bash

#!/bin/bash

MODE=$1         # 'train' or 'evaluate'
DATASET=$2      # Dataset name: 'NASA', 'Oxford', 'CALCE', 'SNL', or 'ISU'

if [ "$MODE" = "train" ]; then
    echo "Starting training on $DATASET dataset..."
    python scripts/train.py --config configs/battery_gpt.yaml --dataset $DATASET --save checkpoints/battery_gpt_$DATASET.pth
elif [ "$MODE" = "evaluate" ]; then
    echo "Starting evaluation on $DATASET dataset..."
    python scripts/evaluate.py --config configs/battery_gpt.yaml --dataset $DATASET --checkpoint checkpoints/battery_gpt_$DATASET.pth --output results/${DATASET}_eval_results.txt
else
    echo "Invalid mode specified. Use 'train' or 'evaluate'."
fi
Make the script executable by:

bash

chmod +x run_pipeline.sh
