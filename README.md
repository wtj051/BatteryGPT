# 🚀 BatteryGPT: Generative Pre-training for Battery SOH and RUL Estimation

---

## 📚 Overview

**BatteryGPT** is a transformer-based generative pre-training model for accurate Battery State-of-Health (SOH) and Remaining Useful Life (RUL) estimation.  
This repository provides complete training and evaluation pipelines to fully reproduce the results reported in our paper.

> ⚠️ **Note:** This repository only includes the training and evaluation code for BatteryGPT. Comparison models (e.g., LSTM, Transformer, PINN) are NOT included.

---

## 🏁 Getting Started

### ⚙️ Requirements

- Python ≥ 3.8

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

### 📂 Data Preparation

Download the following public battery degradation datasets and place them under the `data/` directory:

```
BatteryGPT/
├── data/
│   ├── NASA/
│   ├── Oxford/
│   ├── CALCE/
│   ├── SNL/
│   └── ISU/
```
Refer to the manuscript for dataset sources.

---

## 🛠️ Training and Evaluation Pipelines

A unified entry-point script, `run_pipeline.sh`, is provided for both training and evaluation.

### Command Format

```bash
bash run_pipeline.sh [train|evaluate] [dataset_name]
```

- `[train|evaluate]`: Choose either the training or evaluation workflow
- `[dataset_name]`: Dataset name (NASA, Oxford, CALCE, SNL, ISU)

---

### 📈 Example: Training

To train BatteryGPT on the NASA dataset:

```bash
bash run_pipeline.sh train NASA
```

- Loads and preprocesses the NASA dataset
- Initializes BatteryGPT model (with manuscript parameters)
- Runs the full training pipeline
- Saves model checkpoint to `checkpoints/battery_gpt_NASA.pth`

---

### 🧪 Example: Evaluation

After training, evaluate the model:

```bash
bash run_pipeline.sh evaluate NASA
```

- Loads the trained model checkpoint
- Evaluates on NASA test dataset
- Prints metrics (MAE, RMSE, etc.)
- Saves detailed results to `results/NASA_eval_results.txt`

---

## 📑 Pipeline Details

### 🚂 Training Pipeline (`scripts/train.py`)

1. Data loading and preprocessing:
   - Interpolates missing values
   - Normalizes multi-modal inputs (voltage, current, temperature)

2. Model initialization:
   - Loads parameters from `configs/battery_gpt.yaml`

3. Training loop:
   - Uses Adam optimizer
   - Applies dynamic masking during generative pre-training
   - Regularly saves model checkpoints

---

### 📝 Evaluation Pipeline (`scripts/evaluate.py`)

1. Loads and preprocesses test dataset
2. Loads trained BatteryGPT model
3. Predicts SOH and RUL
4. Computes evaluation metrics (MAE, Relative Error)
5. Saves detailed predictions and metrics

---

## 📂 Directory Structure

```shell
BatteryGPT/
├── checkpoints/            # Saved model checkpoints
├── configs/                # Configuration files
│   └── battery_gpt.yaml
├── data/                   # Datasets (user-provided)
├── models/                 # Model definition code
├── results/                # Evaluation results
├── scripts/                # Core training and evaluation scripts
│   ├── train.py
│   └── evaluate.py
├── utils/                  # Utility functions (data processing, metric calculation)
├── run_pipeline.sh         # Unified training/evaluation script
├── requirements.txt
└── README.md
```

---

## 💡 Unified Pipeline Script Example

```bash
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
```

Make the script executable with:

```bash
chmod +x run_pipeline.sh
```

---

## 📞 Contact

For any questions or suggestions, feel free to open an issue or reach out via email!
