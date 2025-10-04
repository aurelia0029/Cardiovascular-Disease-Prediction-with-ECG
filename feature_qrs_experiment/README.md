# QRS Feature Extraction Experiments

This experiment uses **QRS-based features** from MIT-BIH Arrhythmia Database to classify abnormal heartbeats using a Simple Feedforward Neural Network (SimpleFNN).

## Key Feature: 30-Second History Filtering

This experiment uses careful filtering based on the previous 30 seconds:

**For Normal (N) beats**:
- Check previous 30 seconds
- If **ANY abnormal** (V, E, F, !) found → **Skip**
- If clean → Extract QRS features → Label as Normal (0)

**For Abnormal (V, E, F, !) beats**:
- Check previous 30 seconds
- If **SAME TYPE** abnormal found → **Skip** (e.g., if current is V, skip if V in history)
- If clean → Extract QRS features → Label as Abnormal (1)

**Why Different Rules?**:
- **Normal**: Ensures truly normal rhythm (no contamination from any abnormality)
- **Abnormal**: Avoids clustered abnormalities (e.g., consecutive V beats)

---

## Table of Contents

1. [Overview](#overview)
2. [Data Collection Strategy](#data-collection-strategy)
3. [Two Experiments](#two-experiments)
4. [QRS Features](#qrs-features)
5. [Model Architecture](#model-architecture)
6. [Getting Started](#getting-started)
7. [Usage Examples](#usage-examples)
8. [Understanding Results](#understanding-results)
9. [Project Structure](#project-structure)

---

## Overview

### Abnormal Beat Types

The experiments focus on ventricular abnormalities:
- **V**: Ventricular premature beat
- **E**: Ventricular escape beat
- **F**: Fusion beat
- **!**: Ventricular flutter wave

### Binary Classification
- **Abnormal (label = 1)**: V, E, F, !
- **Normal (label = 0)**: N

---

## Data Collection Strategy

### 30-Second History Filtering

**For Normal Beats (N)**:
1. Find normal beat at time T
2. Check previous 30 seconds (T-30s to T)
3. If **ANY abnormal** beat (V, E, F, !) in history → **Skip**
4. If clean → Extract QRS features from window → Label as 0

**For Abnormal Beats (V, E, F, !)**:
1. Find abnormal beat at time T (e.g., V)
2. Check previous 30 seconds (T-30s to T)
3. If **SAME TYPE** abnormal in history (e.g., another V) → **Skip**
4. If clean → Extract QRS features from window → Label as 1

**Rationale**:
- **Normal filtering**: Ensures QRS features come from stable, normal rhythm
- **Abnormal filtering**: Avoids clustered/consecutive abnormalities (which may have different characteristics)
- Focuses on **first occurrence** or **isolated** abnormal beats after normal periods

---

## Two Experiments

### Experiment 1: Fully Balanced Dataset

**Strategy**:
1. Split abnormal samples: Train (60%) / Val (20%) / Test (20%)
2. Sample **equal number** of normal samples for each split
3. All splits are **balanced** (50% normal, 50% abnormal)

**Characteristics**:
- ✓ Easy to interpret (baseline = 50% accuracy)
- ✓ Focuses on model's ability to distinguish patterns
- ✗ Not realistic (real data is highly imbalanced)

**Run**:
```bash
python train_balanced.py
```

**Expected Results**:
- Accuracy: 70-85%
- Precision: 70-85%
- Recall: 70-85%

---

### Experiment 2: Semi-Balanced Dataset

**Strategy**:
1. Split abnormal samples: Train (60%) / Val (20%) / Test (20%)
2. Sample equal normal for train/val (balanced)
3. **Test set**: Use ALL abnormal test samples + ALL remaining normal samples
4. Test set is **unbalanced** (realistic ~1-5% abnormal)

**Characteristics**:
- ✓ Trains on balanced data (prevents bias)
- ✓ Tests on realistic imbalanced data
- ✓ Reflects real-world deployment scenario

**Run**:
```bash
python train_semi_balanced.py
```

**Expected Results**:
- Accuracy: 90-98% (misleading due to imbalance)
- **Precision**: 30-60% (key metric)
- **Recall**: 60-85% (key metric)
- **F1-Score**: 40-70% (balanced metric)

**Important**: With imbalanced test sets, **accuracy is misleading**. A model predicting all normal achieves 95-99% accuracy but is useless. Focus on **Precision, Recall, and F1-Score**.

---

## QRS Features

Each sample uses **4 QRS-based features** extracted from the clean 30-second history:

### 1. Mean QRS Area
- Average area under QRS complex
- Reflects ventricular depolarization strength

### 2. Std QRS Area
- Standard deviation of QRS areas
- Measures beat-to-beat variability

### 3. Mean R-peak Amplitude
- Average height of R-peaks
- Indicates signal strength

### 4. Std R-peak Amplitude
- Standard deviation of R-peak heights
- Measures amplitude variability

**QRS Complex Window**:
- Width: ±18 samples @ 360Hz = ±50ms
- Captures main ventricular depolarization

**Why QRS Features?**:
- ✓ Clinically interpretable
- ✓ Computationally efficient
- ✓ Robust to noise
- ✓ Established in cardiology

---

## Model Architecture

### SimpleFNN (Simple Feedforward Neural Network)

```
Input Layer:    4 features
                ↓
Hidden Layer 1: 16 neurons + ReLU + Dropout(0.3)
                ↓
Hidden Layer 2: 8 neurons + ReLU + Dropout(0.3)
                ↓
Output Layer:   1 neuron + Sigmoid
                ↓
Prediction:     P(abnormal)
```

**Architecture Details**:
- Input: 4 QRS features
- Hidden: [16, 8] (configurable)
- Activation: ReLU
- Regularization: Dropout (0.3)
- Output: Sigmoid (binary classification)
- Loss: Binary Cross-Entropy

**Why SimpleFNN?**:
- Low-dimensional features (only 4)
- Fast training
- Easy to interpret
- Baseline for more complex models

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- MIT-BIH Arrhythmia Database

### Installation

```bash
cd ECG_experiments/feature_qrs_experiment_v2

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

The MIT-BIH database should be in the dataset directory:

```
ECG_experiments/
├── dataset/
│   └── mitdb/                  # MIT-BIH database
└── feature_qrs_experiment/
    ├── data_loader.py
    ├── model.py
    ├── train_balanced.py
    └── train_semi_balanced.py
```

Path auto-detection finds `../dataset/mitdb` automatically.

---

## Usage Examples

### Experiment 1: Fully Balanced

```bash
# Basic run
python train_balanced.py

# Custom hyperparameters
python train_balanced.py \
    --hidden_sizes 32 16 \
    --dropout 0.5 \
    --lr 0.0005 \
    --batch_size 64 \
    --epochs 150
```

**Output**:
- `best_model_balanced.pth`: Trained model
- `results_balanced.json`: Metrics and configuration

---

### Experiment 2: Semi-Balanced (Recommended)

```bash
# Basic run
python train_semi_balanced.py

# Custom hyperparameters
python train_semi_balanced.py \
    --hidden_sizes 32 16 \
    --dropout 0.4 \
    --lr 0.001 \
    --batch_size 32 \
    --patience 15
```

**Output**:
- `best_model_semi_balanced.pth`: Trained model
- `results_semi_balanced.json`: Metrics and configuration

---

## Understanding Results

### Experiment 1: Balanced Test Set

**Example Output**:
```
Test Accuracy: 0.7842
Precision: 0.7654
Recall:    0.8123
F1-Score:  0.7881

Confusion Matrix:
[[450  80]   ← Normal:  450 correct, 80 false positives
 [ 95 405]]  ← Abnormal: 405 correct, 95 false negatives
```

**Interpretation**:
- Accuracy ~78%: Model correctly classifies 78% of samples
- Precision ~77%: Of predicted abnormals, 77% are truly abnormal
- Recall ~81%: Catches 81% of actual abnormals
- Balanced dataset makes metrics directly comparable

---

### Experiment 2: Unbalanced Test Set

**Example Output**:
```
Test Accuracy: 0.9523
Precision: 0.4321
Recall:    0.7891
F1-Score:  0.5567

Test set (UNBALANCED): 5000 samples
  - Abnormal: 250 (5%)
  - Normal:   4750 (95%)

Confusion Matrix:
[[4500 250]   ← Normal:  4500 correct, 250 false positives
 [  53 197]]  ← Abnormal: 197 correct, 53 false negatives
```

**Interpretation**:

**Why High Accuracy is Misleading**:
- 95% accuracy sounds great!
- But baseline (predict all normal) = 95% accuracy
- Model must beat this baseline

**Why Precision/Recall Matter**:
- **Precision 43%**: Of 447 predicted abnormals, only 197 are truly abnormal
  - 250 false alarms (patients unnecessarily worried)
- **Recall 79%**: Catches 197 out of 250 true abnormals
  - 53 missed (potentially dangerous)

**Trade-off**:
- High recall is critical in healthcare (don't miss abnormalities)
- But too many false alarms cause alarm fatigue
- F1-Score balances both

**Clinical Decision**:
- Recall 79% + Precision 43% might be acceptable as a **screening tool**
- Positive predictions trigger human review (cardiologist checks ECG)
- False alarms are manageable; missed abnormals are dangerous

---

## Comparison: Balanced vs Semi-Balanced

| Aspect | Experiment 1 (Balanced) | Experiment 2 (Semi-Balanced) |
|--------|------------------------|------------------------------|
| **Train Set** | Balanced (50/50) | Balanced (50/50) |
| **Val Set** | Balanced (50/50) | Balanced (50/50) |
| **Test Set** | Balanced (50/50) | **Unbalanced** (~5% abnormal) |
| **Accuracy** | 70-85% | 90-98% (misleading) |
| **Precision** | 70-85% | 30-60% |
| **Recall** | 70-85% | 60-85% |
| **F1-Score** | 70-85% | 40-70% |
| **Use Case** | Model development | Realistic evaluation |

**Recommendation**: Use **Experiment 2** for realistic performance assessment.

---

## Project Structure

```
feature_qrs_experiment_v2/
│
├── data_loader.py          # Data loading with clean history requirement
├── model.py                # SimpleFNN model definition
├── train_balanced.py       # Experiment 1: Fully balanced
├── train_semi_balanced.py  # Experiment 2: Semi-balanced (realistic)
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Advanced Topics

### Adjusting Clean History Window

Modify in `data_loader.py`:
```python
# Change from 30 seconds to 60 seconds
abnormal_features, abnormal_labels, normal_features = load_features_with_clean_history(
    data_dir='../dataset/mitdb',
    pre_window_sec=60,  # ← Longer history requirement
    abnormal_symbols=['V', 'E', 'F', '!']
)
```

**Trade-off**:
- Longer window → Stricter requirement → Fewer samples → More challenging
- Shorter window → More samples → Easier task

---

### Modifying Abnormal Symbols

Focus on specific abnormalities:
```python
# Only ventricular premature (V)
abnormal_features, abnormal_labels, normal_features = load_features_with_clean_history(
    data_dir='../dataset/mitdb',
    pre_window_sec=30,
    abnormal_symbols=['V']  # ← Only V
)
```

---

### Changing Model Complexity

```bash
# Larger model
python train_semi_balanced.py --hidden_sizes 64 32 16 --dropout 0.4

# Smaller model
python train_semi_balanced.py --hidden_sizes 8 --dropout 0.2
```

---

## Docker Deployment

### Building the Image

```bash
# Build from the ECG_experiments directory (parent of feature_qrs_experiment)
cd ECG_experiments
docker build -f feature_qrs_experiment/Dockerfile -t ecg-feature-qrs:latest .
```

**Note**: Make sure the `dataset/mitdb/` directory exists in the `ECG_experiments/` directory before building.

### Running Experiment 1 (Fully Balanced)

```bash
docker run --rm \
    -v $(pwd)/feature_qrs_experiment:/app/output \
    ecg-feature-qrs:latest \
    python train_balanced.py
```

### Running Experiment 2 (Semi-Balanced)

```bash
docker run --rm \
    -v $(pwd)/feature_qrs_experiment:/app/output \
    ecg-feature-qrs:latest \
    python train_semi_balanced.py
```

### With Custom Parameters

```bash
docker run --rm \
    -v $(pwd)/feature_qrs_experiment:/app/output \
    ecg-feature-qrs:latest \
    python train_semi_balanced.py --hidden_sizes 64 32 16 --epochs 200 --dropout 0.4
```

---

## Troubleshooting

### Low Recall (Missing Too Many Abnormals)

**Problem**: Recall < 60% (dangerous in healthcare)

**Solutions**:
1. Increase model capacity: `--hidden_sizes 32 16 8`
2. Reduce dropout: `--dropout 0.2`
3. Lower learning rate for better convergence: `--lr 0.0005`
4. Train longer: `--epochs 200 --patience 20`

---

### Low Precision (Too Many False Alarms)

**Problem**: Precision < 30% (alarm fatigue)

**Solutions**:
1. Add regularization: `--dropout 0.5`
2. Increase training data quality (stricter history requirement)
3. Use weighted loss to penalize false positives
4. Ensemble multiple models

---

### Insufficient Data

**Problem**: "Abnormal samples: 0" or very few samples

**Solutions**:
1. Reduce history requirement: `pre_window_sec=15` (in code)
2. Include more abnormal symbols: `['V', 'E', 'F', '!', '/']`
3. Check database path is correct

---

## Citation

If you use this experiment in your research:

```
QRS Feature Extraction Experiments with Clean History Requirement
MIT-BIH Arrhythmia Database Analysis
```

Database citation:
```
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
```

---

## License

This experiment is for research and educational purposes. The MIT-BIH Arrhythmia Database is available under the Open Data Commons Open Database License v1.0.
