# QRS Feature Experiment v2 - Summary

## ✅ What Was Created

A complete reorganization of the feature extraction experiments into **2 clean SimpleFNN experiments** with proper data balancing strategies.

### 📁 New Directory Structure

```
feature_qrs_experiment_v2/
├── data_loader.py              # Data loading with clean 30s history requirement
├── model.py                    # SimpleFNN model definition
├── train_balanced.py           # Experiment 1: Fully balanced (all splits)
├── train_semi_balanced.py      # Experiment 2: Semi-balanced (train/val only)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container setup
├── .dockerignore               # Docker build exclusions
├── README.md                   # Comprehensive documentation
├── DOCKER_USAGE.md             # Docker usage instructions
└── SUMMARY.md                  # This file
```

---

## 🔬 Two Experiments

### Experiment 1: Fully Balanced Dataset

**Data Strategy**:
1. Extract abnormal (V, E, F, !) with clean 30-second history
2. Extract normal (N) with clean 30-second history
3. Split abnormal: Train (60%) / Val (20%) / Test (20%)
4. Sample equal normal for ALL splits
5. **Result**: All sets are balanced (50/50)

**Command**:
```bash
python train_balanced.py
```

**Use Case**: Model development and baseline comparison

---

### Experiment 2: Semi-Balanced Dataset (Recommended)

**Data Strategy**:
1. Extract abnormal (V, E, F, !) with clean 30-second history
2. Extract normal (N) with clean 30-second history
3. Split abnormal: Train (60%) / Val (20%) / Test (20%)
4. Sample equal normal for **train/val ONLY**
5. Test set: ALL abnormal test + ALL remaining normal
6. **Result**: Train/val balanced, test unbalanced (realistic ~5% abnormal)

**Command**:
```bash
python train_semi_balanced.py
```

**Use Case**: Realistic evaluation (clinical deployment scenario)

---

## 🧬 30-Second History Filtering (From Original feature_extraction.ipynb)

**Both experiments use this filtering logic**:

**For Normal (N) beats**:
1. Check previous 30 seconds
2. If **ANY abnormal** (V, E, F, !) found → **Skip**
3. If clean → Extract QRS features → Label as 0

**For Abnormal (V, E, F, !) beats**:
1. Check previous 30 seconds
2. If **SAME TYPE** abnormal found → **Skip**
3. If clean → Extract QRS features → Label as 1

**Rationale**:
- **Normal**: Ensures features from truly stable rhythm
- **Abnormal**: Focuses on first/isolated occurrences (not clustered beats)
- Improves data quality for classification

---

## 📊 QRS Features (4 Features)

From each clean 30-second window:
1. **Mean QRS Area**: Average depolarization strength
2. **Std QRS Area**: Beat-to-beat variability
3. **Mean R-peak Amplitude**: Signal strength
4. **Std R-peak Amplitude**: Amplitude variability

---

## 🏗️ Model: SimpleFNN

```
Input (4) → Hidden (16) → ReLU → Dropout(0.3)
         → Hidden (8)  → ReLU → Dropout(0.3)
         → Output (1)  → Sigmoid
```

**Why SimpleFNN?**:
- Only 4 input features (low-dimensional)
- Fast training
- Easy to interpret
- Good baseline

---

## 🚀 Quick Start

### Local Execution

```bash
cd ECG_experiments/feature_qrs_experiment_v2

# Install dependencies
pip install -r requirements.txt

# Run Experiment 1 (Balanced)
python train_balanced.py

# Run Experiment 2 (Semi-Balanced - Recommended)
python train_semi_balanced.py
```

### Docker Execution

```bash
# Build (from ECG_experiments/)
cd ECG_experiments
docker build -f feature_qrs_experiment_v2/Dockerfile -t qrs-feature-exp:latest .

# Run Experiment 1
docker run --rm \
    -v $(pwd)/feature_qrs_experiment_v2:/app/output \
    qrs-feature-exp:latest \
    python train_balanced.py

# Run Experiment 2
docker run --rm \
    -v $(pwd)/feature_qrs_experiment_v2:/app/output \
    qrs-feature-exp:latest \
    python train_semi_balanced.py
```

---

## 📈 Expected Results

### Experiment 1 (Balanced Test)

- **Accuracy**: 70-85%
- **Precision**: 70-85%
- **Recall**: 70-85%
- **Interpretation**: Direct performance comparison (baseline = 50%)

### Experiment 2 (Unbalanced Test - Realistic)

- **Accuracy**: 90-98% ⚠️ Misleading!
- **Precision**: 30-60% ← Focus here
- **Recall**: 60-85% ← Focus here
- **F1-Score**: 40-70% ← Balanced metric
- **Test Set**: ~5% abnormal (realistic)

**Important**: With unbalanced test sets, **ignore accuracy**. Focus on Precision, Recall, F1.

---

## 🔄 Differences from Original

### Old Structure (feature_qrs_experiment)
- Multiple complex experiments
- Mixed CNN, RNN, transformer models
- Unclear data balancing strategy
- Hard to understand and reproduce

### New Structure (feature_qrs_experiment_v2)
- **Only 2 experiments** (clear comparison)
- **Only SimpleFNN** (focused)
- **Clear balancing strategies** (documented)
- **Clean history requirement** (quality control)
- **Comprehensive README** (easy to use)

---

## 📝 Key Takeaways

1. **Clean History Matters**: 30-second requirement ensures quality
2. **Balancing Strategy Affects Results**: Semi-balanced is more realistic
3. **Metrics Matter**: Accuracy misleading with imbalanced data
4. **SimpleFNN Works**: Good baseline for 4-feature classification
5. **Clinical Relevance**: High recall > high precision in healthcare

---

## 🎯 Recommendations

**For Research**:
- Use Experiment 1 for model development
- Compare against 50% baseline
- Easy to interpret

**For Clinical Evaluation**:
- Use Experiment 2 for realistic assessment
- Focus on Precision/Recall trade-off
- Reflects real-world deployment

**For Both**:
- Always report all metrics (not just accuracy)
- Include confusion matrix
- Discuss false positive/negative implications
