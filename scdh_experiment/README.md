# SCDH Experiments: Predicting Ventricular Fibrillation Onset

This experiment evaluates the ability to predict ventricular fibrillation (VF) onset using ECG data from the **Sudden Cardiac Death Holter Database (SCDH)** and **Normal Sinus Rhythm Database (NSR)**.

---

## Table of Contents

1. [Overview](#overview)
2. [The Challenge](#the-challenge)
3. [Two Experiments](#two-experiments)
4. [Feature Extraction](#feature-extraction)
5. [Dataset Details](#dataset-details)
6. [Getting Started](#getting-started)
7. [Usage Examples](#usage-examples)
8. [Configuring Time Windows](#configuring-time-windows)
9. [Understanding Results](#understanding-results)
10. [Project Structure](#project-structure)
11. [Docker Deployment](#docker-deployment)

---

## Overview

Sudden cardiac death (SCD) is a major cause of mortality worldwide, often preceded by ventricular fibrillation (VF). This experiment investigates whether we can **predict VF onset** by analyzing ECG patterns **minutes before** the event occurs.

### Key Innovation

Unlike traditional classification tasks, this experiment:
- Extracts ECG data from a **configurable time window** BEFORE VF onset
- Uses **comprehensive feature extraction** (statistical, nonlinear, frequency domain)
- Employs **ensemble machine learning** methods for robust prediction
- Validates using **Stratified 10-Fold Cross-Validation**

---

## The Challenge

### What Makes VF Prediction Difficult?

**Ventricular Fibrillation (VF)** is a life-threatening arrhythmia where the heart's ventricles quiver instead of pumping blood effectively. Key challenges:

1. **Sudden Onset**: VF can occur with minimal warning
2. **Complex Patterns**: Precursor signals are subtle and vary between patients
3. **Imbalanced Data**: Normal ECG far outnumbers pre-VF patterns
4. **Time Sensitivity**: Useful prediction requires sufficient lead time for intervention

### Clinical Significance

Successful prediction could enable:
- **Early Warning Systems**: Alert medical staff before VF occurs
- **Preventive Interventions**: Administer treatment before life-threatening event
- **Risk Stratification**: Identify high-risk patients for closer monitoring

---

## Two Experiments

### Experiment 1: SCDH Only

**Dataset**: Only the Sudden Cardiac Death Holter Database

**Data Composition**:
- **Onset segments** (label = 1): ECG from `{min}` minutes BEFORE VF onset
- **Normal segments** (label = 0): ECG from `{min}` minutes AFTER VF ends

**Purpose**: Test if VF onset can be distinguished from post-VF normal rhythm using the same patient population.

**Run Command**:
```bash
python train_scdh_only.py --min 20
```

---

### Experiment 2: SCDH + NSR

**Dataset**: SCDH + Normal Sinus Rhythm Database

**Data Composition**:
- **Onset segments** (label = 1): SCDH ECG from `{min}` minutes BEFORE VF onset
- **Normal segments** (label = 0): NSR database (healthy individuals)

**Purpose**: Test if VF onset patterns can be distinguished from truly normal ECG from healthy individuals.

**Run Command**:
```bash
python train_scdh_nsr.py --min 20
```

---

## Feature Extraction

### Comprehensive Feature Set (27 Features)

Each 3-second ECG segment is transformed into 27 features across four categories:

#### 1. Statistical Features (9)
- `mean`: Average amplitude
- `std`: Standard deviation
- `var`: Variance
- `rms`: Root mean square
- `skewness`: Distribution asymmetry
- `kurtosis`: Distribution tail heaviness
- `min`, `max`, `median`: Signal bounds and center

#### 2. Nonlinear Features (7)
- `higuchi_fd`: Higuchi fractal dimension (signal complexity)
- `petrosian_fd`: Petrosian fractal dimension
- `sample_entropy`: Regularity measure
- `hurst`: Hurst exponent (long-range dependence)
- `lempel_ziv`: Lempel-Ziv complexity
- `app_entropy`: Approximate entropy
- `dfa`: Detrended fluctuation analysis (scaling behavior)

#### 3. Frequency Domain Features (6)
- `peak_freq`: Dominant frequency
- `total_power`: Total spectral power
- `mean_power`: Average spectral power
- `power_0_5hz`: Low frequency band power (0-5 Hz)
- `power_5_15hz`: Mid frequency band power (5-15 Hz)
- `power_15_40hz`: High frequency band power (15-40 Hz)

#### 4. Signal Variation Features (4)
- `max_diff`: Maximum first-order difference
- `sum_abs_diff`: Total variation
- `signal_range`: Peak-to-peak amplitude
- `zero_crossing`: Number of zero crossings

### Why These Features?

- **Statistical**: Capture basic signal characteristics
- **Nonlinear**: Detect chaotic/complex patterns unique to VF precursors
- **Frequency**: Identify spectral changes associated with arrhythmias
- **Variation**: Measure signal instability and irregularity

---

## Dataset Details

### SCDH (Sudden Cardiac Death Holter Database)

**Source**: PhysioNet (https://physionet.org/content/sddb/)

**Description**:
- 23 Holter recordings from patients who experienced sudden cardiac death
- Each contains VF events with annotated onset times
- Sampling rate: 250 Hz (most records)

**Records Used**: 22 out of 23 (excluding paced/problematic records: 40, 42, 49)

**VF Onset Times** (hardcoded in `data_loader.py`):
```python
{
    '30': '18:31:53',
    '31': '19:20:45',
    '32': '00:09:32',
    # ... (22 total)
}
```

**Data Extraction Strategy**:
- **Onset window**: Extract 3-minute segment `{min}` minutes BEFORE VF onset
- **Normal window**: Extract 3-minute segment `{min}` minutes AFTER VF ends
- Each 3-minute window split into 60 segments (3 seconds each @ 250 Hz = 750 samples)

---

### NSR (Normal Sinus Rhythm Database)

**Source**: PhysioNet (https://physionet.org/content/nsrdb/)

**Description**:
- 18 long-term ECG recordings from healthy individuals
- No significant arrhythmias
- Sampling rate: 128 Hz (resampled to 250 Hz for consistency)

**Data Extraction Strategy**:
- Extract 60 evenly-spaced 3-minute windows per record
- Each window split into 60 segments (3 seconds each)

---

## Getting Started

### Prerequisites

- Python 3.8+
- SCDH database (`sddb/` directory)
- NSR database (`nsrdb/` directory)

### Installation

```bash
cd ECG_experiments/scdh_experiment

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

The databases should be in the project root:

```
MIT-BIH/
├── sddb/               # SCDH database files
├── nsrdb/              # NSR database files
└── ECG_experiments/
    └── scdh_experiment/
```

**Default paths**:
- SCDH: `../../sddb`
- NSR: `../../nsrdb`

You can override these with `--scdh_dir` and `--nsr_dir` arguments.

---

## Usage Examples

### Experiment 1: SCDH Only

#### Basic Run (Default: 20 minutes before/after onset)
```bash
python train_scdh_only.py
```

#### Custom Time Window (10 minutes)
```bash
python train_scdh_only.py --min 10
```

**This extracts**:
- Onset: 10 minutes BEFORE VF onset
- Normal: 10 minutes AFTER VF ends

#### Custom Time Window (30 minutes)
```bash
python train_scdh_only.py --min 30
```

#### Full Parameter Example
```bash
python train_scdh_only.py \
    --min 20 \
    --segment_len 3 \
    --window_len 180 \
    --target_fs 250 \
    --n_splits 10 \
    --output scdh_20min_results.json
```

**Output**:
- Console: 10-fold CV results for each fold + averages
- `scdh_only_results.json`: Complete results in JSON format

---

### Experiment 2: SCDH + NSR

#### Basic Run (Default: 20 minutes before onset)
```bash
python train_scdh_nsr.py
```

#### Custom Time Window (15 minutes)
```bash
python train_scdh_nsr.py --min 15
```

**This extracts**:
- SCDH onset: 15 minutes BEFORE VF onset
- NSR normal: Random windows from healthy individuals

#### Adjust NSR Windows Per Record
```bash
python train_scdh_nsr.py --min 20 --nsr_windows 100
```

Increase NSR data by extracting more windows per record.

#### Full Parameter Example
```bash
python train_scdh_nsr.py \
    --min 20 \
    --nsr_windows 60 \
    --segment_len 3 \
    --window_len 180 \
    --target_fs 250 \
    --n_splits 10 \
    --output scdh_nsr_20min_results.json
```

**Output**:
- Console: 10-fold CV results for each fold + averages
- `scdh_nsr_results.json`: Complete results in JSON format

---

## Configuring Time Windows

### The `--min` Parameter

**Key Innovation**: Adjustable time window for extracting data relative to VF onset.

**Syntax**:
```bash
--min <minutes>
```

**Examples**:

| Value | Onset Extraction | Clinical Interpretation |
|-------|------------------|-------------------------|
| `--min 5` | 5 min before VF | Very close to event, strong precursor signals expected |
| `--min 10` | 10 min before VF | Moderate lead time, balance of signal strength and prediction utility |
| `--min 20` | 20 min before VF (default) | Longer lead time, weaker signals but more actionable |
| `--min 30` | 30 min before VF | Earliest prediction, most challenging but most useful clinically |

**Trade-off**:
- **Shorter `--min`**: Easier prediction (stronger signals), less clinical utility
- **Longer `--min`**: Harder prediction (weaker signals), more clinical utility

**Recommendation**: Try multiple values to understand prediction-utility trade-off:
```bash
for min_val in 5 10 15 20 25 30; do
    python train_scdh_only.py --min $min_val --output results_${min_val}min.json
done
```

---

## Understanding Results

### Interpreting 10-Fold CV Output

Example output:
```
==================================================
10-FOLD CROSS-VALIDATION RESULTS
==================================================

Fold  1/10:
  Accuracy:  0.8542
  Precision: 0.8563
  Recall:    0.8542
  F1-Score:  0.8541
  Confusion Matrix:
[[1234   156]
 [ 187  1103]]

... (folds 2-10)

==================================================
AVERAGE RESULTS ACROSS 10 FOLDS
==================================================
Accuracy:  0.8521 ± 0.0123
Precision: 0.8534 ± 0.0145
Recall:    0.8521 ± 0.0123
F1-Score:  0.8519 ± 0.0129
==================================================
```

### Key Metrics

- **Accuracy**: Overall correctness (onset + normal)
- **Precision**: Of predicted onsets, how many are true onsets?
  - **High precision**: Few false alarms
  - **Low precision**: Many false alarms (predicting onset when it's actually normal)

- **Recall**: Of actual onsets, how many did we catch?
  - **High recall**: Catch most VF onsets (critical for safety)
  - **Low recall**: Miss many VF onsets (dangerous!)

- **F1-Score**: Harmonic mean of precision and recall (balanced metric)

### Clinical Interpretation

**For VF Prediction**:
- **Recall is critical**: Missing a VF onset (false negative) can be fatal
- **Precision matters for practicality**: Too many false alarms cause alarm fatigue
- **Balance is key**: High recall with acceptable precision

**Example**:
- **Recall = 0.95, Precision = 0.70**
  - Catches 95% of VF onsets (excellent!)
  - But 30% of alarms are false (manageable in ICU setting)
  - **Verdict**: Acceptable for clinical deployment

- **Recall = 0.60, Precision = 0.95**
  - Misses 40% of VF onsets (unacceptable!)
  - Very few false alarms
  - **Verdict**: Too dangerous, not useful

---

## Project Structure

```
scdh_experiment/
│
├── feature_extraction.py   # Feature extraction functions (27 features)
├── data_loader.py           # SCDH and NSR data loading
├── train_scdh_only.py       # Experiment 1: SCDH only (onset vs normal)
├── train_scdh_nsr.py        # Experiment 2: SCDH onset vs NSR normal
│
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker container definition
├── .dockerignore            # Docker build exclusions
└── README.md                # This file
```

### Key Files

- **feature_extraction.py**: All 27 feature extraction functions (statistical, nonlinear, frequency, variation)
- **data_loader.py**:
  - `load_scdh_data()`: Extract onset/normal from SCDH with configurable `time_before_onset`
  - `load_nsr_data()`: Extract normal segments from NSR
  - `prepare_dataframe()`: Convert to pandas DataFrame

- **train_scdh_only.py**: Experiment 1 with configurable `--min` parameter
- **train_scdh_nsr.py**: Experiment 2 with configurable `--min` parameter

---

## Docker Deployment

### Building the Image

```bash
cd ECG_experiments/scdh_experiment
docker build -t scdh-experiment:latest .
```

### Running Experiments

#### Experiment 1 (SCDH Only)
```bash
# Basic run with default parameters
docker run --rm \
    -v $(pwd):/app/output \
    scdh-experiment:latest \
    python train_scdh_only.py

# Custom time window (15 minutes)
docker run --rm \
    -v $(pwd):/app/output \
    scdh-experiment:latest \
    python train_scdh_only.py --min 15
```

#### Experiment 2 (SCDH + NSR)
```bash
# Basic run
docker run --rm \
    -v $(pwd):/app/output \
    scdh-experiment:latest \
    python train_scdh_nsr.py

# Custom time window (25 minutes)
docker run --rm \
    -v $(pwd):/app/output \
    scdh-experiment:latest \
    python train_scdh_nsr.py --min 25
```

### Docker Parameters

- `-v $(pwd):/app/output`: Mount current directory to save results
- `--rm`: Automatically remove container after completion
- Add `--gpus all` if GPU acceleration needed (not required for this experiment)

---

## Advanced Topics

### Comparing Different Time Windows

Create a script to test multiple time windows:

```bash
#!/bin/bash
# test_time_windows.sh

for min_val in 5 10 15 20 25 30; do
    echo "Testing with --min $min_val"

    # Experiment 1
    python train_scdh_only.py \
        --min $min_val \
        --output scdh_only_${min_val}min.json

    # Experiment 2
    python train_scdh_nsr.py \
        --min $min_val \
        --output scdh_nsr_${min_val}min.json
done
```

Then analyze results:
```python
import json
import pandas as pd

results = []
for min_val in [5, 10, 15, 20, 25, 30]:
    with open(f'scdh_only_{min_val}min.json') as f:
        data = json.load(f)
        avg = data['results']['average_results']
        results.append({
            'min': min_val,
            'accuracy': avg['accuracy_mean'],
            'precision': avg['precision_mean'],
            'recall': avg['recall_mean'],
            'f1': avg['f1_mean']
        })

df = pd.DataFrame(results)
print(df)
```

### Feature Importance Analysis

After running experiments, analyze which features are most predictive:

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Load data (example)
df = pd.read_csv('your_data.csv')
X = df.drop(columns=['label'])
y = df['label']

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(importances.head(10))
```

### Expected Results Range

Based on similar VF prediction studies:

**Experiment 1 (SCDH Only)**:
- Expected Accuracy: 70-85%
- Expected Precision: 65-80%
- Expected Recall: 70-90%

**Experiment 2 (SCDH + NSR)**:
- Expected Accuracy: 75-90%
- Expected Precision: 70-85%
- Expected Recall: 75-95%

**Note**: Experiment 2 typically performs better because NSR provides clearer normal patterns.

---

## Troubleshooting

### Common Issues

**1. Missing Database Files**
```
Error: No such file or directory: '../../sddb'
```
**Solution**: Ensure SCDH database is in correct location or use `--scdh_dir` to specify path.

**2. Memory Error**
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce `--nsr_windows` or process records individually.

**3. Feature Extraction Warnings**
```
Warning: Sample entropy calculation failed
```
**Solution**: This is normal for some segments; invalid features are filtered out automatically.

---

## Citation

If you use this experiment in your research:

```
SCDH Experiments: Predicting Ventricular Fibrillation Onset
Using Sudden Cardiac Death Holter Database and Normal Sinus Rhythm Database
```

Database citations:
```
Greenwald SD. Development and Analysis of a Ventricular Fibrillation Detector.
M.S. thesis, MIT Dept. of Electrical Engineering and Computer Science, 1986.

Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet:
Components of a New Research Resource for Complex Physiologic Signals.
Circulation 101(23):e215-e220, 2000.
```

---

## License

This experiment is for research and educational purposes. The SCDH and NSR databases are available under the Open Data Commons Open Database License v1.0.

---

## Contact

For questions or issues, please refer to the main project documentation or open an issue in the repository.
