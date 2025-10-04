# ECG Experiments - Path Standards

This document describes the standardized directory structure and paths for all experiments in the ECG_experiments directory.

## Directory Structure

```
ECG_experiments/
├── dataset/
│   ├── mitdb/              # MIT-BIH Arrhythmia Database
│   ├── sddb/               # Sudden Cardiac Death Holter Database
│   └── nsrdb/              # Normal Sinus Rhythm Database
│
├── cnn_lstm_experiment/
├── window_prediction_experiment/
├── true_prediction_experiment/
├── scdh_experiment/
└── feature_qrs_experiment/
```

## Dataset Locations

All datasets are now centralized in the `dataset/` directory:

- **MIT-BIH Arrhythmia Database**: `ECG_experiments/dataset/mitdb/`
- **SCDH Database**: `ECG_experiments/dataset/sddb/`
- **NSR Database**: `ECG_experiments/dataset/nsrdb/`

## Python Path Standards

When running Python scripts locally from an experiment directory, use:

```python
# For MIT-BIH database
data_dir = '../dataset/mitdb'

# For SCDH database
scdh_dir = '../dataset/sddb'

# For NSR database
nsr_dir = '../dataset/nsrdb'
```

## Docker Path Standards

### Build Context

**All Dockerfiles must be built from the `ECG_experiments/` directory (parent directory):**

```bash
cd ECG_experiments
docker build -f <experiment_name>/Dockerfile -t <image_name>:latest .
```

### COPY Commands in Dockerfile

From the `ECG_experiments/` build context:

```dockerfile
# Copy experiment files
COPY <experiment_name>/requirements.txt .
COPY <experiment_name>/*.py .

# Copy datasets
COPY dataset/mitdb /app/mitdb
COPY dataset/sddb /app/sddb   # If needed
COPY dataset/nsrdb /app/nsrdb # If needed
```

### Running Containers

Mount the experiment directory as output:

```bash
docker run --rm \
    -v $(pwd)/<experiment_name>:/app/output \
    <image_name>:latest \
    python script.py
```

## Experiment-Specific Details

### 1. CNN-LSTM Experiment

**Dataset**: MIT-BIH (`dataset/mitdb/`)

**Build**:
```bash
cd ECG_experiments
docker build -f cnn_lstm_experiment/Dockerfile -t ecg-cnn-lstm:latest .
```

**Run**:
```bash
docker run --rm -v $(pwd)/cnn_lstm_experiment:/app/output ecg-cnn-lstm:latest python train.py
```

---

### 2. Window Prediction Experiment

**Dataset**: MIT-BIH (`dataset/mitdb/`)

**Build**:
```bash
cd ECG_experiments
docker build -f window_prediction_experiment/Dockerfile -t ecg-window-prediction:latest .
```

**Run**:
```bash
docker run --rm -v $(pwd)/window_prediction_experiment:/app/output ecg-window-prediction:latest python window_experiment.py
```

---

### 3. True Prediction Experiment

**Dataset**: MIT-BIH (`dataset/mitdb/`)

**Build**:
```bash
cd ECG_experiments
docker build -f true_prediction_experiment/Dockerfile -t ecg-true-prediction:latest .
```

**Run**:
```bash
docker run --rm -v $(pwd)/true_prediction_experiment:/app/output ecg-true-prediction:latest python train.py
```

---

### 4. SCDH Experiment

**Datasets**: SCDH (`dataset/sddb/`) and NSR (`dataset/nsrdb/`)

**Build**:
```bash
cd ECG_experiments
docker build -f scdh_experiment/Dockerfile -t scdh-experiment:latest .
```

**Run**:
```bash
# Experiment 1: SCDH only
docker run --rm -v $(pwd)/scdh_experiment:/app/output scdh-experiment:latest python train_scdh_only.py --min 20

# Experiment 2: SCDH + NSR
docker run --rm -v $(pwd)/scdh_experiment:/app/output scdh-experiment:latest python train_scdh_nsr.py --min 20
```

**Special Note**: Auto-detection checks multiple paths for backward compatibility:
1. `../dataset/sddb` and `../dataset/nsrdb` (new standard)
2. `../sddb` and `../nsrdb` (old location)
3. `/app/sddb` and `/app/nsrdb` (Docker)

---

### 5. Feature QRS Experiment

**Dataset**: MIT-BIH (`dataset/mitdb/`)

**Build**:
```bash
cd ECG_experiments
docker build -f feature_qrs_experiment/Dockerfile -t ecg-feature-qrs:latest .
```

**Run**:
```bash
# Experiment 1: Fully balanced
docker run --rm -v $(pwd)/feature_qrs_experiment:/app/output ecg-feature-qrs:latest python train_balanced.py

# Experiment 2: Semi-balanced
docker run --rm -v $(pwd)/feature_qrs_experiment:/app/output ecg-feature-qrs:latest python train_semi_balanced.py
```

---

## Migration Guide

If you have datasets in the old location (`MIT-BIH/mitdb`, `MIT-BIH/sddb`, etc.), move them:

```bash
cd MIT-BIH/ECG_experiments

# Create dataset directory
mkdir -p dataset

# Move datasets
mv ../mitdb dataset/
mv ../sddb dataset/
mv ../nsrdb dataset/
```

Or create symbolic links:

```bash
cd MIT-BIH/ECG_experiments
mkdir -p dataset
ln -s ../../mitdb dataset/mitdb
ln -s ../../sddb dataset/sddb
ln -s ../../nsrdb dataset/nsrdb
```

## Troubleshooting

### Error: No such file or directory

**For Python scripts**:
```
Error: No such file or directory: '../dataset/mitdb'
```
**Solution**: Ensure you're running from the experiment directory and `dataset/mitdb` exists in parent

**For Docker**:
```
Error: COPY failed: file not found in build context
```
**Solution**: Ensure you're building from `ECG_experiments/` directory, not from the experiment subdirectory

### Checking Dataset Paths

```bash
# From experiment directory
ls ../dataset/mitdb/  # Should show .dat, .hea, .atr files
ls ../dataset/sddb/   # Should show SCDH files
ls ../dataset/nsrdb/  # Should show NSR files
```

---

## Summary

✅ **Always build Docker images from `ECG_experiments/` directory**
✅ **All datasets are in `ECG_experiments/dataset/` subdirectories**
✅ **Python scripts use `../dataset/{db_name}` as default**
✅ **Docker containers mount experiment dir to `/app/output`**
✅ **SCDH experiment has backward-compatible auto-detection**
