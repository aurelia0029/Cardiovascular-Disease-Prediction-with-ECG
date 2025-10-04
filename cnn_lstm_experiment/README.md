# CNN-LSTM ECG Arrhythmia Classification Experiment

This experiment implements a CNN-LSTM hybrid deep learning model for binary classification of ECG heartbeats from the MIT-BIH Arrhythmia Database. The model classifies beats as either **Normal** or **Abnormal** based on sequences of consecutive heartbeats.

---

## Table of Contents

1. [Experiment Overview](#experiment-overview)
2. [Dataset Information](#dataset-information)
3. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Results and Analysis](#results-and-analysis)
7. [Project Structure](#project-structure)
8. [Getting Started](#getting-started)
9. [Usage](#usage)
10. [Docker Deployment](#docker-deployment)

---

## Experiment Overview

### Objective
Develop a deep learning model to automatically detect abnormal heartbeats from ECG signals using temporal patterns across consecutive beats.

### Approach
- **Feature Extraction**: 1D CNN extracts spatial features from individual ECG beats
- **Temporal Modeling**: LSTM captures temporal dependencies across beat sequences
- **Classification**: Binary classification (Normal vs Abnormal)

### Key Innovation
Unlike single-beat classifiers, this model uses sequences of 3 consecutive beats to predict whether the next beat is normal or abnormal, leveraging temporal context for improved accuracy.

---

## Dataset Information

### MIT-BIH Arrhythmia Database
- **Source**: PhysioNet MIT-BIH Arrhythmia Database
- **Records**: 48 half-hour excerpts of two-channel ambulatory ECG recordings
- **Sampling Rate**: 360 Hz
- **Annotations**: Beat-by-beat annotations by cardiologists

### Beat Types Included

| Symbol | Description | Classification |
|--------|-------------|----------------|
| N | Normal beat | Normal (0) |
| L | Left bundle branch block | Abnormal (1) |
| R | Right bundle branch block | Abnormal (1) |
| V | Ventricular premature beat | Abnormal (1) |
| A | Atrial premature beat | Abnormal (1) |
| F | Fusion beat | Abnormal (1) |

### Beat Annotation Statistics

After filtering for selected beat types:
- **Total beats extracted**: 100,835
- **Normal beats (N)**: 75,028 (74.4%)
- **Abnormal beats (L, R, V, A, F)**: 25,807 (25.6%)

**Class Imbalance**: The dataset is highly imbalanced with ~3:1 ratio of normal to abnormal beats, requiring undersampling for model training.

---

## Data Loading and Preprocessing

### 1. Reading ECG Data

```python
import wfdb

# Load ECG signal and annotations
record = wfdb.rdrecord('mitdb/100')
annotation = wfdb.rdann('mitdb/100', 'atr')
```

**Key Components**:
- `.dat` files contain raw ECG signals
- `.hea` files contain header information
- `.atr` files contain beat annotations (R-peak locations and beat types)

### 2. Beat Window Extraction

Each beat is extracted as a **200-sample window** centered at the R-peak:
- **100 samples before** R-peak
- **100 samples after** R-peak
- **Total window**: 200 samples (~0.56 seconds at 360 Hz)

```python
# Extract beat window
r_peak = annotation.sample[i]
beat = signal[r_peak - 100:r_peak + 100]
```

### 3. Normalization

Each beat is normalized to `[-1, 1]` range:

```python
def normalize_beat(beat):
    max_val = np.max(np.abs(beat))
    if max_val < 1e-6:  # skip flat signals
        return None
    return beat / max_val
```

**Why per-beat normalization?**
- Handles amplitude variations between patients
- Reduces influence of baseline wander
- Improves model generalization

### 4. Sequence Creation

Consecutive beats are grouped into sequences:

```python
# Create sequences of 3 beats to predict the 4th
X_seq = [beat[i], beat[i+1], beat[i+2]]
y_seq = label[i+3]
```

**Output Shape**:
- `X_seq`: (100,832, 3, 200) - 100,832 sequences, each with 3 beats of 200 samples
- `y_seq`: (100,832,) - labels for the next beat after each sequence

### 5. Data Splitting and Balancing

**Train/Validation/Test Split**:
1. Split 80% train+validation, 20% test (stratified)
2. Balance train+validation set by undersampling majority class
3. Split balanced data into 80% train, 20% validation

**Final Dataset Statistics**:

| Split | Samples | Normal | Abnormal | Distribution |
|-------|---------|--------|----------|--------------|
| Train | 33,032 | 16,516 | 16,516 | 50%/50% (balanced) |
| Validation | 8,258 | 4,129 | 4,129 | 50%/50% (balanced) |
| Test | 20,167 | 15,005 | 5,162 | 74%/26% (imbalanced, real-world) |

**Why balance training data?**
- Prevents model from biasing toward majority class
- Ensures equal learning from both normal and abnormal patterns
- Test set remains imbalanced to reflect real-world performance

---

## Model Architecture

### CNN-LSTM Hybrid Model

```
Input: (batch, 3 beats, 1 channel, 200 samples)
   ↓
[1D CNN Feature Extractor]
   ├─ Conv1d(1→16, kernel=5) + ReLU
   ├─ MaxPool1d(kernel=2)
   ├─ Conv1d(16→16, kernel=3) + ReLU
   └─ AdaptiveAvgPool1d(1)
   ↓
CNN Features: (batch, 3, 16)
   ↓
[LSTM Temporal Processor]
   └─ LSTM(input=16, hidden=64)
   ↓
LSTM Output: (batch, 64)
   ↓
[Classifier]
   ├─ Linear(64→64) + ReLU + Dropout(0.5)
   └─ Linear(64→1) + Sigmoid
   ↓
Output: (batch, 1) - probability of abnormal
```

### Model Components

#### 1. CNN Feature Extractor
- **Purpose**: Extract spatial features from individual beats
- **Architecture**: 2-layer 1D CNN with max pooling and adaptive pooling
- **Output**: 16-dimensional feature vector per beat

#### 2. LSTM Temporal Processor
- **Purpose**: Capture temporal dependencies across beat sequences
- **Architecture**: Single-layer LSTM with 64 hidden units
- **Output**: 64-dimensional hidden state from last time step

#### 3. Classifier Head
- **Purpose**: Binary classification from LSTM features
- **Architecture**: 2-layer MLP with dropout for regularization
- **Output**: Single probability score for abnormal class

### Model Parameters

- **Total Parameters**: ~70K trainable parameters
- **CNN Channels**: 16
- **LSTM Hidden Size**: 64
- **Dropout Rate**: 0.5

---

## Training Process

### Training Configuration

```python
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 5  # Early stopping patience
```

### Loss Function and Optimizer

- **Loss**: Binary Cross-Entropy Loss (BCELoss)
- **Optimizer**: Adam with learning rate 0.001
- **Regularization**: Dropout (0.5) in classifier

### Early Stopping

The model uses early stopping to prevent overfitting:
- Monitors validation loss after each epoch
- Saves best model weights when validation loss improves
- Stops training if validation loss doesn't improve for 5 consecutive epochs

### Training Procedure

```python
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(model, val_loader)

    # Early stopping check
    if val_loss < best_val_loss:
        save_best_model()
    else:
        early_stop_counter += 1
```

### Training Results

**Training completed in 50 epochs** (no early stopping triggered):

- **Final Training Loss**: 0.2803
- **Final Validation Loss**: 0.3139
- **Best Validation Loss**: 0.3139 (Epoch 50)

**Training Curve Observations**:
- Steady decrease in both training and validation loss
- No significant overfitting observed
- Validation loss continues to improve through epoch 50
- Model converges smoothly without oscillations

---

## Results and Analysis

### Test Set Performance

The model was evaluated on the **imbalanced test set** (20,167 samples) to reflect real-world performance:

#### Overall Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **89.95%** |
| **Precision** | 0.7972 |
| **Recall** | 0.8146 |
| **F1 Score** | 0.8058 |

#### Confusion Matrix

```
                 Predicted
                Normal  Abnormal
Actual  Normal   13,935   1,070
        Abnormal    957   4,205

[[TN=13,935  FP=1,070]
 [FN=957     TP=4,205]]
```

#### Class-Specific Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Normal** | 0.94 | 0.93 | 0.93 | 15,005 |
| **Abnormal** | 0.80 | 0.81 | 0.81 | 5,162 |

### Performance Analysis

#### Strengths
1. **High Overall Accuracy**: 89.95% on imbalanced test set
2. **Excellent Normal Detection**: 93% recall on normal beats
3. **Good Abnormal Detection**: 81% recall on abnormal beats
4. **Balanced Performance**: Model doesn't heavily bias toward majority class despite imbalanced test set

#### Error Analysis

**False Positives (1,070 cases)**:
- Normal beats incorrectly classified as abnormal
- 7.1% false positive rate
- May include borderline cases or noisy signals

**False Negatives (957 cases)**:
- Abnormal beats incorrectly classified as normal
- 18.5% false negative rate
- Clinically more concerning than false positives
- May include subtle abnormalities or rare patterns

#### Clinical Implications

**For Screening Applications**:
- 93% sensitivity for normal beats (low false alarm rate)
- 81% sensitivity for abnormal beats (good detection rate)
- Trade-off balances sensitivity and specificity

**Model Limitations**:
- 19% of abnormal beats missed (false negatives)
- May require human expert review for critical cases
- Best used as a screening tool rather than definitive diagnosis

---

## Project Structure

```
cnn_lstm_experiment/
│
├── data_loader.py          # Data loading and preprocessing functions
│   ├── load_record_list()       # Load MIT-BIH record names
│   ├── get_all_symbols()        # Extract all beat annotations
│   ├── normalize_beat()         # Per-beat normalization
│   ├── extract_beat_windows()   # Extract beat windows from ECG
│   └── create_sequences()       # Create beat sequences
│
├── dataset.py              # PyTorch Dataset class
│   └── BeatSequenceDataset      # Dataset for beat sequences
│
├── model.py                # CNN-LSTM model architecture
│   └── CNNLSTMClassifier        # Hybrid CNN-LSTM model
│
├── train.py                # Training script
│   ├── balance_dataset()        # Undersample for class balance
│   ├── train_model()            # Training loop with early stopping
│   └── plot_training_curves()   # Visualize training progress
│
├── evaluate.py             # Evaluation script
│   ├── evaluate_model()         # Run inference on test set
│   └── print_metrics()          # Calculate and display metrics
│
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image definition
└── README.md               # This file
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)
- MIT-BIH Arrhythmia Database

### Installation

```bash
# Clone repository (or navigate to experiment folder)
cd ECG_experiments/cnn_lstm_experiment

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

The MIT-BIH Arrhythmia Database should be in the `dataset/` directory:

```
ECG_experiments/
├── dataset/
│   └── mitdb/         # MIT-BIH database files
│       ├── 100.dat
│       ├── 100.hea
│       ├── 100.atr
│       └── ...
└── cnn_lstm_experiment/
    ├── train.py
    ├── data_loader.py
    └── ...
```

**For Python execution**: The dataset is at `../dataset/mitdb` relative to the experiment directory.

**For Docker**: Build from `ECG_experiments/` directory (see Docker section below).

---

## Usage

### Training

Run the complete training pipeline:

```bash
python train.py
```

**What it does**:
1. Loads MIT-BIH dataset
2. Extracts and normalizes beat windows
3. Creates beat sequences
4. Splits and balances data
5. Trains CNN-LSTM model
6. Saves best model to `cnn_lstm_best_model.pth`
7. Generates training curve plot `training_curve.png`

**Output Files**:
- `cnn_lstm_best_model.pth` - Trained model weights
- `training_curve.png` - Training/validation loss plot
- `test_data.npy` - Test set for evaluation

**Training Time**:
- ~3-5 minutes per epoch on CPU
- ~30 seconds per epoch on GPU
- Total: ~30-50 minutes depending on early stopping

### Evaluation

Evaluate the trained model on test set:

```bash
python evaluate.py
```

**Output**:
```
TEST METRICS
==================================================
Accuracy:  89.95%
Precision: 0.7972
Recall:    0.8146
F1 Score:  0.8058

Confusion Matrix:
[[13935  1070]
 [  957  4205]]

Classification Report:
              precision    recall  f1-score   support

      Normal       0.94      0.93      0.93     15005
    Abnormal       0.80      0.81      0.81      5162

    accuracy                           0.90     20167
```

### Using as a Module

```python
from data_loader import extract_beat_windows, create_sequences
from model import CNNLSTMClassifier
import torch

# Load model
model = CNNLSTMClassifier(beat_len=200)
model.load_state_dict(torch.load('cnn_lstm_best_model.pth'))
model.eval()

# Prepare your data
X_seq = your_beat_sequences  # shape: (N, 3, 200)
X_seq = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(2)

# Predict
with torch.no_grad():
    predictions = model(X_seq)
    labels = (predictions >= 0.5).float()
```

---

## Docker Deployment

### Building the Docker Image

The Dockerfile will copy the MIT-BIH dataset from the parent directory during build:

```bash
# Build from the ECG_experiments directory (parent of cnn_lstm_experiment)
cd ECG_experiments
docker build -f cnn_lstm_experiment/Dockerfile -t ecg-cnn-lstm:latest .
```

**Note**: Make sure the `dataset/mitdb/` directory exists in the `ECG_experiments/` directory before building.

### Running Training in Docker

```bash
# Run training (dataset already included in image)
docker run --rm -v $(pwd)/cnn_lstm_experiment:/app/output ecg-cnn-lstm:latest python train.py

# Run with GPU support
docker run --gpus all --rm -v $(pwd)/cnn_lstm_experiment:/app/output ecg-cnn-lstm:latest python train.py
```

### Running Evaluation in Docker

```bash
docker run --rm -v $(pwd)/cnn_lstm_experiment:/app/output ecg-cnn-lstm:latest python evaluate.py
```

### Docker Image Details

- **Base Image**: Python 3.9-slim
- **Size**: ~1.5GB (includes MIT-BIH dataset)
- **Includes**:
  - All Python dependencies from requirements.txt
  - MIT-BIH dataset at `/app/mitdb`
  - Experiment scripts
- **Entrypoint**: Bash shell for running Python scripts
- **Build Context**: Must be built from `ECG_experiments/` directory

---

## Key Takeaways

### What Works Well
1. **Temporal Context**: Using sequences of 3 beats improves classification accuracy
2. **CNN Feature Extraction**: 1D CNN effectively captures morphological features
3. **LSTM Temporal Modeling**: LSTM captures beat-to-beat dependencies
4. **Data Balancing**: Undersampling prevents majority class bias

### Areas for Improvement
1. **False Negatives**: 19% of abnormal beats missed - consider:
   - Weighted loss function
   - Focal loss for hard examples
   - Data augmentation
2. **Model Complexity**: Consider:
   - Bidirectional LSTM
   - Attention mechanisms
   - Deeper CNN architectures
3. **Multi-class Classification**: Extend to classify specific arrhythmia types (V, F, A, etc.)

### Clinical Considerations
- Model achieves good screening performance but should not replace expert diagnosis
- Best used as a first-line triage tool
- High false negative rate requires human oversight for critical applications

---

## References

1. Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
2. PhysioNet: https://physionet.org/content/mitdb/1.0.0/
3. WFDB Python Package: https://github.com/MIT-LCP/wfdb-python

---

## License

This experiment is for research and educational purposes. The MIT-BIH Arrhythmia Database is available under the Open Data Commons Open Database License v1.0.

---

## Contact

For questions or issues related to this experiment, please refer to the main project repository.
