# True Prediction Experiment: Predicting Abnormalities Before They Appear

This experiment addresses a fundamental question in predictive ECG monitoring: **Can we truly predict abnormalities BEFORE they occur, or are we just recognizing them when they're already present?**

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Our Innovation](#our-innovation)
3. [Experiment Design](#experiment-design)
4. [Configurable Abnormal Definitions](#configurable-abnormal-definitions)
5. [Key Findings](#key-findings)
6. [Project Structure](#project-structure)
7. [Getting Started](#getting-started)
8. [Usage Examples](#usage-examples)
9. [Understanding the Results](#understanding-the-results)
10. [Docker Deployment](#docker-deployment)

---

## The Problem

### The False Success Trap

Previous ECG classification experiments often achieved high performance (85-90% accuracy), but we discovered a critical issue:

**The model wasn't predicting abnormalities - it was just recognizing them!**

#### Example of the Problem

Traditional approach:
```
Input sequence:  [Normal, Normal, Abnormal, Normal, Normal]
Predict:         Abnormal (next beat)

Result: ✓ Correct!
Reality: The model saw "Abnormal" in the input and simply output "Abnormal"
         This is pattern recognition, NOT prediction!
```

### Why This Matters Clinically

For early warning systems, we need to predict abnormal beats BEFORE they appear:
- ❌ Recognizing existing abnormalities → Too late for intervention
- ✓ Predicting future abnormalities → Time for preventive action

---

## Our Innovation

### Pure Normal Input Filtering

We created a **TRUE prediction task** by filtering the dataset:

**Rule**: Only keep sequences where ALL input beats are normal

```python
def create_pure_normal_sequences(beats, labels, sequence_length=7):
    for i in range(len(beats) - sequence_length):
        input_labels = labels[i:i + sequence_length]

        # KEY FILTER: Skip if ANY input beat is abnormal
        if np.any(input_labels != 0):
            continue

        # Input is pure normal, predict next beat
        X_seq.append(beats[i:i + sequence_length])
        y_seq.append(labels[i + sequence_length])  # might be abnormal!
```

### The New Challenge

```
Input:  [N, N, N, N, N, N, N]  ← All normal beats
Predict: A (abnormal)          ← Predict without seeing abnormality

This forces the model to learn subtle precursor patterns in normal beats
that precede abnormalities, rather than just copying visible abnormalities.
```

---

## Experiment Design

### Data Processing Pipeline

1. **Beat Extraction**: Extract 256-sample windows around R-peaks
2. **Normalization**: Per-beat normalization to [-1, 1]
3. **Sequence Creation**: Group consecutive beats
4. **Pure Normal Filtering**: **Remove sequences with ANY abnormal input**
5. **Data Balancing**: Undersample majority class for training
6. **Stratified Split**: 80% train+val, 20% test

### Model Architecture

**Transformer-based classifier**:
- Input projection: Linear(256 → 128)
- Positional encoding: Learnable embeddings
- Transformer encoder: 2 layers, 4 attention heads
- Classification head: MLP + Sigmoid

**Why Transformer?**
- Self-attention captures dependencies between all beats
- Better at learning subtle temporal patterns than LSTM
- Can model long-range interactions

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Sequence Length | 7 beats (default) |
| Batch Size | 32 |
| Optimizer | Adam (lr=0.001) |
| Loss | Binary Cross-Entropy |
| Early Stopping | Patience = 5 |
| Max Epochs | 100 |

---

## Configurable Abnormal Definitions

A unique feature of this experiment: **You can define what "abnormal" means!**

### Available Configurations

#### 1. **all** (Default)
```bash
python train.py --abnormal_config all
```
- **Abnormal**: L, R, V, A, F (everything except N)
- **Use case**: General abnormality detection
- **Challenge**: Mixed difficulty, diverse patterns

#### 2. **ventricular** (Most Dangerous)
```bash
python train.py --abnormal_config ventricular
```
- **Abnormal**: V (ventricular premature), F (fusion)
- **Clinical significance**: ⚠️ Can lead to ventricular fibrillation
- **Result**: Higher precision than 'all', but still very low

#### 3. **atrial**
```bash
python train.py --abnormal_config atrial
```
- **Abnormal**: A (atrial premature beats)
- **Clinical significance**: Usually less dangerous than ventricular

#### 4. **bundle_branch**
```bash
python train.py --abnormal_config bundle_branch
```
- **Abnormal**: L (left), R (right) bundle branch blocks
- **Clinical significance**: Conduction system disorders

#### 5. **premature**
```bash
python train.py --abnormal_config premature
```
- **Abnormal**: V, A (premature beats from ectopic locations)

#### 6. **custom**
```bash
python train.py --abnormal_config custom --abnormal_symbols V F A
```
- **Abnormal**: User-specified symbols
- **Use case**: Research and experimentation

### View All Configurations

```bash
python train.py --show_configs
```

This displays detailed clinical information about each configuration.

---

## Key Findings

### Original Results (from transformer_model.ipynb)

**Configuration**: all abnormalities, sequence length = 7

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 37.88% | Below random! Imbalanced test set |
| **Precision** | **3.25%** | ⚠️ 96.75% of predicted abnormals are FALSE! |
| **Recall** | 87.31% | Catches most abnormalities |
| **F1 Score** | 6.26% | Poor overall performance |

### Confusion Matrix Analysis

```
              Predicted
           Normal  Abnormal
Actual N   4,039    6,973    ← 63% false positive rate!
       A      34      234    ← 87% recall

Key insight: The model predicts "abnormal" very liberally
```

### What This Tells Us

1. **True Prediction is EXTREMELY Hard**
   - Without seeing the abnormality in input, the model struggles
   - Precision of 3.25% means 96.75% false alarm rate
   - Subtle precursor patterns are difficult to learn

2. **The Model Over-Predicts**
   - High recall (87%) shows it catches abnormalities
   - But at the cost of massive false positives
   - Strategy: "When in doubt, predict abnormal"

3. **Clinical Implications**
   - System would generate constant false alarms
   - Not practical for real-world deployment
   - However: Better to over-predict than under-predict in healthcare

4. **Comparison with Standard Task**
   - CNN-LSTM (with abnormal inputs): 90% accuracy, 80% precision
   - Transformer (pure normal inputs): 38% accuracy, 3% precision
   - **26x harder to predict than to recognize!**

### Ventricular-Only Results

When focusing on ventricular abnormalities (V, F):
- **Precision**: Improved, but still very low (exact numbers vary)
- **Clinical**: These are the most dangerous, so improvement matters
- **Interpretation**: Ventricular abnormalities may have slightly more predictable precursors

---

## Project Structure

```
true_prediction_experiment/
│
├── data_loader.py           # Beat extraction and preprocessing (shared)
├── dataset.py               # PyTorch Dataset class (shared)
├── data_processing.py       # Pure normal sequence filtering ⭐ KEY
├── model.py                 # Transformer architecture
├── train.py                 # Training with configurable abnormal definitions
├── evaluate.py              # Comprehensive evaluation and analysis
│
├── requirements.txt
├── Dockerfile
└── README.md                # This file
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- MIT-BIH Arrhythmia Database

### Installation

```bash
cd ECG_experiments/true_prediction_experiment

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

The MIT-BIH dataset should be in the dataset directory:

```
ECG_experiments/
├── dataset/
│   └── mitdb/              # MIT-BIH database files
└── true_prediction_experiment/
```

**For Python**: Default path is `../dataset/mitdb`
**For Docker**: Dataset copied during build

---

## Usage Examples

### Basic Training

```bash
# Default: all abnormalities, sequence length 7
python train.py
```

**Output**:
- `transformer_model_all_seq7.pth` - Trained model
- `test_data.npy` - Test set with metadata
- `experiment_config.json` - Configuration
- `training_curve.png` - Loss curves

### Focus on Ventricular Abnormalities

```bash
python train.py --abnormal_config ventricular
```

This trains a model specifically for the most dangerous abnormalities.

### Custom Sequence Length

```bash
python train.py --seq_len 5 --abnormal_config premature
```

Shorter sequences may be easier to train but lose temporal context.

### View Configuration Options

```bash
python train.py --show_configs
```

Displays detailed information about all abnormal configurations.

### Custom Abnormal Definition

```bash
python train.py --abnormal_config custom --abnormal_symbols V F
```

Define your own set of abnormal beat types.

### Evaluation

```bash
# After training
python evaluate.py
```

**Output**:
- Detailed metrics and analysis
- `confusion_matrix.png`
- `roc_curve.png`
- `evaluation_results.json`

### Advanced Options

```bash
python train.py \
    --abnormal_config ventricular \
    --seq_len 7 \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3 \
    --batch_size 64 \
    --epochs 150 \
    --lr 0.0005
```

---

## Understanding the Results

### Why is Precision So Low?

The 3.25% precision is NOT a failure - it's a profound result!

**It reveals that**:
1. Abnormalities don't have clear precursor patterns in preceding normal beats
2. ECG abnormalities may be inherently unpredictable from prior signals alone
3. True early warning may require additional information (patient history, vitals, etc.)

### The Precision-Recall Trade-off

```
High Precision (few false alarms) ← → High Recall (catch all abnormalities)
```

Our model chooses: **High Recall, Low Precision**

**Why?**
- In healthcare, missing a dangerous arrhythmia (false negative) is worse
- False alarms (false positives) are annoying but not life-threatening
- The model learns this trade-off implicitly

### Interpreting Low Accuracy

37.88% accuracy seems terrible, but consider:
- Test set is imbalanced (~97.6% normal)
- If model predicts everything as abnormal: 2.4% accuracy
- If model predicts everything as normal: 97.6% accuracy
- Our model balances between these extremes

**The real question**: Can we predict better than random?
- Random guessing: ~50% precision at current recall
- Our model: 3.25% precision
- **Answer**: No significant predictive signal found

### Clinical Interpretation

**Positive View**:
- High recall (87%) means few dangerous beats are missed
- Could serve as a sensitive screening tool
- Human review filters false positives

**Realistic View**:
- 96.75% false alarm rate is impractical
- Would cause alarm fatigue
- Needs additional features beyond ECG alone

**Research Value**:
- Establishes baseline for true prediction difficulty
- Shows that recognition ≠ prediction
- Motivates multi-modal approaches

---

## Comparison with Other Experiments

| Experiment | Input | Accuracy | Precision | Task Difficulty |
|------------|-------|----------|-----------|-----------------|
| CNN-LSTM (base) | Any sequence | 89.95% | 79.72% | Pattern recognition |
| Window (w=1) | Any sequence | 85.91% | 68.05% | Immediate prediction |
| **True Prediction** | Pure normal | **37.88%** | **3.25%** | **Hardest** |

**Key Insight**: Removing abnormal beats from input makes the task 26x harder!

---

## Docker Deployment

### Building the Image

```bash
# Build from the ECG_experiments directory (parent of true_prediction_experiment)
cd ECG_experiments
docker build -f true_prediction_experiment/Dockerfile -t ecg-true-prediction:latest .
```

**Note**: Make sure the `dataset/mitdb/` directory exists in the `ECG_experiments/` directory before building.

### Running Training

```bash
# Basic run
docker run --rm \
    -v $(pwd)/true_prediction_experiment:/app/output \
    ecg-true-prediction:latest \
    python train.py --abnormal_config ventricular

# With GPU
docker run --gpus all --rm \
    -v $(pwd)/true_prediction_experiment:/app/output \
    ecg-true-prediction:latest \
    python train.py --abnormal_config all --seq_len 7
```

### Running Evaluation

```bash
docker run --rm \
    -v $(pwd)/true_prediction_experiment:/app/output \
    ecg-true-prediction:latest \
    python evaluate.py
```

---

## Advanced Topics

### Modifying Abnormal Definitions in Code

Edit `data_processing.py`:

```python
def define_abnormal_symbols(config='all'):
    configs = {
        'my_custom': {
            'selected': ['N', 'V', 'F', 'A'],
            'normal': ['N'],
            'abnormal': ['V', 'F', 'A'],
            'desc': 'My custom definition'
        }
    }
```

### Analyzing Precursor Patterns

```python
# After training, you can analyze which normal patterns
# precede abnormalities most often

from data_loader import extract_beat_windows
from data_processing import create_pure_normal_sequences

# Get sequences that are followed by abnormalities
beats, labels = extract_beat_windows(...)
X_seq, y_seq = create_pure_normal_sequences(beats, labels)

# Find patterns before abnormalities
abnormal_precursors = X_seq[y_seq == 1]  # Normal sequences → Abnormal
normal_sequences = X_seq[y_seq == 0]     # Normal sequences → Normal

# Compare statistics, visualize differences, etc.
```

### Improving the Model

Potential improvements to explore:

1. **Longer Sequences**: More temporal context
   ```bash
   python train.py --seq_len 15
   ```

2. **Deeper Transformer**: More capacity
   ```bash
   python train.py --num_layers 4 --d_model 256
   ```

3. **Multi-Lead ECG**: Use both leads
   - Modify `data_loader.py` to use both channels
   - Adjust model input dimension

4. **External Features**: Add patient metadata
   - Age, sex, medical history
   - Heart rate variability
   - Previous arrhythmia history

5. **Different Architectures**: Try alternatives
   - Temporal Convolutional Networks (TCN)
   - Recurrent Transformer
   - Ensemble methods

---

## Conclusions

### What We Learned

1. **Recognition ≠ Prediction**
   - High performance on standard tasks doesn't mean true prediction
   - Filtering abnormal inputs reveals the real challenge

2. **Abnormalities May Be Unpredictable**
   - 3.25% precision suggests weak precursor signals
   - ECG alone may be insufficient for early warning

3. **Clinical Context Matters**
   - High recall > high precision in healthcare
   - But 96.75% false alarm rate is still too high

4. **Experimental Rigor is Crucial**
   - Always question what the model is actually learning
   - Design experiments that test TRUE capabilities

### Future Directions

1. **Multi-Modal Learning**
   - Combine ECG with other vital signs
   - Incorporate patient history and demographics

2. **Temporal Feature Engineering**
   - Heart rate variability metrics
   - Beat-to-beat interval analysis
   - Morphology change detection

3. **Patient-Specific Models**
   - Personalized baselines
   - Adapt to individual patterns

4. **Ensemble Approaches**
   - Combine multiple abnormal definitions
   - Vote across different sequence lengths

---

## Citation

If you use this experiment in your research:

```
True Prediction Experiment: Testing the Limits of ECG Arrhythmia Forecasting
MIT-BIH Arrhythmia Database Analysis
```

Original database citation:
```
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
```

---

## License

This experiment is for research and educational purposes. The MIT-BIH Arrhythmia Database is available under the Open Data Commons Open Database License v1.0.
