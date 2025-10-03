# Personalized Patient VF Prediction Models

## Motivation

Previous experiments (`train_scdh_only.py`, `train_scdh_nsr.py`) built **generalized models** using data from multiple patients. While these models can capture population-level patterns, they may miss individual-specific physiological characteristics.

This experiment investigates: **Can personalized models trained on an individual patient's historical data better predict their future VF onset compared to generalized models?**

## Hypothesis

Individual patients may have unique ECG patterns and physiological signatures that precede VF onset. A model trained specifically on one patient's past data might better recognize their personal pre-VF patterns than a generalized model.

## Methodology

### Data Split Strategy

For **EACH patient individually**:
- **Training set**: 2/3 of the patient's segments (both normal and onset)
- **Test set**: 1/3 of the patient's segments (both normal and onset)
- **No data mixing**: Each patient has a completely independent model

### Model Architecture

Ensemble classifier combining:
- **Random Forest** (100 trees)
- **Histogram Gradient Boosting**
- **Logistic Regression** (with standard scaling)

Voting method: Soft voting (probability averaging)

### Feature Extraction

Same features as generalized experiments:
- Heart rate variability (HRV) metrics
- QRS morphology features
- Temporal statistics
- Frequency domain features

(Extracted via `feature_extraction.py`)

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **ROC-AUC**: Area under ROC curve (for patients with both classes)
- **Precision/Recall/F1**: Per-class performance metrics
- **Confusion Matrix**: Detailed classification breakdown

## Usage

### Basic Execution

```bash
cd ECG_experiments/scdh_experiment
python exp_personalized_models.py
```

### Configuration

Edit the `CONFIG` dictionary in `exp_personalized_models.py`:

```python
CONFIG = {
    'DATA_DIR': None,  # Auto-detect or specify path to sddb/
    'OUTPUT_DIR': './personalized_models_output',
    'TRAIN_RATIO': 2/3,
    'TEST_RATIO': 1/3,
    'RANDOM_SEED': 42,
    'SEGMENT_LEN': 3,  # seconds
    'WINDOW_LEN': 180,  # seconds (3 minutes)
    'TIME_BEFORE_ONSET': 20,  # minutes
    'TARGET_FS': 250,  # Hz
    'SKIP_LIST': ['40', '42', '49'],  # Paced or no VF
    'MIN_SEGMENTS_PER_CLASS': 5
}
```

### Output Files

The experiment generates:

```
personalized_models_output/
├── personalized_results_YYYYMMDD_HHMMSS.json  # Summary metrics
├── personalized_results_YYYYMMDD_HHMMSS.pkl   # Full results (predictions, probabilities)
└── plots/
    ├── accuracy_by_patient.png                # Accuracy bar chart
    └── roc_auc_by_patient.png                 # ROC-AUC bar chart
```

## Expected Results

### Success Criteria

A patient model is considered successful if:
- ✓ Has ≥5 segments per class (normal and onset) in training set
- ✓ Model training completes without errors
- ✓ Evaluation produces valid metrics

### Potential Outcomes

1. **Better Performance**: Personalized models outperform generalized models
   - Suggests individual-specific patterns are highly predictive
   - May indicate need for patient-specific calibration in clinical deployment

2. **Similar Performance**: Personalized and generalized models perform comparably
   - Suggests population-level patterns are sufficient
   - Generalized models may be more practical

3. **Worse Performance**: Personalized models underperform
   - May indicate insufficient training data per patient
   - Generalized models benefit from larger, more diverse training sets

## Comparison to Other Experiments

| Experiment | Training Data | Model Scope | Patient Mixing |
|------------|--------------|-------------|----------------|
| `train_scdh_only.py` | All SCDH patients | Single generalized model | Yes |
| `train_scdh_nsr.py` | SCDH + NSR data | Single generalized model | Yes |
| **`exp_personalized_models.py`** | Individual patient | One model per patient | **No** |

## Limitations

1. **Small sample size per patient**: Some patients may have insufficient data
2. **No temporal validation**: Train/test split is random, not time-based
3. **Class imbalance**: Onset segments are typically much fewer than normal
4. **Limited generalization**: Models cannot predict for new patients

## Future Extensions

### 1. Temporal Split
Instead of random split, use chronological split:
- Train: First 2/3 of recording
- Test: Last 1/3 of recording

This better simulates real-world prediction scenario.

### 2. Transfer Learning
- Pre-train on all patients (generalized model)
- Fine-tune on individual patient data
- Compare to pure personalized models

### 3. Hybrid Models
- Combine generalized and personalized predictions
- Weighted voting based on patient-specific confidence

### 4. Feature Importance Analysis
- Analyze which features are most important per patient
- Identify common vs. individual-specific predictors

## Running with Docker

If using the Docker environment:

```bash
docker build -t scdh-personalized .
docker run -v $(pwd):/app scdh-personalized python exp_personalized_models.py
```

## Troubleshooting

### Issue: "Insufficient data" errors
**Solution**: Reduce `MIN_SEGMENTS_PER_CLASS` in config or increase `WINDOW_LEN`

### Issue: "Only 1 class in training data"
**Solution**: Check that VF onset times are correct and segments are being extracted properly

### Issue: Memory errors
**Solution**: Process patients sequentially, reduce ensemble size, or increase system RAM

## Contact

For questions about this experiment, refer to the main project documentation or CLAUDE.md.

---

**Experiment designed to test**: Whether individual physiological patterns are more predictive than population patterns for VF prediction.
