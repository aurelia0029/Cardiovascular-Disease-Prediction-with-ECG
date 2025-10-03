# Personalized VF Prediction Experiment - Summary

## Overview

This directory now contains a complete personalized VF prediction experiment based on the motivation you provided. The experiment tests whether individual patient models outperform generalized population models.

## Files Created

### Core Experiment Files

1. **`exp_personalized_models.py`** - Main experiment script
   - Loads SCDH data organized by patient
   - Creates 2/3 train, 1/3 test split per patient
   - Trains individual models for each patient
   - Evaluates and saves results

2. **`compare_personalized_vs_generalized.py`** - Comparison analysis
   - Loads personalized model results
   - Compares with generalized model results (when available)
   - Performs statistical testing
   - Generates comparison visualizations

3. **`personalized_models_analysis.ipynb`** - Interactive Jupyter notebook
   - Step-by-step analysis workflow
   - Visualization of results
   - Statistical summaries
   - Correlation analyses

4. **`run_personalized_experiment.sh`** - Automated runner script
   - Checks dependencies
   - Runs experiment end-to-end
   - Optional comparison analysis
   - Executable: `./run_personalized_experiment.sh`

### Documentation Files

5. **`PERSONALIZED_EXPERIMENT.md`** - Detailed documentation
   - Motivation and hypothesis
   - Methodology explanation
   - Usage instructions
   - Expected results and interpretation
   - Future extensions

6. **`EXPERIMENT_SUMMARY.md`** - This file
   - Quick reference for all files
   - Quick start guide

## Quick Start

### Option 1: Command Line (Recommended)

```bash
cd /home/aurelia0029/MIT-BIH/ECG_experiments/scdh_experiment
./run_personalized_experiment.sh
```

### Option 2: Python Script Directly

```bash
cd /home/aurelia0029/MIT-BIH/ECG_experiments/scdh_experiment
python exp_personalized_models.py
```

### Option 3: Jupyter Notebook (Interactive)

```bash
cd /home/aurelia0029/MIT-BIH/ECG_experiments/scdh_experiment
jupyter notebook personalized_models_analysis.ipynb
```

## Experiment Design

### Motivation
Previous experiments (`train_scdh_only.py`, `train_scdh_nsr.py`) built generalized models using data from multiple patients. This experiment tests whether personalized models trained on individual patient data can better predict that patient's future VF onset.

### Approach
- **Per-patient split**: 2/3 training, 1/3 testing
- **No data mixing**: Each patient has independent model
- **Same features**: Uses feature extraction from `feature_extraction.py`
- **Ensemble classifier**: RF + HGB + Logistic Regression

### Key Research Question
**Can personalized models trained on an individual patient's historical ECG data more accurately predict their future VF onset compared to generalized population-based models?**

## Expected Output

### Directory Structure After Running

```
personalized_models_output/
├── personalized_results_YYYYMMDD_HHMMSS.json
├── personalized_results_YYYYMMDD_HHMMSS.pkl
└── plots/
    ├── accuracy_by_patient.png
    └── roc_auc_by_patient.png

comparison_output/  (if comparison run)
├── comparison_boxplots.png
├── patient_accuracy_comparison.png
└── comparison_summary.csv
```

### Key Metrics Reported

For each patient:
- Accuracy
- ROC-AUC (if both classes present)
- Precision, Recall, F1-score (Normal and Onset)
- Confusion matrix
- Training/test set sizes

Aggregate statistics:
- Mean ± Std across patients
- Median, Min, Max
- Statistical comparison with generalized models (if available)

## Integration with Existing Experiments

This experiment complements your existing work:

| Experiment File | Approach | Data Source |
|----------------|----------|-------------|
| `train_scdh_only.py` | Generalized model | SCDH only |
| `train_scdh_nsr.py` | Generalized model | SCDH + NSR |
| **`exp_personalized_models.py`** | **Personalized models** | **SCDH per-patient** |

You can now compare all three approaches to determine which is most effective for VF prediction.

## Configuration

Edit `CONFIG` dictionary in `exp_personalized_models.py`:

```python
CONFIG = {
    'DATA_DIR': None,  # Auto-detect or specify sddb path
    'OUTPUT_DIR': './personalized_models_output',
    'TRAIN_RATIO': 2/3,
    'TEST_RATIO': 1/3,
    'RANDOM_SEED': 42,
    'SEGMENT_LEN': 3,  # seconds
    'WINDOW_LEN': 180,  # seconds (3 minutes)
    'TIME_BEFORE_ONSET': 20,  # minutes before/after VF
    'TARGET_FS': 250,  # Hz
    'SKIP_LIST': ['40', '42', '49'],  # Paced or no VF
    'MIN_SEGMENTS_PER_CLASS': 5  # Minimum for training
}
```

## Dependencies

All dependencies are already satisfied by your existing environment:
- ✓ `data_loader.py` (loads SCDH data)
- ✓ `feature_extraction.py` (extracts ECG features)
- ✓ Standard ML libraries (scikit-learn, pandas, numpy)
- ✓ SCDH database (`../../sddb/`)

## Expected Results

### Possible Outcomes

1. **Personalized > Generalized**: Individual patterns are highly predictive
   - Clinical implication: Patient-specific calibration needed
   - Trade-off: Requires historical data per patient

2. **Personalized ≈ Generalized**: Similar performance
   - Clinical implication: Population models sufficient
   - Advantage: Can predict for new patients

3. **Personalized < Generalized**: Worse performance
   - Reason: Insufficient data per patient
   - Solution: More data collection or transfer learning

## Next Steps After Running

1. **Review summary statistics** printed to console
2. **Examine plots** in `personalized_models_output/plots/`
3. **Compare with generalized models** using comparison script
4. **Analyze individual patients**: Which benefit most from personalization?
5. **Investigate failures**: Why did some patients fail training?
6. **Consider extensions**: Transfer learning, temporal validation, etc.

## Troubleshooting

### Common Issues

**Issue**: "SCDH database not found"
```bash
# Solution: Check database path
ls ../../sddb/  # Should show .dat, .hea, .ari files
```

**Issue**: "Insufficient training data"
```python
# Solution: Reduce minimum segments requirement
CONFIG['MIN_SEGMENTS_PER_CLASS'] = 3  # Instead of 5
```

**Issue**: "Only 1 class in training data"
- Check VF onset times in data_loader.py
- Verify TIME_BEFORE_ONSET setting
- Some patients may have very short recordings

## Related Files

### Original Experiments (for comparison)
- `/home/aurelia0029/MIT-BIH/exp3_1_by_1.py` - Similar approach but different data processing
- `/home/aurelia0029/MIT-BIH/exp3_all.py` - All patients pooled
- `/home/aurelia0029/MIT-BIH/exp3_LOPO.py` - Leave-one-patient-out

### Data Loaders
- `data_loader.py` - SCDH and NSR data loading
- `feature_extraction.py` - ECG feature extraction

### Training Scripts
- `train_scdh_only.py` - Generalized SCDH-only model
- `train_scdh_nsr.py` - Generalized SCDH+NSR model

## Citation/Reference

If this experiment produces useful results, remember to document:
- Which approach performed best
- Statistical significance of differences
- Clinical implications
- Computational costs (training time, memory)

## Contact

For questions about this experiment, refer to:
- `PERSONALIZED_EXPERIMENT.md` - Detailed methodology
- `/home/aurelia0029/MIT-BIH/CLAUDE.md` - Project overview
- `/home/aurelia0029/MIT-BIH/ECG_experiments/scdh_experiment/README.md` - General SCDH experiments

---

**Experiment Created**: 2025-10-03
**Based on**: exp3_1_by_1.py methodology
**Motivation**: Test personalized vs generalized VF prediction models
**Status**: Ready to run
