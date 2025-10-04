# Quick Start: Experiment 3 (Personalized Patient Models)

## What is Experiment 3?

**Experiment 3** tests whether **personalized models** (one per patient) can outperform **generalized models** (one for all patients) in predicting VF onset.

### Motivation

Previous experiments (Exp 1 and Exp 2) built generalized models from multiple patients. But each patient may have unique physiological patterns. This experiment investigates whether individual-specific models are more predictive.

## Quick Run

```bash
cd /home/aurelia0029/MIT-BIH/ECG_experiments/scdh_experiment

# Run personalized model experiment
python exp_personalized_models.py
```

## What It Does

For **EACH patient**:
1. Takes 2/3 of their data → Training set
2. Takes 1/3 of their data → Test set
3. Trains a patient-specific model
4. Evaluates on that patient's test data

**No data mixing between patients** - completely independent models.

## Output

```
personalized_models_output/
├── personalized_results_*.json    # Summary metrics
├── personalized_results_*.pkl     # Full results
└── plots/
    ├── accuracy_by_patient.png
    └── roc_auc_by_patient.png
```

## Compare with Generalized Models

```bash
# After running Exp 1, 2, and 3
python compare_personalized_vs_generalized.py
```

Generates statistical comparison and visualizations.

## Interactive Analysis

```bash
jupyter notebook personalized_models_analysis.ipynb
```

Step-by-step analysis with visualizations.

## Key Differences from Exp 1 and Exp 2

| Feature | Exp 1 & 2 | Exp 3 |
|---------|-----------|-------|
| Model type | Generalized | Personalized |
| Training data | All patients pooled | One patient at a time |
| Test data | 10-fold CV across patients | 1/3 of same patient's data |
| Result | Single performance metric | Per-patient performance |

## Expected Results

Three possible outcomes:

1. **Personalized > Generalized**: Individual patterns are highly predictive
   - Implication: Need patient-specific calibration in clinical systems

2. **Personalized ≈ Generalized**: Similar performance
   - Implication: Population patterns are sufficient

3. **Personalized < Generalized**: Worse performance
   - Cause: Insufficient data per patient
   - Solution: More data or transfer learning

## For More Details

See the main README.md for complete documentation of all three experiments.
