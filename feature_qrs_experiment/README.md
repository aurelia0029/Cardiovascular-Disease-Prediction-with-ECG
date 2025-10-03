# Feature QRS Experiment

This experiment refactors the original `feature_extraction.ipynb` workflow into a reusable package. It focuses on
extracting QRS-based features from MIT-BIH style ECG records, assembling balanced datasets, and training baseline
models ranging from classical ML to a small neural network.

## Layout

- `configs/`: YAML configuration files describing data sources, window sizes, and sampling strategies.
- `data_processing/`: Utilities for loading WFDB signals, selecting ventricular events, and computing QRS features.
- `datasets/`: Scripts that orchestrate feature extraction and dataset splitting.
- `models/`: Baseline models that consume the extracted feature vectors.
- `scripts/`: Convenience shell entrypoints for running the training flows end-to-end.
- `train.py`: Command-line interface that ties loaders, datasets, and models together.
- `evaluate.py`: Shared evaluation routines for reporting metrics.
- `artifacts/`: Target directory for generated assets (saved models, metrics exports).

## Getting Started

1. Activate the project virtual environment (e.g. `source MIT_env/bin/activate`) or another Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify that the raw WFDB files are available relative to the project root (default: `./mitdb`, `./nsrdb`, `./sddb`).
4. Run a classical baseline:
   ```bash
   python -m ECG_experiments.feature_qrs_experiment.train --config configs/default.yaml --model logistic_regression
   ```
5. Train the neural baseline (requires PyTorch):
   ```bash
   python -m ECG_experiments.feature_qrs_experiment.train --config configs/default.yaml --model simple_fnn
   ```

Generated artifacts land in `artifacts/` (metrics JSON, confusion matrices, saved models).

## Notes

- The code defaults to MIT-BIH records; update the config to point at CUDB/PAFDB directories once the files are ready.
- Feature extraction currently implements four QRS statistics (mean/std of areas and amplitudes). Extend
  `feature_extractor.py` to add HRV or other feature families.
- Sampling strategies mirror the notebook: balanced sampling and a PCA-based cleaning + oversampling pipeline.
