# Docker Usage Instructions

## Building the Image

**Build from ECG_experiments/ directory**:

```bash
cd ECG_experiments
docker build -f feature_qrs_experiment_v2/Dockerfile -t qrs-feature-exp:latest .
```

## Running Experiments

### Experiment 1: Fully Balanced

```bash
docker run --rm \
    -v $(pwd)/feature_qrs_experiment_v2:/app/output \
    qrs-feature-exp:latest \
    python train_balanced.py
```

### Experiment 2: Semi-Balanced (Realistic)

```bash
docker run --rm \
    -v $(pwd)/feature_qrs_experiment_v2:/app/output \
    qrs-feature-exp:latest \
    python train_semi_balanced.py
```

### Custom Parameters

```bash
docker run --rm \
    -v $(pwd)/feature_qrs_experiment_v2:/app/output \
    qrs-feature-exp:latest \
    python train_semi_balanced.py \
        --hidden_sizes 32 16 \
        --dropout 0.4 \
        --lr 0.0005 \
        --epochs 150
```

### With GPU

```bash
docker run --gpus all --rm \
    -v $(pwd)/feature_qrs_experiment_v2:/app/output \
    qrs-feature-exp:latest \
    python train_semi_balanced.py
```

## Output Files

Results are saved to the mounted directory:
- `best_model_balanced.pth` - Trained model (Experiment 1)
- `best_model_semi_balanced.pth` - Trained model (Experiment 2)
- `results_balanced.json` - Metrics (Experiment 1)
- `results_semi_balanced.json` - Metrics (Experiment 2)

## Troubleshooting

### Build Error: "COPY failed"

Make sure you're in `ECG_experiments/` directory:
```bash
pwd  # Should show: .../MIT-BIH/ECG_experiments
docker build -f feature_qrs_experiment_v2/Dockerfile -t qrs-feature-exp:latest .
```

### Verify Database in Container

```bash
docker run --rm qrs-feature-exp:latest ls -la /app/mitdb | head -10
```

Should show .dat, .hea, .atr files.
