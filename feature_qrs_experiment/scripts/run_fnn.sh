#!/usr/bin/env bash
set -euo pipefail
python -m ECG_experiments.feature_qrs_experiment.train --config configs/default.yaml --model simple_fnn "$@"
