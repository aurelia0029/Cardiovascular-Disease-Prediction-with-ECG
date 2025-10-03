from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pickle

try:
    import joblib
except ImportError:
    joblib = None

try:
    import torch
except ImportError:
    torch = None

import numpy as np
import yaml

from .datasets.build_features import build_raw_feature_sets
from .datasets.split_strategies import prepare_datasets
from .evaluate import evaluate_model
from .models import MODEL_REGISTRY
from .utils.logging import get_logger


logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QRS feature-based models")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--model", default="logistic_regression", choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--seed", type=int, default=None, help="Override seed defined in config")
    parser.add_argument("--skip-cache", action="store_true", help="Do not write the feature cache")
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    return config


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - depends on hardware
            torch.cuda.manual_seed_all(seed)


def _save_model(model: Any, artifact_dir: Path, model_name: str) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "state_dict") and torch is not None:
        path = artifact_dir / f"{model_name}.pt"
        torch.save(model.state_dict(), path)
    else:
        extension = "joblib" if joblib is not None else "pkl"
        path = artifact_dir / f"{model_name}.{extension}"
        if joblib is not None:
            joblib.dump(model, path)
        else:
            with path.open("wb") as fp:
                pickle.dump(model, fp)
    return path


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    config = _load_config(config_path)

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    config["seed"] = seed

    _set_random_seeds(seed)

    logger.info("Using config %s", config_path)
    logger.info("Selected model: %s", args.model)

    raw_datasets = build_raw_feature_sets(config, cache=not args.skip_cache)
    splits = prepare_datasets(raw_datasets, config, seed)

    trainer = MODEL_REGISTRY[args.model]
    model = trainer(splits, config["models"].get(args.model, {}), seed)

    artifact_dir = Path(config["paths"]["artifacts_dir"]).resolve()
    metrics = evaluate_model(model, splits, args.model, artifact_dir)

    model_path = _save_model(model, artifact_dir, args.model)

    summary_path = artifact_dir / f"{args.model}_summary.json"
    summary = {
        "config": str(config_path),
        "seed": seed,
        "model": args.model,
        "model_path": str(model_path),
        "metrics_path": str(artifact_dir / f"{args.model}_metrics.json"),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved training summary to %s", summary_path)


if __name__ == "__main__":
    main()
