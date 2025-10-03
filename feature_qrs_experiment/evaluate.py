from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .datasets.split_strategies import DatasetSplits
from .utils.logging import get_logger


logger = get_logger(__name__)


def _safe_predict_proba(model, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None


def evaluate_model(model: Any, splits: DatasetSplits, model_name: str, artifact_dir: Path) -> Dict[str, Dict[str, float]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics: Dict[str, Dict[str, float]] = {}

    for split_name, X, y in (
        ("train", splits.X_train, splits.y_train),
        ("val", splits.X_val, splits.y_val),
        ("test", splits.X_test, splits.y_test),
    ):
        y_pred = model.predict(X)
        proba = _safe_predict_proba(model, X)
        if proba is not None:
            y_prob = proba[:, 1]
            roc_auc = roc_auc_score(y, y_prob)
        else:
            y_prob = None
            roc_auc = float("nan")

        report = classification_report(y, y_pred, digits=4)
        matrix = confusion_matrix(y, y_pred)

        metrics[split_name] = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc,
        }

        report_path = artifact_dir / f"{model_name}_{split_name}_report.txt"
        matrix_path = artifact_dir / f"{model_name}_{split_name}_confusion.npy"

        report_path.write_text(report)
        np.save(matrix_path, matrix)

        logger.info(
            "%s split - acc: %.4f, prec: %.4f, rec: %.4f, f1: %.4f, roc_auc: %.4f",
            split_name,
            metrics[split_name]["accuracy"],
            metrics[split_name]["precision"],
            metrics[split_name]["recall"],
            metrics[split_name]["f1"],
            metrics[split_name]["roc_auc"],
        )

    metrics_path = artifact_dir / f"{model_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    logger.info("Saved metrics summary to %s", metrics_path)

    return metrics


__all__ = ["evaluate_model"]
