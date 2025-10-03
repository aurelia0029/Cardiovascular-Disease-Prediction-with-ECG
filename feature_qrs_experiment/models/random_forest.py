from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..datasets.split_strategies import DatasetSplits


def train(splits: DatasetSplits, params: Dict, seed: int):
    clf_params = {
        "n_estimators": params.get("n_estimators", 200),
        "max_depth": params.get("max_depth"),
        "class_weight": params.get("class_weight"),
        "random_state": seed,
        "n_jobs": params.get("n_jobs", -1),
    }
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(**clf_params)),
    ])
    model.fit(splits.X_train, splits.y_train)
    return model


__all__ = ["train"]
