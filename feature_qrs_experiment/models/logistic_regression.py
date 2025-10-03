from __future__ import annotations

from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..datasets.split_strategies import DatasetSplits


def train(splits: DatasetSplits, params: Dict, seed: int):
    clf_params = {
        "max_iter": params.get("max_iter", 1000),
        "class_weight": params.get("class_weight"),
        "random_state": seed,
    }
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(**clf_params)),
    ])
    model.fit(splits.X_train, splits.y_train)
    return model


__all__ = ["train"]
