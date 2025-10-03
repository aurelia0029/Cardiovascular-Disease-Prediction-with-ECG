from __future__ import annotations

from typing import Callable, Dict

from . import logistic_regression, random_forest

ModelTrainer = Callable[["DatasetSplits", Dict, int], object]

MODEL_REGISTRY: Dict[str, ModelTrainer] = {
    "logistic_regression": logistic_regression.train,
    "random_forest": random_forest.train,
}

try:  # torch may be unavailable in lightweight environments
    from . import simple_fnn  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    simple_fnn = None
else:
    MODEL_REGISTRY["simple_fnn"] = simple_fnn.train


__all__ = ["MODEL_REGISTRY"]
