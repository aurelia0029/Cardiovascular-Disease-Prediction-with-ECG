from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sk_shuffle

from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    label_map: Dict[str, int]


def _split_abnormal(features: np.ndarray, labels: np.ndarray, config: dict, seed: int) -> Tuple:
    if features.size == 0:
        raise ValueError("No abnormal samples available for splitting")

    sampling_cfg = config["sampling"]
    train_ratio = sampling_cfg.get("train_ratio", 0.6)
    val_ratio = sampling_cfg.get("val_ratio", 0.2)
    test_ratio = 1.0 - train_ratio

    X_abn_train, X_abn_temp, y_abn_train, y_abn_temp = train_test_split(
        features,
        labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )

    if len(y_abn_temp) == 0:
        raise ValueError("Insufficient abnormal samples for validation/test splits")

    val_fraction = val_ratio / (1.0 - train_ratio)

    X_abn_val, X_abn_test, y_abn_val, y_abn_test = train_test_split(
        X_abn_temp,
        y_abn_temp,
        test_size=1.0 - val_fraction,
        random_state=seed,
        stratify=y_abn_temp if len(np.unique(y_abn_temp)) > 1 else None,
    )

    return (X_abn_train, y_abn_train, X_abn_val, y_abn_val, X_abn_test, y_abn_test)


def _sample_normals(normals: np.ndarray, size: int, seed: int, replace: bool = False) -> np.ndarray:
    if size <= 0:
        return np.empty((0, normals.shape[1]))
    rng = np.random.default_rng(seed)
    replace = replace or (len(normals) < size)
    indices = rng.choice(len(normals), size=size, replace=replace)
    return normals[indices]


def _assemble_dataset(X_norm: np.ndarray, X_abn: np.ndarray) -> np.ndarray:
    return np.concatenate([X_norm, X_abn], axis=0)


def _assemble_labels(n_norm: int, n_abn: int) -> np.ndarray:
    return np.concatenate([np.zeros(n_norm, dtype=int), np.ones(n_abn, dtype=int)])


def _shuffle(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    return sk_shuffle(X, y, random_state=seed)


def balanced_split(raw: dict, config: dict, seed: int) -> DatasetSplits:
    abnormal = raw["abnormal"]["features"]
    abnormal_labels = raw["abnormal"]["labels"]
    normals = raw["normal"]["features"]

    X_abn_train, y_abn_train, X_abn_val, y_abn_val, X_abn_test, y_abn_test = _split_abnormal(abnormal, abnormal_labels, config, seed)

    sampling_cfg = config["sampling"]
    train_ratio = sampling_cfg.get("train_ratio", 0.6)
    val_ratio = sampling_cfg.get("val_ratio", 0.2)
    val_fraction = val_ratio / (1.0 - train_ratio)

    X_norm_train, X_norm_temp = train_test_split(normals, test_size=1.0 - train_ratio, random_state=seed)
    if len(X_norm_temp) == 0:
        raise ValueError("Insufficient normal samples for validation/test splits")

    X_norm_val, X_norm_test = train_test_split(X_norm_temp, test_size=1.0 - val_fraction, random_state=seed)

    X_norm_train_bal = _sample_normals(X_norm_train, len(X_abn_train), seed)
    X_norm_val_bal = _sample_normals(X_norm_val, len(X_abn_val), seed + 1)
    X_norm_test_bal = _sample_normals(X_norm_test, len(X_abn_test), seed + 2)

    X_train = _assemble_dataset(X_norm_train_bal, X_abn_train)
    y_train = _assemble_labels(len(X_norm_train_bal), len(X_abn_train))
    X_val = _assemble_dataset(X_norm_val_bal, X_abn_val)
    y_val = _assemble_labels(len(X_norm_val_bal), len(X_abn_val))
    X_test = _assemble_dataset(X_norm_test_bal, X_abn_test)
    y_test = _assemble_labels(len(X_norm_test_bal), len(X_abn_test))

    X_train, y_train = _shuffle(X_train, y_train, seed)
    X_val, y_val = _shuffle(X_val, y_val, seed + 3)
    X_test, y_test = _shuffle(X_test, y_test, seed + 4)

    label_map = {"N": 0, "abnormal": 1}

    return DatasetSplits(X_train, y_train, X_val, y_val, X_test, y_test, label_map)


def _clean_and_sample_normals(normals: np.ndarray, keep_ratio: float, outlier_percentile: float, seed: int) -> np.ndarray:
    if normals.size == 0:
        return normals

    pca = PCA(n_components=min(2, normals.shape[1]))
    reduced = pca.fit_transform(normals)
    center = np.mean(reduced, axis=0)
    dists = np.linalg.norm(reduced - center, axis=1)
    cutoff = np.percentile(dists, outlier_percentile)

    filtered = normals[dists <= cutoff]
    if len(filtered) == 0:
        filtered = normals

    sample_size = max(1, int(len(filtered) * keep_ratio))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(filtered), size=sample_size, replace=False if len(filtered) >= sample_size else True)
    return filtered[indices]


def _oversample_majority(X: np.ndarray, target_size: int, seed: int) -> np.ndarray:
    if len(X) >= target_size:
        return X
    rng = np.random.default_rng(seed)
    need = target_size - len(X)
    indices = rng.choice(len(X), size=need, replace=True)
    return np.concatenate([X, X[indices]], axis=0)


def cleaned_oversample_split(raw: dict, config: dict, seed: int) -> DatasetSplits:
    abnormal = raw["abnormal"]["features"]
    abnormal_labels = raw["abnormal"]["labels"]
    normals = raw["normal"]["features"]

    X_abn_train, y_abn_train, X_abn_val, y_abn_val, X_abn_test, y_abn_test = _split_abnormal(abnormal, abnormal_labels, config, seed)

    normals = sk_shuffle(normals, random_state=seed)
    sampling_cfg = config["sampling"]
    initial_ratio = sampling_cfg.get("initial_normal_ratio", 0.33)
    initial_count = max(1, int(len(normals) * initial_ratio))
    normals_initial = normals[:initial_count]

    train_ratio = sampling_cfg.get("train_ratio", 0.6)
    val_ratio = sampling_cfg.get("val_ratio", 0.2)
    val_fraction = val_ratio / (1.0 - train_ratio)

    X_norm_train, X_norm_temp = train_test_split(normals_initial, test_size=1.0 - train_ratio, random_state=seed)
    if len(X_norm_temp) == 0:
        raise ValueError("Insufficient normal samples for validation/test splits")

    X_norm_val, X_norm_test = train_test_split(X_norm_temp, test_size=1.0 - val_fraction, random_state=seed)

    clean_cfg = sampling_cfg.get("clean_normals", {})
    keep_ratio = clean_cfg.get("keep_ratio", 0.3)
    outlier_percentile = clean_cfg.get("outlier_percentile", 97)

    X_norm_train_clean = _clean_and_sample_normals(X_norm_train, keep_ratio, outlier_percentile, seed)
    X_norm_val_clean = _clean_and_sample_normals(X_norm_val, keep_ratio, outlier_percentile, seed + 1)

    oversample = sampling_cfg.get("oversample_abnormal", True)

    if oversample:
        X_abn_train_bal = _oversample_majority(X_abn_train, len(X_norm_train_clean), seed)
        X_abn_val_bal = _oversample_majority(X_abn_val, len(X_norm_val_clean), seed + 1)
    else:
        X_abn_train_bal = X_abn_train
        X_abn_val_bal = X_abn_val

    X_train = _assemble_dataset(X_norm_train_clean, X_abn_train_bal)
    y_train = _assemble_labels(len(X_norm_train_clean), len(X_abn_train_bal))

    X_val = _assemble_dataset(X_norm_val_clean, X_abn_val_bal)
    y_val = _assemble_labels(len(X_norm_val_clean), len(X_abn_val_bal))

    X_test = _assemble_dataset(X_norm_test, X_abn_test)
    y_test = _assemble_labels(len(X_norm_test), len(X_abn_test))

    X_train, y_train = _shuffle(X_train, y_train, seed)
    X_val, y_val = _shuffle(X_val, y_val, seed + 2)
    X_test, y_test = _shuffle(X_test, y_test, seed + 3)

    label_map = {"N": 0, "abnormal": 1}

    return DatasetSplits(X_train, y_train, X_val, y_val, X_test, y_test, label_map)


def prepare_datasets(raw: dict, config: dict, seed: int) -> DatasetSplits:
    strategy = config["sampling"].get("strategy", "balanced").lower()
    if strategy == "balanced":
        return balanced_split(raw, config, seed)
    if strategy == "cleaned_oversample":
        return cleaned_oversample_split(raw, config, seed)
    raise ValueError(f"Unknown sampling strategy: {strategy}")


__all__ = [
    "DatasetSplits",
    "prepare_datasets",
]
