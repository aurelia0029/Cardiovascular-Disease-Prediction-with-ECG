from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from tqdm import tqdm

from ..data_processing.event_selection import FeatureSample, collect_abnormal_samples, collect_normal_samples
from ..data_processing.wfdb_loader import iter_records, load_record
from ..utils.logging import get_logger


logger = get_logger(__name__)


def _unique_directories(paths: Iterable[str]) -> List[Path]:
    unique = []
    seen = set()
    for raw in paths:
        directory = Path(raw).resolve()
        if directory not in seen:
            seen.add(directory)
            unique.append(directory)
    return unique


def _collect_samples_from_dirs(directories: Iterable[Path], config: dict, collector) -> List[FeatureSample]:
    samples: List[FeatureSample] = []
    directories = list(directories)
    for directory in directories:
        record_pairs = list(iter_records([directory]))
        for current_dir, record_id in tqdm(record_pairs, desc=f"{collector.__name__}:{directory.name}"):
            try:
                record = load_record(current_dir, record_id)
            except FileNotFoundError:
                logger.warning("Missing record %s in %s", record_id, current_dir)
                continue
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load record %s: %s", record_id, exc)
                continue

            collected = collector(record, config)
            if collected:
                samples.extend(collected)
    return samples


def build_raw_feature_sets(config: dict, cache: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract abnormal and normal feature sets according to *config*."""
    abnormal_dirs = _unique_directories(config["paths"].get("abnormal_sources", []))
    normal_dirs = _unique_directories(config["paths"].get("normal_sources", []))

    logger.info("Collecting abnormal samples from %d directories", len(abnormal_dirs))
    abnormal_samples = _collect_samples_from_dirs(abnormal_dirs, config, collect_abnormal_samples)

    logger.info("Collecting normal samples from %d directories", len(normal_dirs))
    normal_samples = _collect_samples_from_dirs(normal_dirs, config, collect_normal_samples)

    def _to_arrays(samples: List[FeatureSample]) -> Dict[str, np.ndarray]:
        if not samples:
            return {"features": np.empty((0, 4)), "labels": np.empty((0,), dtype=object), "records": np.empty((0,), dtype=object)}
        features = np.stack([sample.features for sample in samples])
        labels = np.array([sample.label for sample in samples], dtype=object)
        records = np.array([sample.record_id for sample in samples], dtype=object)
        return {"features": features, "labels": labels, "records": records}

    abnormal = _to_arrays(abnormal_samples)
    normal = _to_arrays(normal_samples)

    if cache:
        cache_path = Path(config["paths"]["feature_cache"]).resolve()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            abnormal_features=abnormal["features"],
            abnormal_labels=abnormal["labels"],
            abnormal_records=abnormal["records"],
            normal_features=normal["features"],
            normal_labels=normal["labels"],
            normal_records=normal["records"],
        )
        logger.info("Saved feature cache to %s", cache_path)

    logger.info(
        "Collected %d abnormal samples and %d normal samples",
        abnormal["features"].shape[0],
        normal["features"].shape[0],
    )

    return {"abnormal": abnormal, "normal": normal}


__all__ = ["build_raw_feature_sets"]
