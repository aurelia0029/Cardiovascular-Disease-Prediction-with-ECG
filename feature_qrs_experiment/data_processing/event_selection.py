from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .feature_extractor import extract_qrs_features
from .wfdb_loader import RecordData


@dataclass
class FeatureSample:
    """A single feature vector associated with a record and label."""

    record_id: str
    label: str
    features: np.ndarray


def _build_window_mask(samples: np.ndarray, start: int, end: int) -> np.ndarray:
    """Return boolean mask for annotation samples between start (inclusive) and end (exclusive)."""
    return (samples >= start) & (samples < end)


def collect_abnormal_samples(record: RecordData, config: dict) -> List[FeatureSample]:
    extraction = config["extraction"]
    v_symbols: Sequence[str] = extraction["ventricular_symbols"]
    non_beats: Sequence[str] = extraction["non_heartbeat_symbols"]
    qrs_half_width: int = extraction["qrs_half_width"]
    pre_event_window_sec: float = extraction["pre_event_window_sec"]

    pre_event_window_samples = int(pre_event_window_sec * record.sampling_rate)

    samples: List[FeatureSample] = []

    for idx, symbol in enumerate(record.symbols):
        if symbol not in v_symbols:
            continue

        r_peak = int(record.samples[idx])
        start = r_peak - pre_event_window_samples
        if start < 0:
            continue

        window_mask = _build_window_mask(record.samples, start, r_peak)
        if not np.any(window_mask):
            continue

        window_symbols = record.symbols[window_mask]
        if any(sym in v_symbols for sym in window_symbols):
            # Skip windows that already contain ventricular events.
            continue

        window_samples = record.samples[window_mask]
        normal_r_peaks = [int(window_samples[j]) for j, sym in enumerate(window_symbols)
                          if sym not in v_symbols and sym not in non_beats]
        if not normal_r_peaks:
            continue

        features = extract_qrs_features(record.signal, normal_r_peaks, width=qrs_half_width)
        if features is None:
            continue

        samples.append(FeatureSample(record.record_id, symbol, features))

    return samples


def collect_normal_samples(record: RecordData, config: dict) -> List[FeatureSample]:
    extraction = config["extraction"]
    v_symbols: Sequence[str] = extraction["ventricular_symbols"]
    non_beats: Sequence[str] = extraction["non_heartbeat_symbols"]
    qrs_half_width: int = extraction["qrs_half_width"]
    pre_event_window_sec: float = extraction["pre_event_window_sec"]

    pre_event_window_samples = int(pre_event_window_sec * record.sampling_rate)

    samples: List[FeatureSample] = []

    for idx, symbol in enumerate(record.symbols):
        if symbol != "N":
            continue

        r_peak = int(record.samples[idx])
        start = r_peak - pre_event_window_samples
        if start < 0:
            continue

        window_mask = _build_window_mask(record.samples, start, r_peak)
        if not np.any(window_mask):
            continue

        window_symbols = record.symbols[window_mask]
        if any(sym in v_symbols for sym in window_symbols):
            continue

        window_samples = record.samples[window_mask]
        normal_r_peaks = [int(window_samples[j]) for j, sym in enumerate(window_symbols)
                          if sym not in v_symbols and sym not in non_beats]
        if not normal_r_peaks:
            continue

        features = extract_qrs_features(record.signal, normal_r_peaks, width=qrs_half_width)
        if features is None:
            continue

        samples.append(FeatureSample(record.record_id, "N", features))

    return samples


def merge_samples(sample_groups: Iterable[Sequence[FeatureSample]]) -> Iterable[FeatureSample]:
    for group in sample_groups:
        yield from group


__all__ = [
    "FeatureSample",
    "collect_abnormal_samples",
    "collect_normal_samples",
    "merge_samples",
]
