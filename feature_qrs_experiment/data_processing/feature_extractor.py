from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def extract_qrs_features(signal: np.ndarray, r_peaks: Iterable[int], width: int = 18) -> Optional[np.ndarray]:
    """Return basic QRS statistics for the given R-peak indices.

    Parameters
    ----------
    signal
        One-dimensional ECG signal.
    r_peaks
        Iterable of sample indices containing normal beats.
    width
        Half-width (in samples) for the QRS window on each side of the R-peak.
    """
    qrs_areas = []
    r_amplitudes = []

    for r in r_peaks:
        r = int(r)
        if r - width < 0 or r + width >= len(signal):
            continue
        window = signal[r - width : r + width]
        qrs_areas.append(np.sum(window))
        r_amplitudes.append(signal[r])

    if not qrs_areas:
        return None

    qrs_areas = np.asarray(qrs_areas)
    r_amplitudes = np.asarray(r_amplitudes)

    return np.array([
        float(np.mean(qrs_areas)),
        float(np.std(qrs_areas)),
        float(np.mean(r_amplitudes)),
        float(np.std(r_amplitudes)),
    ])


__all__ = ["extract_qrs_features"]
