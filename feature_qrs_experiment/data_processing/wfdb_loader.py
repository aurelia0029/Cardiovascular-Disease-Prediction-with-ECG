from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import wfdb


@dataclass
class RecordData:
    """Container for signal and annotation data from a WFDB record."""

    record_id: str
    signal: np.ndarray
    samples: np.ndarray
    symbols: np.ndarray
    sampling_rate: int


def list_records(directory: Path) -> List[str]:
    """Return sorted record ids for all `.dat` files in *directory*."""
    directory = Path(directory)
    return sorted({path.stem for path in directory.glob("*.dat")})


def iter_records(directories: Iterable[Path]) -> Iterable[tuple[Path, str]]:
    """Yield `(directory, record_id)` pairs for each WFDB record."""
    for directory in directories:
        for record_id in list_records(directory):
            yield directory, record_id


def load_record(directory: Path, record_id: str, annotation: str = "atr", channel: int = 0) -> RecordData:
    """Load WFDB record + annotations and wrap them in `RecordData`."""
    base_path = Path(directory) / record_id
    ann = wfdb.rdann(str(base_path), annotation)
    rec = wfdb.rdrecord(str(base_path))
    signal = rec.p_signal[:, channel]
    samples = np.asarray(ann.sample, dtype=np.int64)
    symbols = np.asarray(ann.symbol, dtype=object)
    sampling_rate = int(rec.fs)
    return RecordData(
        record_id=record_id,
        signal=signal,
        samples=samples,
        symbols=symbols,
        sampling_rate=sampling_rate,
    )


__all__ = ["RecordData", "list_records", "iter_records", "load_record"]
