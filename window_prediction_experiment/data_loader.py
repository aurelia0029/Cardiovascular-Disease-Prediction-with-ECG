"""
Data loading and preprocessing module for MIT-BIH ECG dataset.

This module handles:
- Loading MIT-BIH Arrhythmia Database files
- Extracting beat windows from ECG signals
- Normalizing ECG beats
- Creating beat sequences for temporal modeling
"""

import wfdb
import os
import numpy as np
from tqdm import tqdm
from collections import Counter


def load_record_list(data_dir):
    """
    Load list of available records from MIT-BIH dataset directory.

    Args:
        data_dir (str): Path to MIT-BIH dataset directory

    Returns:
        list: Sorted list of record names
    """
    record_list = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.dat')]
    record_list = list(set(record_list))  # remove duplicates
    return sorted(record_list)


def get_all_symbols(data_dir, record_list):
    """
    Extract all beat annotation symbols from the dataset.

    Args:
        data_dir (str): Path to MIT-BIH dataset directory
        record_list (list): List of record names to process

    Returns:
        list: All annotation symbols across all records
        Counter: Frequency count of each symbol
    """
    all_symbols = []

    for record_name in record_list:
        record_path = os.path.join(data_dir, record_name)
        try:
            ann = wfdb.rdann(record_path, 'atr')
            all_symbols.extend(ann.symbol)
        except Exception as e:
            print(f"Skipping {record_name} due to error: {e}")

    symbol_counts = Counter(all_symbols)
    return all_symbols, symbol_counts


def normalize_beat(beat):
    """
    Normalize a single beat window to [-1, 1] range.

    Args:
        beat (numpy.ndarray): Raw ECG beat window

    Returns:
        numpy.ndarray or None: Normalized beat, or None if signal is flat
    """
    max_val = np.max(np.abs(beat))
    if max_val < 1e-6:  # flat or almost-flat signal
        return None
    return beat / max_val  # scale to [-1, 1]


def extract_beat_windows(data_dir, record_list, selected_symbols, normal_symbols, window_size=100):
    """
    Extract beat windows from ECG signals with annotations.

    Args:
        data_dir (str): Path to MIT-BIH dataset directory
        record_list (list): List of record names to process
        selected_symbols (list): Beat types to include (e.g., ['N', 'L', 'R', 'V', 'A', 'F'])
        normal_symbols (list): Symbols considered as normal (e.g., ['N'])
        window_size (int): Number of samples before/after R-peak (default: 100, total 200 samples)

    Returns:
        tuple: (beat_windows, beat_labels) as numpy arrays
    """
    beat_windows = []
    beat_labels = []
    abnormal_symbols = [s for s in selected_symbols if s not in normal_symbols]

    for record in tqdm(record_list, desc="Extracting beat windows"):
        try:
            record_path = os.path.join(data_dir, record)
            ann = wfdb.rdann(record_path, 'atr')
            rec = wfdb.rdrecord(record_path)
            signal = rec.p_signal[:, 0]  # using lead 0

            for i, r_peak in enumerate(ann.sample):
                # Skip beats too close to signal boundaries
                if r_peak < window_size or r_peak + window_size >= len(signal):
                    continue

                symbol = ann.symbol[i]
                if symbol not in selected_symbols:
                    continue

                # Extract and normalize beat window
                beat = signal[r_peak - window_size:r_peak + window_size]
                beat = normalize_beat(beat)
                if beat is None:
                    continue

                # Assign label: 0 for normal, 1 for abnormal
                label = 0 if symbol in normal_symbols else 1

                beat_windows.append(beat)
                beat_labels.append(label)

        except Exception as e:
            print(f"Skipping {record} due to error: {e}")

    return np.array(beat_windows), np.array(beat_labels)


def create_sequences(beats, labels, sequence_length=3):
    """
    Create sequences of consecutive beats for temporal modeling.

    Args:
        beats (numpy.ndarray): Array of beat windows, shape (num_beats, beat_length)
        labels (numpy.ndarray): Array of beat labels, shape (num_beats,)
        sequence_length (int): Number of consecutive beats per sequence

    Returns:
        tuple: (X_seq, y_seq) where X_seq has shape (num_samples, seq_len, beat_len)
               and y_seq contains the label of the next beat after the sequence
    """
    X_seq, y_seq = [], []

    for i in range(len(beats) - sequence_length):
        X_seq.append(beats[i:i + sequence_length])
        y_seq.append(labels[i + sequence_length])

    return np.array(X_seq), np.array(y_seq)


def map_to_binary_class(symbol, included_symbols, normal_symbols):
    """
    Map beat annotation symbol to binary class (normal vs abnormal).

    Args:
        symbol (str): Beat annotation symbol
        included_symbols (list): Symbols to include in dataset
        normal_symbols (list): Symbols considered as normal

    Returns:
        int or None: 0 for normal, 1 for abnormal, None to drop
    """
    if symbol not in included_symbols:
        return None
    return 0 if symbol in normal_symbols else 1
