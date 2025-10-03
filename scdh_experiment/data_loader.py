"""
Data Loading for SCDH Experiments

Loads ECG data from SCDH (Sudden Cardiac Death Holter Database)
and NSR (Normal Sinus Rhythm Database).

Extracts segments before VF (ventricular fibrillation) onset and
normal segments for binary classification.
"""

import os
import wfdb
import numpy as np
import pandas as pd
from datetime import timedelta
from tqdm import tqdm
from scipy.signal import resample

from feature_extraction import extract_features


def get_default_data_dir(db_name):
    """
    Get default data directory, auto-detecting if running in Docker or locally.

    Parameters:
    -----------
    db_name : str
        Database name ('sddb' or 'nsrdb')

    Returns:
    --------
    path : str
        Default path to database
    """
    # Check if running in Docker (data in /app/)
    docker_path = os.path.join('/app', db_name)
    if os.path.exists(docker_path):
        return docker_path

    # Check parent directory (when running locally from scdh_experiment/)
    local_path = os.path.join('..', db_name)
    if os.path.exists(local_path):
        return local_path

    # Check current directory
    current_path = db_name
    if os.path.exists(current_path):
        return current_path

    # Default to parent directory path
    return local_path


# VF onset times for each SCDH record (from annotations)
VF_ONSET_TIMES = {
    '30': '18:31:53',
    '31': '19:20:45',
    '32': '00:09:32',
    '33': '00:35:03',
    '34': '19:36:12',
    '35': '12:13:03',
    '36': '12:58:00',
    '37': '10:28:48',
    '38': '17:12:08',
    '39': '23:54:32',
    '41': '22:00:16',
    '43': '11:17:16',
    '44': '21:50:33',
    '45': '18:31:00',
    '46': '11:55:00',
    '47': '12:04:00',
    '48': '20:17:00',
    '50': '13:50:00',
    '51': '14:48:00',
    '52': '17:01:00',
    '53': '11:19:03',
    '54': '21:32:00'
}


def load_scdh_data(data_dir=None, segment_len=3, window_len=180,
                   time_before_onset=20, target_fs=250, skip_list=None):
    """
    Load ECG data from SCDH database.

    Extracts two types of segments:
    1. Onset segments: {time_before_onset} minutes BEFORE VF onset (labeled as 1)
    2. Normal segments: {time_before_onset} minutes AFTER VF ends (labeled as 0)

    Parameters:
    -----------
    data_dir : str
        Path to SCDH database directory
    segment_len : int
        Length of each segment in seconds (default: 3)
    window_len : int
        Length of extraction window in seconds (default: 180 = 3 minutes)
    time_before_onset : int
        How many minutes before/after VF onset to extract data (default: 20)
    target_fs : int
        Target sampling frequency for resampling (default: 250 Hz)
    skip_list : list
        List of record IDs to skip (default: ['40', '42', '49'] - paced/no VF)

    Returns:
    --------
    X : list
        List of feature dictionaries
    y : list
        List of labels (1 = onset, 0 = normal)
    """
    if data_dir is None:
        data_dir = get_default_data_dir('sddb')

    if skip_list is None:
        skip_list = ['40', '42', '49']

    X, y = [], []
    ari_files = [f for f in os.listdir(data_dir) if f.endswith('.ari')]

    print(f"Loading SCDH data from {data_dir}")
    print(f"Extracting {window_len}s windows {time_before_onset} min before/after VF onset")
    print(f"Segment length: {segment_len}s, Target sampling rate: {target_fs} Hz")

    for f in tqdm(ari_files, desc="Processing SCDH records"):
        record = f.replace('.ari', '')
        if record in skip_list or record not in VF_ONSET_TIMES:
            continue

        try:
            # Load annotation and signal
            ann = wfdb.rdann(os.path.join(data_dir, record), 'ari')
            sig, fields = wfdb.rdsamp(os.path.join(data_dir, record), channels=[0])
            fs = fields['fs']
            ecg = sig.flatten()

            # Resample if needed
            if fs != target_fs:
                ecg = resample(ecg, int(len(ecg) * target_fs / fs))
                fs = target_fs
                ann_samples = (ann.sample * target_fs / fields['fs']).astype(int)
            else:
                ann_samples = ann.sample

            # Parse VF onset time
            onset_time = VF_ONSET_TIMES[record]
            h, m, s = map(int, onset_time.split(':'))
            onset_sample = int(timedelta(hours=h, minutes=m, seconds=s).total_seconds() * fs)

            # Find VF start and end indices in annotations
            first_v_index = None
            for i, sym in enumerate(ann.symbol):
                if sym == 'V' and ann_samples[i] >= onset_sample:
                    first_v_index = i
                    break

            if first_v_index is None:
                continue

            last_v_index = first_v_index
            while last_v_index < len(ann.symbol) - 1 and ann.symbol[last_v_index + 1] == 'V':
                last_v_index += 1

            first_v_sample = ann_samples[first_v_index]
            last_v_sample = ann_samples[last_v_index]

            # Extract ONSET segment (time_before_onset minutes BEFORE VF onset)
            onset_start = first_v_sample - time_before_onset * 60 * fs
            onset_end = onset_start + window_len * fs
            if onset_start >= 0 and onset_end < len(ecg):
                onset_window = ecg[int(onset_start):int(onset_end)]
                # Split into smaller segments
                for i in range(0, len(onset_window), int(segment_len * fs)):
                    seg = onset_window[i:i + int(segment_len * fs)]
                    if len(seg) == int(segment_len * fs):
                        if np.isnan(seg).any() or np.std(seg) == 0:
                            continue  # Skip invalid segments
                        features = extract_features(seg, fs)
                        if not any(np.isnan(val) for val in features.values()):
                            X.append(features)
                            y.append(1)  # Onset = 1

            # Extract NORMAL segment (time_before_onset minutes AFTER VF ends)
            normal_start = last_v_sample + time_before_onset * 60 * fs
            normal_end = normal_start + window_len * fs
            if normal_end < len(ecg):
                normal_window = ecg[int(normal_start):int(normal_end)]
                # Split into smaller segments
                for i in range(0, len(normal_window), int(segment_len * fs)):
                    seg = normal_window[i:i + int(segment_len * fs)]
                    if len(seg) == int(segment_len * fs):
                        if np.isnan(seg).any() or np.std(seg) == 0:
                            continue
                        features = extract_features(seg, fs)
                        if not any(np.isnan(val) for val in features.values()):
                            X.append(features)
                            y.append(0)  # Normal = 0

        except Exception as e:
            print(f"[{record}] Error: {e}")

    print(f"\nExtracted {len(X)} segments total")
    print(f"  Onset (label=1): {sum(y)}")
    print(f"  Normal (label=0): {len(y) - sum(y)}")

    return X, y


def load_nsr_data(data_dir=None, segment_len=3, window_len=180,
                  num_windows=60, target_fs=250):
    """
    Load normal ECG data from NSR (Normal Sinus Rhythm) database.

    Parameters:
    -----------
    data_dir : str
        Path to NSR database directory
    segment_len : int
        Length of each segment in seconds (default: 3)
    window_len : int
        Length of extraction window in seconds (default: 180 = 3 minutes)
    num_windows : int
        Number of windows to extract per record (default: 60)
    target_fs : int
        Target sampling frequency (default: 250 Hz)

    Returns:
    --------
    X : list
        List of feature dictionaries (all labeled as normal)
    """
    if data_dir is None:
        data_dir = get_default_data_dir('nsrdb')

    X = []
    nsr_files = [f.replace('.hea', '') for f in os.listdir(data_dir) if f.endswith('.hea')]

    print(f"\nLoading NSR data from {data_dir}")
    print(f"Extracting {num_windows} windows of {window_len}s per record")

    for record in tqdm(nsr_files, desc="Processing NSR records"):
        try:
            sig, fields = wfdb.rdsamp(os.path.join(data_dir, record), channels=[0])
            fs = fields['fs']
            ecg = sig.flatten()

            # Resample if needed
            if fs != target_fs:
                ecg = resample(ecg, int(len(ecg) * target_fs / fs))
                fs = target_fs

            # Extract multiple windows from the record
            max_start = len(ecg) - window_len * fs
            if max_start <= 0:
                continue

            # Evenly spaced window starts
            starts = np.linspace(0, max_start, num_windows, dtype=int)

            for start in starts:
                window = ecg[start:start + int(window_len * fs)]
                # Split window into segments
                for i in range(0, len(window), int(segment_len * fs)):
                    seg = window[i:i + int(segment_len * fs)]
                    if len(seg) == int(segment_len * fs):
                        if np.isnan(seg).any() or np.std(seg) == 0:
                            continue
                        features = extract_features(seg, fs)
                        if not any(np.isnan(val) for val in features.values()):
                            X.append(features)

        except Exception as e:
            print(f"[NSR {record}] Error: {e}")

    print(f"Extracted {len(X)} normal segments from NSR")

    return X


def prepare_dataframe(X, y=None):
    """
    Convert feature list to pandas DataFrame.

    Parameters:
    -----------
    X : list
        List of feature dictionaries
    y : list (optional)
        List of labels

    Returns:
    --------
    df : pd.DataFrame
        DataFrame with features and optional labels
    """
    df = pd.DataFrame(X)

    if y is not None:
        df['label'] = y

    print(f"\nDataFrame shape: {df.shape}")
    if 'label' in df.columns:
        print(f"Class distribution:\n{df['label'].value_counts()}")

    return df
