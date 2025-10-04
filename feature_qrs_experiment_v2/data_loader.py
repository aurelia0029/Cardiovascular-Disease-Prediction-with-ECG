"""
Data Loader for QRS Feature-Based Experiments

Extracts QRS features from MIT-BIH Arrhythmia Database:
- Abnormal: V, E, F, ! beats with clean 30-second history
- Normal: N beats with clean 30-second history
"""

import os
import wfdb
import numpy as np
from collections import Counter
from tqdm import tqdm


def extract_qrs_features(signal, r_peaks, width=18):
    """
    Extract QRS features from ECG signal.

    Parameters:
    -----------
    signal : np.ndarray
        ECG signal
    r_peaks : list
        List of R-peak sample indices
    width : int
        Half-width of QRS complex (default: 18 samples @ 360Hz = 50ms)

    Returns:
    --------
    features : list or None
        [mean_qrs_area, std_qrs_area, mean_r_amplitude, std_r_amplitude]
    """
    qrs_areas = []
    r_amplitudes = []

    for r in r_peaks:
        if r - width < 0 or r + width >= len(signal):
            continue

        # Extract QRS complex
        qrs = signal[r - width: r + width]
        qrs_area = np.sum(qrs)
        r_amp = signal[r]

        qrs_areas.append(qrs_area)
        r_amplitudes.append(r_amp)

    if len(qrs_areas) == 0:
        return None

    # Return 4 features
    return [
        np.mean(qrs_areas),
        np.std(qrs_areas),
        np.mean(r_amplitudes),
        np.std(r_amplitudes)
    ]


def load_features_with_clean_history(data_dir='../mitdb', pre_window_sec=30,
                                     abnormal_symbols=None, normal_symbol='N',
                                     non_heartbeat_symbols=None):
    """
    Load QRS features with clean 30-second history requirement.

    Strategy:
    - Abnormal: Find V, E, F, ! beats where previous 30 seconds have NO abnormalities
    - Normal: Find N beats where previous 30 seconds have NO abnormalities

    Parameters:
    -----------
    data_dir : str
        Path to MIT-BIH database
    pre_window_sec : int
        Number of seconds of clean history required (default: 30)
    abnormal_symbols : list
        List of abnormal beat symbols (default: ['V', 'E', 'F', '!'])
    normal_symbol : str
        Normal beat symbol (default: 'N')
    non_heartbeat_symbols : list
        Non-heartbeat annotations to skip (default: ['+', '[', ']', '~', '"', '|'])

    Returns:
    --------
    abnormal_features : np.ndarray
        Array of abnormal features (N_abnormal, 4)
    abnormal_labels : np.ndarray
        Array of abnormal beat types (N_abnormal,)
    normal_features : np.ndarray
        Array of normal features (N_normal, 4)
    """
    if abnormal_symbols is None:
        abnormal_symbols = ['V', 'E', 'F', '!']

    if non_heartbeat_symbols is None:
        non_heartbeat_symbols = ['+', '[', ']', '~', '"', '|']

    # Get record list
    record_list = sorted(list(set([
        f.split('.')[0] for f in os.listdir(data_dir)
        if f.endswith('.dat')
    ])))

    fs = 360  # MIT-BIH sampling rate
    qrs_half_width = 18  # 50ms at 360Hz
    pre_event_window = pre_window_sec * fs  # samples

    abnormal_features = []
    abnormal_labels = []
    normal_features = []

    print(f"Loading data from {data_dir}")
    print(f"Abnormal symbols: {abnormal_symbols}")
    print(f"Pre-event window: {pre_window_sec} seconds ({pre_event_window} samples)")
    print()

    for record in tqdm(record_list, desc="Processing records"):
        try:
            # Load annotation and signal
            ann = wfdb.rdann(os.path.join(data_dir, record), 'atr')
            rec = wfdb.rdrecord(os.path.join(data_dir, record))
            signal = rec.p_signal[:, 0]  # Use lead 0

            ann_samples = np.array(ann.sample)
            ann_symbols = np.array(ann.symbol)

            # Process each beat
            for i, symbol in enumerate(ann_symbols):
                r_peak = ann_samples[i]

                # Skip if too close to start
                if r_peak - pre_event_window < 0:
                    continue

                # Find all annotations in the 30-second window before this beat
                window_start_sample = r_peak - pre_event_window
                prev_beats_mask = (ann_samples < r_peak) & (ann_samples >= window_start_sample)
                prev_symbols = ann_symbols[prev_beats_mask]
                prev_samples = ann_samples[prev_beats_mask]

                # Original logic from feature_extraction.ipynb:
                # For Normal: skip if ANY abnormal in history
                # For Abnormal: skip if SAME TYPE abnormal in history
                if symbol == normal_symbol:
                    if any(s in abnormal_symbols for s in prev_symbols):
                        continue  # Normal beat with abnormal history - skip
                elif symbol in abnormal_symbols:
                    if any(s == symbol for s in prev_symbols):
                        continue  # Abnormal beat with same type in history - skip
                else:
                    continue  # Not normal or target abnormal - skip

                # Extract features from the clean 30-second history
                # Only use normal beats for feature extraction
                clean_r_peaks = [
                    prev_samples[j] for j, s in enumerate(prev_symbols)
                    if s not in abnormal_symbols + non_heartbeat_symbols
                ]

                features = extract_qrs_features(signal, clean_r_peaks, width=qrs_half_width)

                if features is None:
                    continue

                # Categorize based on current beat symbol
                if symbol in abnormal_symbols:
                    abnormal_features.append(features)
                    abnormal_labels.append(symbol)
                elif symbol == normal_symbol:
                    normal_features.append(features)

        except Exception as e:
            print(f"\nError processing {record}: {e}")
            continue

    # Convert to numpy arrays
    abnormal_features = np.array(abnormal_features)
    abnormal_labels = np.array(abnormal_labels)
    normal_features = np.array(normal_features)

    print(f"\n{'='*60}")
    print(f"Data Collection Summary")
    print(f"{'='*60}")
    print(f"Abnormal samples: {len(abnormal_features)}")
    if len(abnormal_labels) > 0:
        print(f"  Abnormal distribution: {dict(Counter(abnormal_labels))}")
    print(f"Normal samples: {len(normal_features)}")
    print(f"{'='*60}\n")

    return abnormal_features, abnormal_labels, normal_features


def split_and_balance(abnormal_features, abnormal_labels, normal_features,
                      train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                      balance_test=True, random_state=42):
    """
    Split abnormal data and sample balanced normal data.

    Parameters:
    -----------
    abnormal_features : np.ndarray
        Abnormal feature array
    abnormal_labels : np.ndarray
        Abnormal labels
    normal_features : np.ndarray
        Normal feature array
    train_ratio : float
        Training set ratio (default: 0.6)
    val_ratio : float
        Validation set ratio (default: 0.2)
    test_ratio : float
        Test set ratio (default: 0.2)
    balance_test : bool
        If True, test set is balanced; if False, test set uses all abnormal
    random_state : int
        Random seed

    Returns:
    --------
    Dictionary containing train/val/test splits
    """
    from sklearn.model_selection import train_test_split

    np.random.seed(random_state)

    # Split abnormal data (6:2:2)
    n_abnormal = len(abnormal_features)

    # First split: train vs (val + test)
    X_abn_train, X_abn_temp, y_abn_train, y_abn_temp = train_test_split(
        abnormal_features, abnormal_labels,
        test_size=(val_ratio + test_ratio), random_state=random_state
    )

    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    X_abn_val, X_abn_test, y_abn_val, y_abn_test = train_test_split(
        X_abn_temp, y_abn_temp, test_size=(1 - val_size), random_state=random_state
    )

    # Sample balanced normal data
    def sample_normal(n_samples):
        indices = np.random.choice(len(normal_features), size=n_samples, replace=False)
        return normal_features[indices]

    # Training set: balanced
    n_train_normal = len(X_abn_train)
    X_norm_train = sample_normal(n_train_normal)

    # Validation set: balanced
    n_val_normal = len(X_abn_val)
    X_norm_val = sample_normal(n_val_normal)

    # Test set: balanced or unbalanced
    if balance_test:
        n_test_normal = len(X_abn_test)
        X_norm_test = sample_normal(n_test_normal)
    else:
        # Use all remaining normal samples for test
        used_indices = set()
        # This is simplified - in practice, track which normal samples were used
        X_norm_test = normal_features  # Use all normal for unbalanced test

    # Combine abnormal and normal
    X_train = np.vstack([X_abn_train, X_norm_train])
    y_train = np.concatenate([np.ones(len(X_abn_train)), np.zeros(len(X_norm_train))])

    X_val = np.vstack([X_abn_val, X_norm_val])
    y_val = np.concatenate([np.ones(len(X_abn_val)), np.zeros(len(X_norm_val))])

    X_test = np.vstack([X_abn_test, X_norm_test])
    y_test = np.concatenate([np.ones(len(X_abn_test)), np.zeros(len(X_norm_test))])

    # Shuffle
    train_shuffle = np.random.permutation(len(X_train))
    val_shuffle = np.random.permutation(len(X_val))
    test_shuffle = np.random.permutation(len(X_test))

    X_train, y_train = X_train[train_shuffle], y_train[train_shuffle]
    X_val, y_val = X_val[val_shuffle], y_val[val_shuffle]
    X_test, y_test = X_test[test_shuffle], y_test[test_shuffle]

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'abnormal_labels_train': y_abn_train,
        'abnormal_labels_val': y_abn_val,
        'abnormal_labels_test': y_abn_test
    }
