"""
Feature Extraction for SCDH Experiments

Extracts comprehensive features from ECG segments including:
- Statistical features (mean, std, skewness, kurtosis, etc.)
- Nonlinear features (entropy, fractal dimension, Hurst exponent, etc.)
- Frequency domain features (power spectral density, band power, etc.)
- Signal variation features (zero-crossing rate, signal range, etc.)
"""

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from antropy import (
    higuchi_fd, petrosian_fd, sample_entropy, app_entropy
)
from nolds import hurst_rs, dfa


def lempel_ziv_complexity(signal):
    """
    Compute Lempel-Ziv Complexity (LZ76) for a 1D numpy array.
    The signal will be binarized by comparing with the mean.

    Parameters:
    -----------
    signal : np.ndarray
        1D signal array

    Returns:
    --------
    complexity : int
        Lempel-Ziv complexity value
    """
    binary_seq = ''.join(['1' if x > np.mean(signal) else '0' for x in signal])
    i, k, l = 0, 1, 1
    complexity = 1

    while True:
        if i + k > len(binary_seq):
            complexity += 1
            break
        substring = binary_seq[i:i+k]
        if substring not in binary_seq[0:l]:
            complexity += 1
            l += k
            i = l
            k = 1
        else:
            k += 1

    return complexity


def bandpower(freqs, psd, low, high):
    """
    Calculate power in a specific frequency band.

    Parameters:
    -----------
    freqs : np.ndarray
        Frequency array from welch()
    psd : np.ndarray
        Power spectral density from welch()
    low : float
        Lower frequency bound (Hz)
    high : float
        Upper frequency bound (Hz)

    Returns:
    --------
    power : float
        Total power in the specified band
    """
    return np.sum(psd[(freqs >= low) & (freqs < high)])


def extract_features(signal, fs=250):
    """
    Extract comprehensive features from ECG segment.

    Parameters:
    -----------
    signal : np.ndarray
        1D ECG signal array (typically 3 seconds @ 250 Hz = 750 samples)
    fs : int
        Sampling frequency (Hz)

    Returns:
    --------
    features : dict
        Dictionary containing 27 extracted features
    """
    # Handle NaN values
    signal = np.nan_to_num(signal)

    # Compute power spectral density
    freqs, psd = welch(signal, fs=fs)

    # Extract features in groups

    # Statistical features (9)
    stat_features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'var': np.var(signal),
        'rms': np.sqrt(np.mean(signal**2)),
        'skewness': skew(signal),
        'kurtosis': kurtosis(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'median': np.median(signal)
    }

    # Nonlinear features (6)
    nonlinear_features = {
        'higuchi_fd': higuchi_fd(signal),
        'petrosian_fd': petrosian_fd(signal),
        'sample_entropy': sample_entropy(signal),
        'hurst': hurst_rs(signal),
        'lempel_ziv': lempel_ziv_complexity(signal > np.mean(signal)),
        'app_entropy': app_entropy(signal),
        'dfa': dfa(signal)
    }

    # Frequency domain features (6)
    freq_features = {
        'peak_freq': freqs[np.argmax(psd)],
        'total_power': np.sum(psd),
        'mean_power': np.mean(psd),
        'power_0_5hz': bandpower(freqs, psd, 0, 5),
        'power_5_15hz': bandpower(freqs, psd, 5, 15),
        'power_15_40hz': bandpower(freqs, psd, 15, 40)
    }

    # Signal variation features (4)
    variation_features = {
        'max_diff': np.max(np.abs(np.diff(signal))),
        'sum_abs_diff': np.sum(np.abs(np.diff(signal))),
        'signal_range': np.max(signal) - np.min(signal),
        'zero_crossing': int(((signal[:-1] * signal[1:]) < 0).sum())
    }

    # Combine all features
    features = {}
    features.update(stat_features)
    features.update(nonlinear_features)
    features.update(freq_features)
    features.update(variation_features)

    return features


def extract_features_list(signal, fs=250):
    """
    Extract features as a list (for compatibility with original code).

    Parameters:
    -----------
    signal : np.ndarray
        1D ECG signal array
    fs : int
        Sampling frequency (Hz)

    Returns:
    --------
    features : list
        List of 27 feature values
    """
    feature_dict = extract_features(signal, fs)
    return list(feature_dict.values())
