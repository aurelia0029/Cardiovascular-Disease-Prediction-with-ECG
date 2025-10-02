"""
Data processing module for True Prediction Experiment.

This module provides specialized functions for creating sequences that
filter out any input containing abnormal beats, ensuring the model learns
to predict abnormalities before they appear, not just recognize them.
"""

import numpy as np
from collections import Counter


def create_pure_normal_sequences(beats, labels, sequence_length=7, abnormal_symbols=None):
    """
    Create sequences where ALL INPUT beats are normal (label=0).

    This function filters out sequences where ANY beat in the input window
    is abnormal, ensuring the model predicts future abnormalities rather
    than just recognizing existing ones.

    Key Innovation:
    ---------------
    Previous experiments showed that models often achieve high performance
    by simply detecting abnormal beats already present in the input sequence.
    This function creates a true prediction task by requiring:

    - Input: [N, N, N, N, N, N, N] (all normal)
    - Predict: Next beat (which might be abnormal)

    Args:
        beats (numpy.ndarray): ECG beat windows, shape (num_beats, beat_length)
        labels (numpy.ndarray): Binary labels (0=normal, 1=abnormal)
        sequence_length (int): Number of consecutive beats in input
        abnormal_symbols (list, optional): For logging purposes

    Returns:
        tuple: (X_seq, y_seq) where:
            - X_seq: shape (num_samples, seq_len, beat_len) - pure normal inputs
            - y_seq: shape (num_samples,) - labels for next beat

    Example:
        If we have beats with labels [0,0,0,1,0,0,0,1,1,0,0]:

        Original approach might create:
        - Input: [0,0,0,1,0,0,0] -> Predict: 1 (easy, abnormal already in input)

        This function creates:
        - Input: [0,0,0] -> Predict: 1 (hard, must predict before seeing abnormal)
        - Input: [0,0,0,0,0] -> Skip (next beat is 1, but we keep this)
        - Input: [1,0,0,0] -> Skip (input contains abnormal)
    """
    X_seq, y_seq = [], []

    for i in range(len(beats) - sequence_length):
        # Get labels of input sequence
        input_labels = labels[i:i + sequence_length]

        # Filter: ONLY keep sequences where ALL inputs are normal (0)
        if np.any(input_labels != 0):
            continue

        # Input is pure normal, label is the next beat
        X_seq.append(beats[i:i + sequence_length])
        y_seq.append(labels[i + sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Print statistics
    label_dist = Counter(y_seq)
    print(f"\n  Filtered to {len(X_seq)} pure normal sequences:")
    print(f"  - Next beat normal: {label_dist.get(0, 0)}")
    print(f"  - Next beat abnormal: {label_dist.get(1, 0)}")
    print(f"  - Abnormal ratio: {label_dist.get(1, 0)/len(y_seq)*100:.2f}%")

    return X_seq, y_seq


def define_abnormal_symbols(config='all'):
    """
    Define which beat types are considered abnormal (label=1).

    This allows experimenting with different definitions of "abnormal"
    to understand which types of abnormalities are most predictable.

    Args:
        config (str): Abnormal definition configuration. Options:
            - 'all': All non-normal beats ['L', 'R', 'V', 'A', 'F']
            - 'ventricular': Only ventricular-origin abnormalities ['V', 'F']
            - 'atrial': Only atrial-origin abnormalities ['A']
            - 'bundle_branch': Only bundle branch blocks ['L', 'R']
            - 'premature': Only premature beats ['V', 'A']
            - 'custom': Return empty list (user specifies)

    Returns:
        tuple: (selected_symbols, normal_symbols, abnormal_symbols, description)

    Beat Type Definitions:
    ---------------------
    N - Normal beat (label=0)
    L - Left bundle branch block
    R - Right bundle branch block
    V - Ventricular premature beat (危險！)
    A - Atrial premature beat
    F - Fusion beat (ventricular + normal)

    Clinical Significance:
    ---------------------
    Ventricular abnormalities (V, F) are generally more dangerous:
    - Can lead to ventricular fibrillation
    - Associated with sudden cardiac death
    - Higher clinical priority

    Experimental Results (from original notebook):
    - All abnormalities: Precision=3.25%, Recall=87.31%
    - Ventricular only: Precision improved but still low

    The low precision indicates the model struggles to predict
    abnormalities before they occur, highlighting the difficulty
    of true predictive monitoring.
    """

    configs = {
        'all': {
            'selected': ['N', 'L', 'R', 'V', 'A', 'F'],
            'normal': ['N'],
            'abnormal': ['L', 'R', 'V', 'A', 'F'],
            'desc': 'All non-normal beats'
        },
        'ventricular': {
            'selected': ['N', 'V', 'F'],
            'normal': ['N'],
            'abnormal': ['V', 'F'],
            'desc': 'Ventricular-origin only (V=premature, F=fusion) - Most dangerous'
        },
        'atrial': {
            'selected': ['N', 'A'],
            'normal': ['N'],
            'abnormal': ['A'],
            'desc': 'Atrial premature beats only'
        },
        'bundle_branch': {
            'selected': ['N', 'L', 'R'],
            'normal': ['N'],
            'abnormal': ['L', 'R'],
            'desc': 'Bundle branch blocks only (L=left, R=right)'
        },
        'premature': {
            'selected': ['N', 'V', 'A'],
            'normal': ['N'],
            'abnormal': ['V', 'A'],
            'desc': 'Premature beats (ventricular + atrial)'
        },
        'custom': {
            'selected': ['N'],
            'normal': ['N'],
            'abnormal': [],
            'desc': 'Custom configuration (specify your own)'
        }
    }

    if config not in configs:
        print(f"Warning: Unknown config '{config}', using 'all'")
        config = 'all'

    cfg = configs[config]
    print(f"\n{'='*60}")
    print(f"Abnormal Definition: {config.upper()}")
    print(f"{'='*60}")
    print(f"Description: {cfg['desc']}")
    print(f"Selected symbols: {cfg['selected']}")
    print(f"Normal symbols: {cfg['normal']}")
    print(f"Abnormal symbols: {cfg['abnormal']}")
    print(f"{'='*60}\n")

    return cfg['selected'], cfg['normal'], cfg['abnormal'], cfg['desc']


def get_available_configs():
    """Return list of available abnormal definition configurations."""
    return ['all', 'ventricular', 'atrial', 'bundle_branch', 'premature', 'custom']


def print_config_help():
    """Print detailed help about abnormal definition configurations."""
    print("\n" + "="*70)
    print("ABNORMAL DEFINITION CONFIGURATIONS")
    print("="*70)
    print("""
This experiment allows you to define which beat types are considered
"abnormal" (label=1). This is crucial because:

1. Different abnormalities have different clinical significance
2. Some may be easier/harder to predict than others
3. Results help understand which patterns the model can learn

Available Configurations:
------------------------

1. all (default)
   - Abnormal: L, R, V, A, F (everything except N)
   - Use case: General abnormality detection
   - Challenge: Mixed difficulty, diverse patterns

2. ventricular
   - Abnormal: V, F (ventricular origin)
   - Use case: Focus on most dangerous arrhythmias
   - Clinical: Ventricular abnormalities can be life-threatening
   - Result: Higher precision but still challenging

3. atrial
   - Abnormal: A (atrial premature beats)
   - Use case: Atrial arrhythmia detection
   - Clinical: Usually less dangerous than ventricular

4. bundle_branch
   - Abnormal: L, R (bundle branch blocks)
   - Use case: Conduction system disorders
   - Clinical: May indicate underlying heart disease

5. premature
   - Abnormal: V, A (premature beats)
   - Use case: Ectopic beat prediction
   - Clinical: Extra beats from abnormal locations

6. custom
   - Abnormal: [] (specify your own)
   - Use case: Research and experimentation

Example Usage:
-------------
python train.py --abnormal_config ventricular --seq_len 7
python train.py --abnormal_config all --seq_len 5

Custom Usage:
------------
python train.py --abnormal_config custom --abnormal_symbols V F A

Experimental Findings:
--------------------
From the original transformer_model.ipynb:

- Config: all
  Accuracy: 37.88%
  Precision: 3.25%  ← Very low! Hard to predict
  Recall: 87.31%    ← Model predicts many abnormals

- Config: ventricular
  Precision improved (but still low)
  Focus on dangerous cases helps slightly

Key Insight:
-----------
The low precision across all configurations shows that predicting
abnormalities BEFORE they appear (true prediction) is extremely
challenging. The model tends to over-predict abnormalities rather
than under-predict, which is clinically safer but reduces precision.

""")
    print("="*70 + "\n")
