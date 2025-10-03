"""
Personalized Patient Models for VF Prediction

Motivation:
Previous experiments (train_scdh_only.py, train_scdh_nsr.py) built generalized models
using data from multiple patients. This experiment investigates whether personalized
models trained on an individual patient's historical data can better predict their
future VF onset.

Approach:
- For EACH patient individually:
  - Use 2/3 of their data for training
  - Use 1/3 of their data for testing
  - Build a patient-specific prediction model
- Compare personalized model performance to generalized models

This explores whether individual physiological patterns are more predictive than
population-level patterns for VF prediction.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (classification_report, accuracy_score,
                           confusion_matrix, roc_auc_score, roc_curve,
                           precision_recall_fscore_support)

from data_loader import load_scdh_data, prepare_dataframe, VF_ONSET_TIMES


# Configuration
CONFIG = {
    'DATA_DIR': None,  # Auto-detect SCDH database path
    'OUTPUT_DIR': './personalized_models_output',
    'TRAIN_RATIO': 2/3,
    'TEST_RATIO': 1/3,
    'RANDOM_SEED': 42,
    'SEGMENT_LEN': 3,
    'WINDOW_LEN': 180,
    'TIME_BEFORE_ONSET': 20,
    'TARGET_FS': 250,
    'SKIP_LIST': ['40', '42', '49'],  # Paced or no VF
    'MIN_SEGMENTS_PER_CLASS': 5  # Minimum segments needed per class to train
}


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def load_scdh_by_patient(data_dir=None):
    """
    Load SCDH data organized by patient.

    Returns:
        dict: {patient_id: {'X': features, 'y': labels}}
    """
    print("Loading SCDH data organized by patient...")

    # Use load_scdh_data from data_loader
    X, y = load_scdh_data(
        data_dir=data_dir,
        segment_len=CONFIG['SEGMENT_LEN'],
        window_len=CONFIG['WINDOW_LEN'],
        time_before_onset=CONFIG['TIME_BEFORE_ONSET'],
        target_fs=CONFIG['TARGET_FS'],
        skip_list=CONFIG['SKIP_LIST']
    )

    # For now, we need to modify load_scdh_data to return patient IDs
    # As a workaround, we'll create a modified version
    print("\nNote: Using modified data loading to track patient IDs...")

    patient_data = load_scdh_data_with_patient_ids(data_dir)

    return patient_data


def load_scdh_data_with_patient_ids(data_dir=None):
    """
    Load SCDH data with patient ID tracking.

    Returns:
        dict: {patient_id: {'X': list of features, 'y': list of labels}}
    """
    import wfdb
    from datetime import timedelta
    from scipy.signal import resample
    from feature_extraction import extract_features

    if data_dir is None:
        # Auto-detect path
        for path in ['../../sddb', '../sddb', './sddb', '/app/sddb']:
            if os.path.exists(path):
                data_dir = path
                break

    if data_dir is None or not os.path.exists(data_dir):
        raise FileNotFoundError(f"SCDH database not found. Please specify data_dir.")

    patient_data = {}
    ari_files = [f for f in os.listdir(data_dir) if f.endswith('.ari')]

    print(f"Loading SCDH data from {data_dir}")
    print(f"Extracting {CONFIG['WINDOW_LEN']}s windows {CONFIG['TIME_BEFORE_ONSET']} min before/after VF onset")

    for f in tqdm(ari_files, desc="Processing SCDH records"):
        record = f.replace('.ari', '')
        if record in CONFIG['SKIP_LIST'] or record not in VF_ONSET_TIMES:
            continue

        patient_data[record] = {'X': [], 'y': []}

        try:
            # Load annotation and signal
            ann = wfdb.rdann(os.path.join(data_dir, record), 'ari')
            sig, fields = wfdb.rdsamp(os.path.join(data_dir, record), channels=[0])
            fs = fields['fs']
            ecg = sig.flatten()

            # Resample if needed
            if fs != CONFIG['TARGET_FS']:
                ecg = resample(ecg, int(len(ecg) * CONFIG['TARGET_FS'] / fs))
                fs = CONFIG['TARGET_FS']
                ann_samples = (ann.sample * CONFIG['TARGET_FS'] / fields['fs']).astype(int)
            else:
                ann_samples = ann.sample

            # Parse VF onset time
            onset_time = VF_ONSET_TIMES[record]
            h, m, s = map(int, onset_time.split(':'))
            onset_sample = int(timedelta(hours=h, minutes=m, seconds=s).total_seconds() * fs)

            # Find VF start and end indices
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

            # Extract ONSET segment
            onset_start = first_v_sample - CONFIG['TIME_BEFORE_ONSET'] * 60 * fs
            onset_end = onset_start + CONFIG['WINDOW_LEN'] * fs
            if onset_start >= 0 and onset_end < len(ecg):
                onset_window = ecg[int(onset_start):int(onset_end)]
                for i in range(0, len(onset_window), int(CONFIG['SEGMENT_LEN'] * fs)):
                    seg = onset_window[i:i + int(CONFIG['SEGMENT_LEN'] * fs)]
                    if len(seg) == int(CONFIG['SEGMENT_LEN'] * fs):
                        if np.isnan(seg).any() or np.std(seg) == 0:
                            continue
                        features = extract_features(seg, fs)
                        if not any(np.isnan(val) for val in features.values()):
                            patient_data[record]['X'].append(features)
                            patient_data[record]['y'].append(1)  # Onset

            # Extract NORMAL segment
            normal_start = last_v_sample + CONFIG['TIME_BEFORE_ONSET'] * 60 * fs
            normal_end = normal_start + CONFIG['WINDOW_LEN'] * fs
            if normal_end < len(ecg):
                normal_window = ecg[int(normal_start):int(normal_end)]
                for i in range(0, len(normal_window), int(CONFIG['SEGMENT_LEN'] * fs)):
                    seg = normal_window[i:i + int(CONFIG['SEGMENT_LEN'] * fs)]
                    if len(seg) == int(CONFIG['SEGMENT_LEN'] * fs):
                        if np.isnan(seg).any() or np.std(seg) == 0:
                            continue
                        features = extract_features(seg, fs)
                        if not any(np.isnan(val) for val in features.values()):
                            patient_data[record]['X'].append(features)
                            patient_data[record]['y'].append(0)  # Normal

        except Exception as e:
            print(f"[{record}] Error: {e}")

    # Print summary
    total_segments = sum(len(data['y']) for data in patient_data.values())
    total_onset = sum(sum(data['y']) for data in patient_data.values())
    print(f"\nLoaded {len(patient_data)} patients, {total_segments} segments total")
    print(f"  Onset segments: {total_onset}")
    print(f"  Normal segments: {total_segments - total_onset}")

    return patient_data


def create_patient_train_test_split(X, y, train_ratio=2/3, random_seed=42):
    """
    Create train/test split for a single patient with stratification.

    Parameters:
    -----------
    X : list
        Feature dictionaries
    y : list
        Labels
    train_ratio : float
        Proportion for training
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    X = np.array(X)
    y = np.array(y)

    # Stratified split by class
    np.random.seed(random_seed)

    # Separate by class
    onset_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]

    # Shuffle
    np.random.shuffle(onset_idx)
    np.random.shuffle(normal_idx)

    # Split each class
    onset_split = int(len(onset_idx) * train_ratio)
    normal_split = int(len(normal_idx) * train_ratio)

    train_idx = np.concatenate([onset_idx[:onset_split], normal_idx[:normal_split]])
    test_idx = np.concatenate([onset_idx[onset_split:], normal_idx[normal_split:]])

    # Shuffle combined indices
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def train_patient_model(X_train, y_train):
    """
    Train a personalized model for one patient.

    Returns:
        Trained classifier or None if training fails
    """
    # Check if we have enough data and both classes
    unique_classes = np.unique(y_train)
    onset_count = np.sum(y_train == 1)
    normal_count = np.sum(y_train == 0)

    if len(unique_classes) < 2:
        return None, f"Only {len(unique_classes)} class in training data"

    if onset_count < CONFIG['MIN_SEGMENTS_PER_CLASS'] or normal_count < CONFIG['MIN_SEGMENTS_PER_CLASS']:
        return None, f"Insufficient data: {normal_count} normal, {onset_count} onset (need ≥{CONFIG['MIN_SEGMENTS_PER_CLASS']})"

    # Convert feature dicts to array
    if isinstance(X_train[0], dict):
        X_train_array = np.array([list(x.values()) for x in X_train])
    else:
        X_train_array = X_train

    try:
        # Create ensemble classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=CONFIG['RANDOM_SEED'], n_jobs=1)
        hgb = HistGradientBoostingClassifier(random_state=CONFIG['RANDOM_SEED'])
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=CONFIG['RANDOM_SEED']))

        ensemble_clf = VotingClassifier([
            ('rf', rf),
            ('hgb', hgb),
            ('lr', lr)
        ], voting='soft')

        ensemble_clf.fit(X_train_array, y_train)

        return ensemble_clf, None

    except Exception as e:
        return None, f"Training error: {str(e)}"


def evaluate_patient_model(clf, X_test, y_test):
    """
    Evaluate patient model on test data.

    Returns:
        dict: Evaluation metrics
    """
    # Convert feature dicts to array
    if isinstance(X_test[0], dict):
        X_test_array = np.array([list(x.values()) for x in X_test])
    else:
        X_test_array = X_test

    try:
        y_pred = clf.predict(X_test_array)
        y_prob = clf.predict_proba(X_test_array)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'test_size': len(y_test),
            'onset_count': int(np.sum(y_test == 1)),
            'normal_count': int(np.sum(y_test == 0))
        }

        # Detailed metrics if both classes present
        if len(np.unique(y_test)) == 2:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])

            results.update({
                'roc_auc': roc_auc,
                'normal_precision': report['0']['precision'],
                'normal_recall': report['0']['recall'],
                'normal_f1': report['0']['f1-score'],
                'onset_precision': report['1']['precision'],
                'onset_recall': report['1']['recall'],
                'onset_f1': report['1']['f1-score'],
                'both_classes': True,
                'y_pred': y_pred.tolist(),
                'y_prob': y_prob.tolist(),
                'y_test': y_test.tolist()
            })
        else:
            results['both_classes'] = False

        return results, None

    except Exception as e:
        return None, f"Evaluation error: {str(e)}"


def run_personalized_experiments(patient_data):
    """
    Run personalized model experiments for all patients.

    Returns:
        dict: Results for all patients
    """
    print(f"\n{'='*80}")
    print("Running personalized model experiments...")
    print(f"{'='*80}")

    all_results = {
        'successful': [],
        'failed': [],
        'patient_results': {}
    }

    for patient_id in tqdm(sorted(patient_data.keys()), desc="Training patient models"):
        print(f"\n[Patient {patient_id}]")

        X = patient_data[patient_id]['X']
        y = patient_data[patient_id]['y']

        onset_total = sum(y)
        normal_total = len(y) - onset_total

        print(f"  Total segments: {len(y)} ({normal_total} normal, {onset_total} onset)")

        # Create train/test split
        X_train, X_test, y_train, y_test = create_patient_train_test_split(
            X, y, train_ratio=CONFIG['TRAIN_RATIO'], random_seed=CONFIG['RANDOM_SEED']
        )

        onset_train = int(np.sum(y_train == 1))
        normal_train = int(np.sum(y_train == 0))
        onset_test = int(np.sum(y_test == 1))
        normal_test = int(np.sum(y_test == 0))

        print(f"  Train: {len(y_train)} ({normal_train} normal, {onset_train} onset)")
        print(f"  Test: {len(y_test)} ({normal_test} normal, {onset_test} onset)")

        # Train model
        clf, error = train_patient_model(X_train, y_train)

        if clf is None:
            print(f"  ❌ Training failed: {error}")
            all_results['failed'].append(patient_id)
            all_results['patient_results'][patient_id] = {
                'status': 'failed',
                'error': error,
                'train_size': len(y_train),
                'test_size': len(y_test)
            }
            continue

        print(f"  ✓ Model trained successfully")

        # Evaluate model
        eval_results, error = evaluate_patient_model(clf, X_test, y_test)

        if eval_results is None:
            print(f"  ❌ Evaluation failed: {error}")
            all_results['failed'].append(patient_id)
            all_results['patient_results'][patient_id] = {
                'status': 'eval_failed',
                'error': error
            }
            continue

        print(f"  ✓ Accuracy: {eval_results['accuracy']:.4f}")
        if eval_results.get('both_classes', False):
            print(f"  ✓ ROC-AUC: {eval_results['roc_auc']:.4f}")
            print(f"  ✓ Onset F1: {eval_results['onset_f1']:.4f}")

        all_results['successful'].append(patient_id)
        all_results['patient_results'][patient_id] = {
            'status': 'success',
            'train_size': len(y_train),
            'train_onset': onset_train,
            'train_normal': normal_train,
            'test_size': len(y_test),
            'test_onset': onset_test,
            'test_normal': normal_test,
            'metrics': eval_results
        }

    return all_results


def plot_patient_results(all_results, output_dir):
    """
    Generate visualization plots for patient results.
    """
    print("\nGenerating result plots...")

    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    successful = all_results['successful']

    if not successful:
        print("No successful patients to plot")
        return

    # Extract metrics
    accuracies = []
    roc_aucs = []
    onset_f1s = []
    patient_ids = []

    for patient_id in successful:
        result = all_results['patient_results'][patient_id]
        metrics = result['metrics']

        patient_ids.append(patient_id)
        accuracies.append(metrics['accuracy'])

        if metrics.get('both_classes', False):
            roc_aucs.append(metrics['roc_auc'])
            onset_f1s.append(metrics['onset_f1'])

    # Plot 1: Accuracy bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(patient_ids)), accuracies, color='steelblue', alpha=0.7)
    ax.set_xlabel('Patient ID', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Personalized Model Accuracy per Patient', fontsize=14)
    ax.set_xticks(range(len(patient_ids)))
    ax.set_xticklabels(patient_ids, rotation=45)
    ax.axhline(y=np.mean(accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(accuracies):.3f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'accuracy_by_patient.png'), dpi=300)
    plt.close()

    # Plot 2: ROC-AUC if available
    if roc_aucs:
        fig, ax = plt.subplots(figsize=(12, 6))
        patient_ids_with_both = [pid for pid in patient_ids
                                  if all_results['patient_results'][pid]['metrics'].get('both_classes', False)]
        ax.bar(range(len(patient_ids_with_both)), roc_aucs, color='coral', alpha=0.7)
        ax.set_xlabel('Patient ID', fontsize=12)
        ax.set_ylabel('ROC-AUC', fontsize=12)
        ax.set_title('Personalized Model ROC-AUC per Patient', fontsize=14)
        ax.set_xticks(range(len(patient_ids_with_both)))
        ax.set_xticklabels(patient_ids_with_both, rotation=45)
        ax.axhline(y=np.mean(roc_aucs), color='red', linestyle='--', label=f'Mean: {np.mean(roc_aucs):.3f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'roc_auc_by_patient.png'), dpi=300)
        plt.close()

    print(f"✓ Plots saved to {plot_dir}")


def save_results(all_results, output_dir):
    """
    Save experiment results to files.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON summary
    summary = {
        'experiment': 'Personalized Patient Models',
        'timestamp': timestamp,
        'config': CONFIG,
        'successful_patients': all_results['successful'],
        'failed_patients': all_results['failed'],
        'patient_results': all_results['patient_results']
    }

    json_file = os.path.join(output_dir, f'personalized_results_{timestamp}.json')
    with open(json_file, 'w') as f:
        json.dump(convert_numpy_types(summary), f, indent=2)

    # Save pickle with full data
    pkl_file = os.path.join(output_dir, f'personalized_results_{timestamp}.pkl')
    with open(pkl_file, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n✓ Results saved:")
    print(f"  - JSON: {json_file}")
    print(f"  - Pickle: {pkl_file}")

    return json_file, pkl_file


def print_summary(all_results):
    """
    Print comprehensive results summary.
    """
    print(f"\n{'='*80}")
    print("PERSONALIZED MODEL EXPERIMENT RESULTS")
    print(f"{'='*80}")

    successful = all_results['successful']
    failed = all_results['failed']

    print(f"\nPatient Processing:")
    print(f"  Total patients: {len(successful) + len(failed)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    if successful:
        # Calculate aggregate statistics
        accuracies = []
        roc_aucs = []
        onset_recalls = []
        onset_precisions = []

        for patient_id in successful:
            metrics = all_results['patient_results'][patient_id]['metrics']
            accuracies.append(metrics['accuracy'])

            if metrics.get('both_classes', False):
                roc_aucs.append(metrics['roc_auc'])
                onset_recalls.append(metrics['onset_recall'])
                onset_precisions.append(metrics['onset_precision'])

        print(f"\nOverall Performance:")
        print(f"  Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"  Median Accuracy: {np.median(accuracies):.4f}")
        print(f"  Range: {np.min(accuracies):.4f} - {np.max(accuracies):.4f}")

        if roc_aucs:
            print(f"\n  Mean ROC-AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
            print(f"  Mean Onset Precision: {np.mean(onset_precisions):.4f}")
            print(f"  Mean Onset Recall: {np.mean(onset_recalls):.4f}")

        print(f"\nPer-Patient Results:")
        print(f"{'Patient':<10} {'Accuracy':<10} {'ROC-AUC':<10} {'Onset F1':<10} {'Train N/O':<12} {'Test N/O':<12}")
        print("-" * 80)

        for patient_id in sorted(successful):
            result = all_results['patient_results'][patient_id]
            metrics = result['metrics']

            acc = metrics['accuracy']
            roc = metrics.get('roc_auc', 0.0) if metrics.get('both_classes', False) else None
            f1 = metrics.get('onset_f1', 0.0) if metrics.get('both_classes', False) else None

            train_str = f"{result['train_normal']}/{result['train_onset']}"
            test_str = f"{result['test_normal']}/{result['test_onset']}"

            roc_str = f"{roc:.4f}" if roc is not None else "N/A"
            f1_str = f"{f1:.4f}" if f1 is not None else "N/A"

            print(f"{patient_id:<10} {acc:<10.4f} {roc_str:<10} {f1_str:<10} {train_str:<12} {test_str:<12}")

    if failed:
        print(f"\nFailed Patients:")
        for patient_id in sorted(failed):
            result = all_results['patient_results'][patient_id]
            print(f"  {patient_id}: {result.get('error', 'Unknown error')}")

    print(f"\n{'='*80}")


def main():
    """
    Main execution function.
    """
    print("="*80)
    print("PERSONALIZED PATIENT VF PREDICTION MODELS")
    print("="*80)
    print("\nMotivation:")
    print("  Previous experiments built generalized models from multiple patients.")
    print("  This experiment builds personalized models using each patient's")
    print("  historical data to predict their own future VF onset.")
    print("\nApproach:")
    print(f"  - For each patient: {CONFIG['TRAIN_RATIO']:.1%} train, {CONFIG['TEST_RATIO']:.1%} test")
    print("  - Independent models for each patient")
    print("  - Ensemble classifier (RF + HGB + LR)")
    print("="*80)

    # Set random seed
    np.random.seed(CONFIG['RANDOM_SEED'])

    # Create output directory
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

    # Step 1: Load data by patient
    print("\nStep 1: Loading SCDH data by patient...")
    patient_data = load_scdh_data_with_patient_ids(CONFIG['DATA_DIR'])

    # Step 2: Run experiments
    print("\nStep 2: Training personalized models...")
    all_results = run_personalized_experiments(patient_data)

    # Step 3: Plot results
    print("\nStep 3: Generating visualizations...")
    plot_patient_results(all_results, CONFIG['OUTPUT_DIR'])

    # Step 4: Save results
    print("\nStep 4: Saving results...")
    save_results(all_results, CONFIG['OUTPUT_DIR'])

    # Step 5: Print summary
    print_summary(all_results)

    print(f"\n✓ Experiment completed!")
    print(f"✓ Results saved to: {CONFIG['OUTPUT_DIR']}")


if __name__ == "__main__":
    main()
