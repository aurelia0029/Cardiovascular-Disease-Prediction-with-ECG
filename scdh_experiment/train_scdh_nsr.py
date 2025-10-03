"""
Experiment 2: Training & Testing on SCDH & NSR

This experiment trains and evaluates models using BOTH databases:
- SCDH onset segments: {min} minutes BEFORE VF onset (label = 1)
- NSR normal segments: Normal sinus rhythm data (label = 0)

Uses Stratified 10-Fold Cross-Validation with ensemble classifiers.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json

from data_loader import load_scdh_data, load_nsr_data, prepare_dataframe


def train_and_evaluate(X, y, n_splits=10, random_state=42):
    """
    Perform stratified K-fold cross-validation with ensemble classifiers.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Label vector
    n_splits : int
        Number of folds for cross-validation
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    results : dict
        Dictionary containing metrics for each fold and averages
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Define base classifiers
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    gb = HistGradientBoostingClassifier(random_state=random_state)
    lr = LogisticRegression(max_iter=1000, random_state=random_state)

    # Create ensemble using soft voting
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
        voting='soft'
    )

    # Store results for each fold
    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'confusion_matrices': []
    }

    print(f"\n{'='*70}")
    print(f"10-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*70}\n")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train ensemble
        ensemble.fit(X_train, y_train)

        # Predict
        y_pred = ensemble.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Extract metrics from classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']

        # Store results
        fold_results['accuracy'].append(acc)
        fold_results['precision'].append(precision)
        fold_results['recall'].append(recall)
        fold_results['f1'].append(f1)
        fold_results['confusion_matrices'].append(cm.tolist())

        # Print fold results
        print(f"Fold {fold:2d}/{n_splits}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        print()

    # Calculate average metrics
    avg_results = {
        'accuracy_mean': np.mean(fold_results['accuracy']),
        'accuracy_std': np.std(fold_results['accuracy']),
        'precision_mean': np.mean(fold_results['precision']),
        'precision_std': np.std(fold_results['precision']),
        'recall_mean': np.mean(fold_results['recall']),
        'recall_std': np.std(fold_results['recall']),
        'f1_mean': np.mean(fold_results['f1']),
        'f1_std': np.std(fold_results['f1'])
    }

    # Print average results
    print(f"\n{'='*70}")
    print(f"AVERAGE RESULTS ACROSS {n_splits} FOLDS")
    print(f"{'='*70}")
    print(f"Accuracy:  {avg_results['accuracy_mean']:.4f} ± {avg_results['accuracy_std']:.4f}")
    print(f"Precision: {avg_results['precision_mean']:.4f} ± {avg_results['precision_std']:.4f}")
    print(f"Recall:    {avg_results['recall_mean']:.4f} ± {avg_results['recall_std']:.4f}")
    print(f"F1-Score:  {avg_results['f1_mean']:.4f} ± {avg_results['f1_std']:.4f}")
    print(f"{'='*70}\n")

    # Combine results
    results = {
        'fold_results': fold_results,
        'average_results': avg_results
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 2: Training & Testing on SCDH & NSR (10-Fold CV)'
    )

    # SCDH parameters
    parser.add_argument('--scdh_dir', type=str, default=None,
                        help='Path to SCDH database directory (default: auto-detect)')
    parser.add_argument('--min', type=int, default=20,
                        help='Minutes before VF onset to extract SCDH onset data (default: 20)')

    # NSR parameters
    parser.add_argument('--nsr_dir', type=str, default=None,
                        help='Path to NSR database directory (default: auto-detect)')
    parser.add_argument('--nsr_windows', type=int, default=60,
                        help='Number of windows to extract per NSR record (default: 60)')

    # Common parameters
    parser.add_argument('--segment_len', type=int, default=3,
                        help='Length of each segment in seconds (default: 3)')
    parser.add_argument('--window_len', type=int, default=180,
                        help='Length of extraction window in seconds (default: 180 = 3 min)')
    parser.add_argument('--target_fs', type=int, default=250,
                        help='Target sampling frequency (default: 250 Hz)')
    parser.add_argument('--n_splits', type=int, default=10,
                        help='Number of folds for cross-validation (default: 10)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output', type=str, default='scdh_nsr_results.json',
                        help='Output file for results (default: scdh_nsr_results.json)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 2: TRAINING & TESTING ON SCDH & NSR")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  SCDH directory: {args.scdh_dir}")
    print(f"  NSR directory: {args.nsr_dir}")
    print(f"  SCDH time window: {args.min} minutes before VF onset")
    print(f"  NSR windows per record: {args.nsr_windows}")
    print(f"  Segment length: {args.segment_len} seconds")
    print(f"  Window length: {args.window_len} seconds ({args.window_len//60} minutes)")
    print(f"  Sampling frequency: {args.target_fs} Hz")
    print(f"  Cross-validation: {args.n_splits}-fold")
    print()

    # Load SCDH onset data (only onset, not normal)
    print("Loading SCDH onset data...")
    X_scdh, y_scdh = load_scdh_data(
        data_dir=args.scdh_dir,
        segment_len=args.segment_len,
        window_len=args.window_len,
        time_before_onset=args.min,
        target_fs=args.target_fs
    )

    # Filter to keep only onset data (label = 1)
    X_scdh_onset = [x for x, label in zip(X_scdh, y_scdh) if label == 1]
    y_scdh_onset = [1] * len(X_scdh_onset)

    print(f"\nSCDH onset segments: {len(X_scdh_onset)}")

    # Load NSR normal data
    print("\nLoading NSR normal data...")
    X_nsr = load_nsr_data(
        data_dir=args.nsr_dir,
        segment_len=args.segment_len,
        window_len=args.window_len,
        num_windows=args.nsr_windows,
        target_fs=args.target_fs
    )
    y_nsr = [0] * len(X_nsr)

    print(f"NSR normal segments: {len(X_nsr)}")

    # Combine SCDH onset and NSR normal
    X_combined = X_scdh_onset + X_nsr
    y_combined = y_scdh_onset + y_nsr

    print(f"\nTotal combined segments: {len(X_combined)}")
    print(f"  Onset (SCDH): {sum(y_combined)}")
    print(f"  Normal (NSR): {len(y_combined) - sum(y_combined)}")

    # Convert to DataFrame
    df = prepare_dataframe(X_combined, y_combined)

    # Separate features and labels
    X_df = df.drop(columns=['label'])
    y_df = df['label']

    # Train and evaluate
    results = train_and_evaluate(
        X_df, y_df,
        n_splits=args.n_splits,
        random_state=args.random_state
    )

    # Save results
    output_data = {
        'experiment': 'scdh_nsr',
        'configuration': vars(args),
        'data_info': {
            'total_samples': len(X_combined),
            'scdh_onset_samples': len(X_scdh_onset),
            'nsr_normal_samples': len(X_nsr),
            'feature_names': list(X_combined[0].keys()) if X_combined else []
        },
        'results': results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
