"""
Experiment 1: Training & Testing on SCDH Only

This experiment trains and evaluates models using ONLY the SCDH database.
Data consists of:
- Onset segments: Extracted {min} minutes BEFORE VF onset (label = 1)
- Normal segments: Extracted {min} minutes AFTER VF ends (label = 0)

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

from data_loader import load_scdh_data, prepare_dataframe


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
        description='Experiment 1: Training & Testing on SCDH Only (10-Fold CV)'
    )

    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to SCDH database directory (default: auto-detect)')
    parser.add_argument('--min', type=int, default=20,
                        help='Minutes before/after VF onset to extract data (default: 20)')
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
    parser.add_argument('--output', type=str, default='scdh_only_results.json',
                        help='Output file for results (default: scdh_only_results.json)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 1: TRAINING & TESTING ON SCDH ONLY")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Time window: {args.min} minutes before/after VF onset")
    print(f"  Segment length: {args.segment_len} seconds")
    print(f"  Window length: {args.window_len} seconds ({args.window_len//60} minutes)")
    print(f"  Sampling frequency: {args.target_fs} Hz")
    print(f"  Cross-validation: {args.n_splits}-fold")
    print()

    # Load SCDH data
    X, y = load_scdh_data(
        data_dir=args.data_dir,
        segment_len=args.segment_len,
        window_len=args.window_len,
        time_before_onset=args.min,
        target_fs=args.target_fs
    )

    # Convert to DataFrame
    df = prepare_dataframe(X, y)

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
        'experiment': 'scdh_only',
        'configuration': vars(args),
        'data_info': {
            'total_samples': len(X),
            'onset_samples': int(sum(y)),
            'normal_samples': int(len(y) - sum(y)),
            'feature_names': list(X[0].keys()) if X else []
        },
        'results': results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
