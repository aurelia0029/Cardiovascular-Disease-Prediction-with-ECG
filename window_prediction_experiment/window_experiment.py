"""
Window Prediction Experiment - Analyze model performance vs prediction horizon.

This script trains CNN-LSTM models with different prediction windows (w_before)
to analyze how far in advance the model can predict abnormal heartbeats.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import json
from tqdm import tqdm

from data_loader import load_record_list, extract_beat_windows
from dataset import BeatSequenceDataset
from model import CNNLSTMClassifier


def create_sequences_with_offset(beats, labels, sequence_length=3, w_before=1):
    """
    Create sequences with flexible prediction offset.

    Args:
        beats (np.ndarray): ECG beat windows
        labels (np.ndarray): Binary labels (0 or 1)
        sequence_length (int): Number of beats in input sequence
        w_before (int): How many beats ahead to predict

    Returns:
        tuple: (X_seq, y_seq) arrays
    """
    X_seq, y_seq = [], []
    for i in range(len(beats) - sequence_length - w_before + 1):
        X_seq.append(beats[i:i + sequence_length])
        y_seq.append(labels[i + sequence_length + w_before - 1])
    return np.array(X_seq), np.array(y_seq)


def balance_dataset(X, y, random_seed=42):
    """Balance dataset by undersampling majority class."""
    X = np.array(X)
    y = np.array(y)

    normal_idx = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]
    min_count = min(len(normal_idx), len(abnormal_idx))

    np.random.seed(random_seed)
    selected_normal = np.random.choice(normal_idx, min_count, replace=False)
    selected_abnormal = np.random.choice(abnormal_idx, min_count, replace=False)

    balanced_idx = np.concatenate([selected_normal, selected_abnormal])
    np.random.shuffle(balanced_idx)

    return X[balanced_idx], y[balanced_idx]


def train_single_model(X_train, y_train, X_val, y_val, device, num_epochs=25, patience=5):
    """Train a single CNN-LSTM model with early stopping."""
    train_loader = DataLoader(BeatSequenceDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(BeatSequenceDataset(X_val, y_val), batch_size=32, shuffle=False)

    model = CNNLSTMClassifier(beat_len=X_train.shape[2]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_model(model, X_test, y_test, device):
    """Evaluate model and return metrics."""
    test_loader = DataLoader(BeatSequenceDataset(X_test, y_test), batch_size=32, shuffle=False)

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (outputs >= 0.5).float().cpu().numpy().flatten()
            y_pred.extend(preds)
            y_true.extend(targets.numpy().flatten())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }


def run_window_experiment(args):
    """Run window prediction experiment with configurable parameters."""

    print(f"Window Prediction Experiment")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Window range: {args.min_window} to {args.max_window} (step: {args.step})")
    print(f"Half window size: {args.half_window_size}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading MIT-BIH dataset...")
    record_list = load_record_list(args.data_dir)

    selected_symbols = ['N', 'L', 'R', 'V', 'A', 'F']
    normal_symbols = ['N']

    beat_windows, beat_labels = extract_beat_windows(
        args.data_dir, record_list, selected_symbols, normal_symbols,
        window_size=args.half_window_size
    )
    print(f"Extracted beats: {beat_windows.shape}")
    print(f"Label distribution: {Counter(beat_labels)}\n")

    # Generate window values
    w_before_list = list(range(args.min_window, args.max_window + 1, args.step))
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Experiment loop
    for w_before in tqdm(w_before_list, desc="Window experiments"):
        print(f"\n{'='*60}")
        print(f"w_before = {w_before} beats")
        print(f"{'='*60}")

        # Create sequences
        X_seq, y_seq = create_sequences_with_offset(
            beat_windows, beat_labels, args.seq_len, w_before
        )
        print(f"Sequence shape: {X_seq.shape}")
        print(f"Label distribution: {Counter(y_seq)}")

        # Split data
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=args.seed
        )

        # Balance training data
        X_balanced, y_balanced = balance_dataset(X_trainval, y_trainval, args.seed)

        X_train, X_val, y_train, y_val = train_test_split(
            X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=args.seed
        )

        # Train model
        print(f"Training model...")
        model = train_single_model(
            X_train, y_train, X_val, y_val, device,
            num_epochs=args.epochs, patience=args.patience
        )

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, device)

        print(f"✅ Results:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1 Score:  {metrics['f1']:.4f}")

        results.append({
            'w_before': w_before,
            **metrics
        })

    # Save results
    output_file = f"window_results_seq{args.seq_len}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Experiment completed!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Window Prediction Experiment - Analyze prediction horizon effects'
    )

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../../mitdb',
                        help='Path to MIT-BIH dataset directory')
    parser.add_argument('--half_window_size', type=int, default=128,
                        help='Half window size for beat extraction (default: 128, total 256 samples)')

    # Experiment parameters
    parser.add_argument('--seq_len', type=int, default=3,
                        help='Sequence length (number of beats in input)')
    parser.add_argument('--min_window', type=int, default=1,
                        help='Minimum prediction window (beats ahead)')
    parser.add_argument('--max_window', type=int, default=1200,
                        help='Maximum prediction window (beats ahead)')
    parser.add_argument('--step', type=int, default=100,
                        help='Step size for window values')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=25,
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Run experiment
    results = run_window_experiment(args)


if __name__ == "__main__":
    main()
