"""
Training script for True Prediction Experiment.

This experiment tests whether models can truly predict abnormalities
BEFORE they appear, by filtering out all input sequences containing
abnormal beats.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import json

from data_loader import load_record_list, extract_beat_windows
from data_processing import create_pure_normal_sequences, define_abnormal_symbols, print_config_help
from dataset import BeatSequenceDataset
from model import TransformerClassifier


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


def train_model(model, train_loader, val_loader, device, num_epochs=100, patience=5, lr=0.001):
    """Train the transformer model with early stopping."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


def plot_training_curves(train_losses, val_losses, save_path='training_curve.png'):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss - True Prediction Experiment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Training curve saved to {save_path}")


def main():
    """Main training pipeline."""

    parser = argparse.ArgumentParser(
        description='True Prediction Experiment - Predict abnormalities before they appear',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default (all abnormalities)
  python train.py

  # Focus on ventricular abnormalities only
  python train.py --abnormal_config ventricular

  # Custom sequence length
  python train.py --seq_len 5 --abnormal_config premature

  # Show help about abnormal configurations
  python train.py --show_configs

  # Custom abnormal symbols
  python train.py --abnormal_config custom --abnormal_symbols V F
        """
    )

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../dataset/mitdb',
                        help='Path to MIT-BIH dataset directory')
    parser.add_argument('--half_window_size', type=int, default=128,
                        help='Half window size for beat extraction (default: 128, total 256 samples)')

    # Abnormal definition
    parser.add_argument('--abnormal_config', type=str, default='all',
                        choices=['all', 'ventricular', 'atrial', 'bundle_branch', 'premature', 'custom'],
                        help='Configuration for defining abnormal beats')
    parser.add_argument('--abnormal_symbols', nargs='+', default=None,
                        help='Custom abnormal symbols (only used with --abnormal_config custom)')
    parser.add_argument('--show_configs', action='store_true',
                        help='Show detailed help about abnormal configurations and exit')

    # Model parameters
    parser.add_argument('--seq_len', type=int, default=7,
                        help='Sequence length (number of beats in input)')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Transformer embedding dimension')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of transformer layers')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.3,
                        help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Show config help and exit if requested
    if args.show_configs:
        print_config_help()
        return

    print(f"\n{'='*70}")
    print(f"TRUE PREDICTION EXPERIMENT")
    print(f"{'='*70}")
    print(f"Task: Predict abnormalities BEFORE they appear")
    print(f"Method: Filter out all sequences with abnormal beats in input")
    print(f"{'='*70}\n")

    # Define abnormal symbols based on configuration
    selected_symbols, normal_symbols, abnormal_symbols, desc = define_abnormal_symbols(args.abnormal_config)

    # Handle custom abnormal symbols
    if args.abnormal_config == 'custom' and args.abnormal_symbols:
        abnormal_symbols = args.abnormal_symbols
        selected_symbols = normal_symbols + abnormal_symbols
        print(f"Custom abnormal symbols: {abnormal_symbols}")

    print(f"\nExperiment Configuration:")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Abnormal definition: {desc}")
    print(f"  Half window size: {args.half_window_size}")

    # Load data
    print(f"\nLoading MIT-BIH dataset from: {args.data_dir}")
    record_list = load_record_list(args.data_dir)
    print(f"Found {len(record_list)} records")

    # Extract beat windows
    beat_windows, beat_labels = extract_beat_windows(
        args.data_dir, record_list, selected_symbols, normal_symbols,
        window_size=args.half_window_size
    )
    print(f"\nExtracted beats: {beat_windows.shape}")
    print(f"Overall label distribution: {Counter(beat_labels)}")

    # Create pure normal sequences (KEY INNOVATION)
    print(f"\n{'='*70}")
    print(f"FILTERING TO PURE NORMAL INPUT SEQUENCES")
    print(f"{'='*70}")
    print(f"Removing any sequence where input contains abnormal beats...")

    X_seq, y_seq = create_pure_normal_sequences(
        beat_windows, beat_labels, args.seq_len, abnormal_symbols
    )

    print(f"\nFinal sequence shape: {X_seq.shape}")
    print(f"This is a MUCH HARDER task than standard classification!")

    # Split data
    print(f"\n{'='*70}")
    print(f"DATA SPLITTING")
    print(f"{'='*70}")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=args.seed
    )

    # Balance training data
    print(f"\nBalancing training data...")
    X_balanced, y_balanced = balance_dataset(X_trainval, y_trainval, args.seed)
    print(f"Balanced data: {X_balanced.shape}")
    print(f"Balanced distribution: {Counter(y_balanced)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_balanced, y_balanced, test_size=args.val_split,
        stratify=y_balanced, random_state=args.seed
    )

    print(f"\nFinal splits:")
    print(f"  Train: {X_train.shape} - {Counter(y_train)}")
    print(f"  Val:   {X_val.shape} - {Counter(y_val)}")
    print(f"  Test:  {X_test.shape} - {Counter(y_test)}")

    # Create data loaders
    train_dataset = BeatSequenceDataset(X_train, y_train)
    val_dataset = BeatSequenceDataset(X_val, y_val)
    test_dataset = BeatSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"MODEL INITIALIZATION")
    print(f"{'='*70}")
    print(f"Device: {device}")

    model = TransformerClassifier(
        seq_len=args.seq_len,
        beat_len=X_train.shape[2],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TransformerClassifier")
    print(f"Total parameters: {total_params:,}")

    # Train model
    print(f"\n{'='*70}")
    print(f"TRAINING")
    print(f"{'='*70}")

    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, device,
        num_epochs=args.epochs, patience=args.patience, lr=args.lr
    )

    # Plot training curves
    plot_training_curves(train_losses, val_losses)

    # Save model and metadata
    model_path = f"transformer_model_{args.abnormal_config}_seq{args.seq_len}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Model saved to {model_path}")

    # Save test data and metadata
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'abnormal_config': args.abnormal_config,
        'abnormal_symbols': abnormal_symbols,
        'selected_symbols': selected_symbols,
        'normal_symbols': normal_symbols,
        'seq_len': args.seq_len,
        'beat_len': X_train.shape[2]
    }
    np.save('test_data.npy', test_data)
    print(f"✅ Test data saved to test_data.npy")

    # Save configuration
    config = {
        'abnormal_config': args.abnormal_config,
        'abnormal_description': desc,
        'selected_symbols': selected_symbols,
        'normal_symbols': normal_symbols,
        'abnormal_symbols': abnormal_symbols,
        'seq_len': args.seq_len,
        'beat_len': int(X_train.shape[2]),
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'total_sequences': int(len(X_seq)),
        'train_size': int(len(X_train)),
        'val_size': int(len(X_val)),
        'test_size': int(len(X_test))
    }

    with open('experiment_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Configuration saved to experiment_config.json")

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nNext step: Run evaluation")
    print(f"  python evaluate.py")


if __name__ == "__main__":
    main()
