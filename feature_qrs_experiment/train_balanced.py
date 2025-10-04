"""
Experiment 1: Fully Balanced Dataset

Strategy:
1. Extract abnormal (V, E, F, !) and normal (N) beats with clean 30-second history
2. Split abnormal into train/val/test (6:2:2)
3. Sample equal number of normal beats for each split
4. Train SimpleFNN on balanced datasets

All splits (train, val, test) are balanced.
"""

import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

from data_loader import load_features_with_clean_history, split_and_balance
from model import SimpleFNN, SimpleFNNTrainer


def train_and_evaluate(args):
    """Main training and evaluation pipeline."""

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 1: FULLY BALANCED DATASET")
    print(f"{'='*70}\n")

    # Load data
    print("Loading data with clean 30-second history requirement...")
    abnormal_features, abnormal_labels, normal_features = load_features_with_clean_history(
        data_dir=args.data_dir,
        pre_window_sec=30,
        abnormal_symbols=['V', 'E', 'F', '!']
    )

    if len(abnormal_features) == 0 or len(normal_features) == 0:
        print("Error: Insufficient data!")
        return

    # Split and balance
    print("Splitting and balancing data...")
    data_splits = split_and_balance(
        abnormal_features, abnormal_labels, normal_features,
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        balance_test=True,  # ← KEY: Test set is also balanced
        random_state=args.random_state
    )

    # Print dataset info
    print(f"\n{'='*70}")
    print(f"Dataset Summary (Fully Balanced)")
    print(f"{'='*70}")
    print(f"Training set:   {len(data_splits['X_train'])} samples")
    print(f"  - Abnormal: {int(data_splits['y_train'].sum())}")
    print(f"  - Normal:   {int(len(data_splits['y_train']) - data_splits['y_train'].sum())}")
    print(f"\nValidation set: {len(data_splits['X_val'])} samples")
    print(f"  - Abnormal: {int(data_splits['y_val'].sum())}")
    print(f"  - Normal:   {int(len(data_splits['y_val']) - data_splits['y_val'].sum())}")
    print(f"\nTest set:       {len(data_splits['X_test'])} samples")
    print(f"  - Abnormal: {int(data_splits['y_test'].sum())}")
    print(f"  - Normal:   {int(len(data_splits['y_test']) - data_splits['y_test'].sum())}")
    print(f"{'='*70}\n")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(data_splits['X_train']).float(),
        torch.from_numpy(data_splits['y_train']).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(data_splits['X_val']).float(),
        torch.from_numpy(data_splits['y_val']).float()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(data_splits['X_test']).float(),
        torch.from_numpy(data_splits['y_test']).float()
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    model = SimpleFNN(
        input_size=4,
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout
    )

    trainer = SimpleFNNTrainer(model, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"{'='*70}")
    print(f"Training SimpleFNN")
    print(f"{'='*70}\n")

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer)

        # Validate
        val_metrics = trainer.evaluate(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")

        # Early stopping
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model_balanced.pth')
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    print(f"\n{'='*70}")
    print(f"Best model from epoch {best_epoch} (Val Acc: {best_val_acc:.4f})")
    print(f"{'='*70}\n")

    model.load_state_dict(torch.load('best_model_balanced.pth'))
    test_metrics = trainer.evaluate(test_loader)

    # Print results
    print(f"{'='*70}")
    print(f"TEST SET RESULTS (Balanced)")
    print(f"{'='*70}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Loss:     {test_metrics['loss']:.4f}")
    print()

    # Classification report
    y_true = test_metrics['targets']
    y_pred = test_metrics['predictions']

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("\n[[TN  FP]")
    print(" [FN  TP]]")

    # Calculate detailed metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nDetailed Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Save results
    results = {
        'experiment': 'fully_balanced',
        'config': vars(args),
        'dataset_info': {
            'train_size': len(data_splits['X_train']),
            'val_size': len(data_splits['X_val']),
            'test_size': len(data_splits['X_test']),
            'test_balanced': True
        },
        'best_epoch': best_epoch,
        'best_val_accuracy': float(best_val_acc),
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'loss': float(test_metrics['loss']),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'confusion_matrix': cm.tolist()
    }

    with open('results_balanced.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to results_balanced.json")
    print(f"Model saved to best_model_balanced.pth")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 1: Fully Balanced Dataset (SimpleFNN)'
    )

    parser.add_argument('--data_dir', type=str, default='../dataset/mitdb',
                        help='Path to MIT-BIH database')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[16, 8],
                        help='Hidden layer sizes (default: 16 8)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum epochs (default: 100)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_state)

    train_and_evaluate(args)


if __name__ == "__main__":
    main()
