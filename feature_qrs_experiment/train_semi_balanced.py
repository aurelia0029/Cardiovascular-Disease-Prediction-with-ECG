"""
Experiment 2: Semi-Balanced Dataset

Strategy:
1. Extract abnormal (V, E, F, !) and normal (N) beats with clean 30-second history
2. Split abnormal into train/val/test (6:2:2)
3. Sample equal number of normal beats for train/val ONLY
4. Test set: Use ALL abnormal from test split + ALL remaining normal (unbalanced)
5. Train SimpleFNN on balanced train/val, evaluate on realistic unbalanced test

Train/Val are balanced, but Test is unbalanced (realistic scenario).
"""

import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import json

from data_loader import load_features_with_clean_history, extract_qrs_features
from model import SimpleFNN, SimpleFNNTrainer


def split_semi_balanced(abnormal_features, abnormal_labels, normal_features,
                        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                        random_state=42):
    """
    Split with semi-balancing: train/val balanced, test unbalanced.

    Parameters:
    -----------
    abnormal_features : np.ndarray
        Abnormal feature array
    abnormal_labels : np.ndarray
        Abnormal labels
    normal_features : np.ndarray
        Normal feature array
    train_ratio, val_ratio, test_ratio : float
        Split ratios
    random_state : int
        Random seed

    Returns:
    --------
    Dictionary containing train/val/test splits
    """
    np.random.seed(random_state)

    # Split abnormal data (6:2:2)
    X_abn_train, X_abn_temp, y_abn_train, y_abn_temp = train_test_split(
        abnormal_features, abnormal_labels,
        test_size=(val_ratio + test_ratio), random_state=random_state
    )

    val_size = val_ratio / (val_ratio + test_ratio)
    X_abn_val, X_abn_test, y_abn_val, y_abn_test = train_test_split(
        X_abn_temp, y_abn_temp, test_size=(1 - val_size), random_state=random_state
    )

    # Sample balanced normal for train/val
    n_train_normal = len(X_abn_train)
    n_val_normal = len(X_abn_val)

    # Ensure we don't sample the same normals for train and val
    all_normal_indices = np.arange(len(normal_features))
    np.random.shuffle(all_normal_indices)

    train_normal_indices = all_normal_indices[:n_train_normal]
    val_normal_indices = all_normal_indices[n_train_normal:n_train_normal + n_val_normal]
    test_normal_indices = all_normal_indices[n_train_normal + n_val_normal:]  # All remaining

    X_norm_train = normal_features[train_normal_indices]
    X_norm_val = normal_features[val_normal_indices]
    X_norm_test = normal_features[test_normal_indices]  # ← All remaining (unbalanced)

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


def train_and_evaluate(args):
    """Main training and evaluation pipeline."""

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 2: SEMI-BALANCED DATASET")
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

    # Split with semi-balancing
    print("Splitting data (train/val balanced, test unbalanced)...")
    data_splits = split_semi_balanced(
        abnormal_features, abnormal_labels, normal_features,
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        random_state=args.random_state
    )

    # Print dataset info
    print(f"\n{'='*70}")
    print(f"Dataset Summary (Semi-Balanced)")
    print(f"{'='*70}")
    print(f"Training set (BALANCED):   {len(data_splits['X_train'])} samples")
    print(f"  - Abnormal: {int(data_splits['y_train'].sum())}")
    print(f"  - Normal:   {int(len(data_splits['y_train']) - data_splits['y_train'].sum())}")
    print(f"\nValidation set (BALANCED): {len(data_splits['X_val'])} samples")
    print(f"  - Abnormal: {int(data_splits['y_val'].sum())}")
    print(f"  - Normal:   {int(len(data_splits['y_val']) - data_splits['y_val'].sum())}")
    print(f"\nTest set (UNBALANCED):     {len(data_splits['X_test'])} samples")
    print(f"  - Abnormal: {int(data_splits['y_test'].sum())}")
    print(f"  - Normal:   {int(len(data_splits['y_test']) - data_splits['y_test'].sum())}")
    abnormal_ratio = data_splits['y_test'].sum() / len(data_splits['y_test']) * 100
    print(f"  - Abnormal ratio: {abnormal_ratio:.2f}%")
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
            torch.save(model.state_dict(), 'best_model_semi_balanced.pth')
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    print(f"\n{'='*70}")
    print(f"Best model from epoch {best_epoch} (Val Acc: {best_val_acc:.4f})")
    print(f"{'='*70}\n")

    model.load_state_dict(torch.load('best_model_semi_balanced.pth'))
    test_metrics = trainer.evaluate(test_loader)

    # Print results
    print(f"{'='*70}")
    print(f"TEST SET RESULTS (Unbalanced - Realistic)")
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

    print(f"\n{'='*70}")
    print(f"INTERPRETATION")
    print(f"{'='*70}")
    print(f"This experiment reflects a more realistic scenario:")
    print(f"- Training on balanced data (equal normal/abnormal)")
    print(f"- Testing on unbalanced data (many more normal than abnormal)")
    print(f"- Abnormal ratio in test: {abnormal_ratio:.2f}%")
    print(f"\nNote: Accuracy can be misleading with imbalanced test sets.")
    print(f"Focus on Precision, Recall, and F1-Score for better evaluation.")
    print(f"{'='*70}\n")

    # Save results
    results = {
        'experiment': 'semi_balanced',
        'config': vars(args),
        'dataset_info': {
            'train_size': len(data_splits['X_train']),
            'val_size': len(data_splits['X_val']),
            'test_size': len(data_splits['X_test']),
            'test_balanced': False,
            'test_abnormal_ratio': float(abnormal_ratio)
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

    with open('results_semi_balanced.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to results_semi_balanced.json")
    print(f"Model saved to best_model_semi_balanced.pth\n")


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 2: Semi-Balanced Dataset (SimpleFNN)'
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
