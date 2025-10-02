"""
Evaluation script for CNN-LSTM ECG arrhythmia classifier.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from dataset import BeatSequenceDataset
from model import CNNLSTMClassifier


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set.

    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: torch.device

    Returns:
        tuple: (y_true, y_pred) arrays
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (outputs >= 0.5).float().cpu().numpy().flatten()
            y_pred.extend(preds)
            y_true.extend(targets.numpy().flatten())

    return np.array(y_true), np.array(y_pred)


def print_metrics(y_true, y_pred):
    """
    Calculate and print evaluation metrics.

    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("\n" + "="*50)
    print("TEST METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\n[[TN  FP]")
    print(" [FN  TP]]")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"]))


def main():
    """Main evaluation pipeline."""

    BATCH_SIZE = 32
    MODEL_PATH = "cnn_lstm_best_model.pth"
    TEST_DATA_PATH = "test_data.npy"

    # Load test data
    print("Loading test data...")
    test_data = np.load(TEST_DATA_PATH, allow_pickle=True).item()
    X_test = test_data['X_test']
    y_test = test_data['y_test']

    print(f"Test data shape: {X_test.shape}")

    # Create test loader
    test_dataset = BeatSequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CNNLSTMClassifier(beat_len=X_test.shape[2]).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Model loaded from {MODEL_PATH}")

    # Evaluate
    print("\nEvaluating model...")
    y_true, y_pred = evaluate_model(model, test_loader, device)

    # Print metrics
    print_metrics(y_true, y_pred)


if __name__ == "__main__":
    main()
