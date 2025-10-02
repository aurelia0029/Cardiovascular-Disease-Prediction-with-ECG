"""
Evaluation script for True Prediction Experiment.

Evaluates the trained transformer model on the test set and provides
detailed analysis of prediction performance.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import json

from dataset import BeatSequenceDataset
from model import TransformerClassifier


def evaluate_model(model, test_loader, device):
    """Evaluate model and return predictions."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = outputs.cpu().numpy().flatten()
            preds = (outputs >= 0.5).float().cpu().numpy().flatten()

            y_prob.extend(probs)
            y_pred.extend(preds)
            y_true.extend(targets.numpy().flatten())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Normal', 'Abnormal'],
           yticklabels=['Normal', 'Abnormal'],
           title='Confusion Matrix - True Prediction Experiment',
           ylabel='True label',
           xlabel='Predicted label')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)

    fig.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_prob, save_path='roc_curve.png'):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - True Prediction Experiment')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ ROC curve saved to {save_path}")


def print_metrics(y_true, y_pred, y_prob, config):
    """Calculate and print all evaluation metrics."""

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except:
        roc_auc = None

    print(f"\n{'='*70}")
    print(f"TEST METRICS - TRUE PREDICTION EXPERIMENT")
    print(f"{'='*70}")

    print(f"\nAbnormal Definition: {config.get('abnormal_config', 'unknown').upper()}")
    print(f"Description: {config.get('abnormal_description', 'N/A')}")
    print(f"Abnormal symbols: {config.get('abnormal_symbols', [])}")

    print(f"\n{'='*70}")
    print(f"OVERALL PERFORMANCE")
    print(f"{'='*70}")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}  ← Key metric: How many predicted abnormals are correct?")
    print(f"Recall:    {recall:.4f}  ← Key metric: How many actual abnormals did we find?")
    print(f"F1 Score:  {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC:   {roc_auc:.4f}")

    print(f"\n{'='*70}")
    print(f"CONFUSION MATRIX")
    print(f"{'='*70}")
    print(conf_matrix)
    print("\n[[TN  FP]    TN = True Negative  (correctly predicted normal)")
    print(" [FN  TP]]    FP = False Positive (incorrectly predicted abnormal)")
    print("              FN = False Negative (missed abnormal)")
    print("              TP = True Positive  (correctly predicted abnormal)")

    # Detailed analysis
    tn, fp, fn, tp = conf_matrix.ravel()
    total_normal = tn + fp
    total_abnormal = fn + tp

    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS")
    print(f"{'='*70}")

    print(f"\nNormal beats (actually normal):")
    print(f"  Total: {total_normal}")
    print(f"  Correctly predicted: {tn} ({tn/total_normal*100:.2f}%)")
    print(f"  Falsely predicted as abnormal: {fp} ({fp/total_normal*100:.2f}%)")

    print(f"\nAbnormal beats (actually abnormal):")
    print(f"  Total: {total_abnormal}")
    print(f"  Correctly predicted: {tp} ({tp/total_abnormal*100:.2f}%)")
    print(f"  Missed (predicted as normal): {fn} ({fn/total_abnormal*100:.2f}%)")

    print(f"\nModel behavior:")
    total_pred_abnormal = fp + tp
    total_pred_normal = tn + fn
    print(f"  Predicted abnormal: {total_pred_abnormal}")
    print(f"  Predicted normal: {total_pred_normal}")

    print(f"\n{'='*70}")
    print(f"CLASSIFICATION REPORT")
    print(f"{'='*70}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"]))

    print(f"\n{'='*70}")
    print(f"KEY INSIGHTS")
    print(f"{'='*70}")

    if precision < 0.1:
        print("\n⚠️  VERY LOW PRECISION detected!")
        print("    This means most predicted abnormalities are false alarms.")
        print("    Interpretation: The model struggles to predict abnormalities")
        print("    BEFORE they appear - it tends to over-predict.")
        print("\n    This is expected and highlights the difficulty of true prediction.")
        print("    When the model can't see the abnormality in the input, it")
        print("    can only guess based on subtle patterns in preceding normal beats.")

    if recall > 0.8:
        print("\n✓  HIGH RECALL detected!")
        print("    The model is sensitive and catches most abnormalities.")
        print("    However, combined with low precision, this suggests the model")
        print("    predicts 'abnormal' very liberally to avoid missing cases.")

    if accuracy < 0.5:
        print("\n⚠️  LOW OVERALL ACCURACY detected!")
        print("    This is due to the imbalanced test set and high false positive rate.")
        print("    The model's tendency to over-predict abnormalities hurts accuracy")
        print("    when most test samples are actually normal.")

    # Compare with original results
    print(f"\n{'='*70}")
    print(f"COMPARISON WITH ORIGINAL EXPERIMENT")
    print(f"{'='*70}")
    print("\nOriginal transformer_model.ipynb results (all abnormalities):")
    print("  Accuracy:  37.88%")
    print("  Precision: 0.0325 (3.25%)")
    print("  Recall:    0.8731 (87.31%)")
    print("  F1 Score:  0.0626")
    print("\nYour results:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f}")

    # Save results
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'confusion_matrix': conf_matrix.tolist(),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'abnormal_config': config.get('abnormal_config', 'unknown'),
        'abnormal_symbols': config.get('abnormal_symbols', [])
    }

    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to evaluation_results.json")

    return results


def main():
    """Main evaluation pipeline."""

    parser = argparse.ArgumentParser(
        description='Evaluate True Prediction Experiment'
    )

    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model (default: auto-detect from config)')
    parser.add_argument('--test_data_path', type=str, default='test_data.npy',
                        help='Path to test data file')
    parser.add_argument('--config_path', type=str, default='experiment_config.json',
                        help='Path to experiment configuration file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"TRUE PREDICTION EXPERIMENT - EVALUATION")
    print(f"{'='*70}\n")

    # Load configuration
    print(f"Loading configuration from {args.config_path}...")
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    # Determine model path
    if args.model_path is None:
        args.model_path = f"transformer_model_{config['abnormal_config']}_seq{config['seq_len']}.pth"

    print(f"Model path: {args.model_path}")
    print(f"Test data path: {args.test_data_path}")

    # Load test data
    print(f"\nLoading test data...")
    test_data = np.load(args.test_data_path, allow_pickle=True).item()
    X_test = test_data['X_test']
    y_test = test_data['y_test']

    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels: {len(y_test)}")

    # Create test loader
    test_dataset = BeatSequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = TransformerClassifier(
        seq_len=config['seq_len'],
        beat_len=config['beat_len'],
        d_model=config.get('d_model', 128),
        nhead=config.get('nhead', 4),
        num_layers=config.get('num_layers', 2)
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"✅ Model loaded from {args.model_path}")

    # Evaluate
    print(f"\nEvaluating model...")
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)

    # Print metrics and analysis
    results = print_metrics(y_true, y_pred, y_prob, config)

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm)

    # Plot ROC curve
    plot_roc_curve(y_true, y_prob)

    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
