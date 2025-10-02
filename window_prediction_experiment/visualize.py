"""
Visualization script for window prediction and sequence length experiments.

This script loads experiment results from JSON files and generates
publication-quality plots showing performance metrics vs parameters.
"""

import argparse
import json
import matplotlib.pyplot as plt
import os


def load_results(filename):
    """Load experiment results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def plot_window_results(results, output_file=None):
    """
    Plot performance metrics vs prediction window (w_before).

    Args:
        results (list): List of result dictionaries with metrics
        output_file (str, optional): Path to save the plot
    """
    w_values = [r['w_before'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1_scores = [r['f1'] for r in results]

    plt.figure(figsize=(12, 7))

    plt.plot(w_values, accuracies, marker='o', linewidth=2, markersize=8, label='Accuracy')
    plt.plot(w_values, precisions, marker='s', linewidth=2, markersize=8, label='Precision')
    plt.plot(w_values, recalls, marker='^', linewidth=2, markersize=8, label='Recall')
    plt.plot(w_values, f1_scores, marker='d', linewidth=2, markersize=8, label='F1 Score')

    plt.xlabel('Prediction Window (beats ahead)', fontsize=14)
    plt.ylabel('Performance Metric', fontsize=14)
    plt.title('Model Performance vs Prediction Horizon', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def plot_sequence_results(results, output_file=None):
    """
    Plot performance metrics vs sequence length.

    Args:
        results (list): List of result dictionaries with metrics
        output_file (str, optional): Path to save the plot
    """
    seq_lengths = [r['sequence_length'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1_scores = [r['f1'] for r in results]

    plt.figure(figsize=(12, 7))

    plt.plot(seq_lengths, accuracies, marker='o', linewidth=2, markersize=8, label='Accuracy')
    plt.plot(seq_lengths, precisions, marker='s', linewidth=2, markersize=8, label='Precision')
    plt.plot(seq_lengths, recalls, marker='^', linewidth=2, markersize=8, label='Recall')
    plt.plot(seq_lengths, f1_scores, marker='d', linewidth=2, markersize=8, label='F1 Score')

    plt.xlabel('Sequence Length (number of beats)', fontsize=14)
    plt.ylabel('Performance Metric', fontsize=14)
    plt.title('Model Performance vs Sequence Length', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)

    # Set x-axis to show odd numbers if available
    odd_seq_lengths = [x for x in seq_lengths if x % 2 == 1]
    if odd_seq_lengths:
        plt.xticks(odd_seq_lengths)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def plot_comparison(window_results, seq_results, output_file=None):
    """
    Create side-by-side comparison of window and sequence experiments.

    Args:
        window_results (list): Window experiment results
        seq_results (list): Sequence experiment results
        output_file (str, optional): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Window plot
    w_values = [r['w_before'] for r in window_results]
    w_acc = [r['accuracy'] for r in window_results]
    w_prec = [r['precision'] for r in window_results]
    w_rec = [r['recall'] for r in window_results]
    w_f1 = [r['f1'] for r in window_results]

    ax1.plot(w_values, w_acc, marker='o', linewidth=2, markersize=6, label='Accuracy')
    ax1.plot(w_values, w_prec, marker='s', linewidth=2, markersize=6, label='Precision')
    ax1.plot(w_values, w_rec, marker='^', linewidth=2, markersize=6, label='Recall')
    ax1.plot(w_values, w_f1, marker='d', linewidth=2, markersize=6, label='F1 Score')
    ax1.set_xlabel('Prediction Window (beats ahead)', fontsize=12)
    ax1.set_ylabel('Performance Metric', fontsize=12)
    ax1.set_title('Performance vs Prediction Horizon', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Sequence plot
    seq_lens = [r['sequence_length'] for r in seq_results]
    s_acc = [r['accuracy'] for r in seq_results]
    s_prec = [r['precision'] for r in seq_results]
    s_rec = [r['recall'] for r in seq_results]
    s_f1 = [r['f1'] for r in seq_results]

    ax2.plot(seq_lens, s_acc, marker='o', linewidth=2, markersize=6, label='Accuracy')
    ax2.plot(seq_lens, s_prec, marker='s', linewidth=2, markersize=6, label='Precision')
    ax2.plot(seq_lens, s_rec, marker='^', linewidth=2, markersize=6, label='Recall')
    ax2.plot(seq_lens, s_f1, marker='d', linewidth=2, markersize=6, label='F1 Score')
    ax2.set_xlabel('Sequence Length (number of beats)', fontsize=12)
    ax2.set_ylabel('Performance Metric', fontsize=12)
    ax2.set_title('Performance vs Sequence Length', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    odd_seq_lens = [x for x in seq_lens if x % 2 == 1]
    if odd_seq_lens:
        ax2.set_xticks(odd_seq_lens)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def print_summary(results, experiment_type='window'):
    """Print summary statistics of experiment results."""
    print(f"\n{'='*60}")
    print(f"{experiment_type.upper()} EXPERIMENT SUMMARY")
    print(f"{'='*60}")

    if experiment_type == 'window':
        param_key = 'w_before'
        param_name = 'Prediction Window'
    else:
        param_key = 'sequence_length'
        param_name = 'Sequence Length'

    # Find best performance
    best_acc = max(results, key=lambda x: x['accuracy'])
    best_f1 = max(results, key=lambda x: x['f1'])

    print(f"\nBest Accuracy: {best_acc['accuracy']:.4f}")
    print(f"  {param_name}: {best_acc[param_key]}")
    print(f"  Precision: {best_acc['precision']:.4f}")
    print(f"  Recall: {best_acc['recall']:.4f}")
    print(f"  F1 Score: {best_acc['f1']:.4f}")

    print(f"\nBest F1 Score: {best_f1['f1']:.4f}")
    print(f"  {param_name}: {best_f1[param_key]}")
    print(f"  Accuracy: {best_f1['accuracy']:.4f}")
    print(f"  Precision: {best_f1['precision']:.4f}")
    print(f"  Recall: {best_f1['recall']:.4f}")

    print(f"\nPerformance Range:")
    print(f"  Accuracy:  [{min(r['accuracy'] for r in results):.4f}, {max(r['accuracy'] for r in results):.4f}]")
    print(f"  Precision: [{min(r['precision'] for r in results):.4f}, {max(r['precision'] for r in results):.4f}]")
    print(f"  Recall:    [{min(r['recall'] for r in results):.4f}, {max(r['recall'] for r in results):.4f}]")
    print(f"  F1 Score:  [{min(r['f1'] for r in results):.4f}, {max(r['f1'] for r in results):.4f}]")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize window prediction and sequence length experiment results'
    )

    parser.add_argument('--window_results', type=str,
                        help='Path to window experiment results JSON file')
    parser.add_argument('--sequence_results', type=str,
                        help='Path to sequence experiment results JSON file')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save output plots')
    parser.add_argument('--compare', action='store_true',
                        help='Create comparison plot (requires both window and sequence results)')

    args = parser.parse_args()

    if not args.window_results and not args.sequence_results:
        print("Error: Please provide at least one results file (--window_results or --sequence_results)")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and visualize window results
    if args.window_results:
        print(f"Loading window results from: {args.window_results}")
        window_results = load_results(args.window_results)
        print_summary(window_results, 'window')

        output_file = os.path.join(args.output_dir, 'window_performance.png')
        plot_window_results(window_results, output_file)

    # Load and visualize sequence results
    if args.sequence_results:
        print(f"Loading sequence results from: {args.sequence_results}")
        seq_results = load_results(args.sequence_results)
        print_summary(seq_results, 'sequence')

        output_file = os.path.join(args.output_dir, 'sequence_performance.png')
        plot_sequence_results(seq_results, output_file)

    # Create comparison plot if both results are available
    if args.compare and args.window_results and args.sequence_results:
        output_file = os.path.join(args.output_dir, 'comparison_plot.png')
        plot_comparison(window_results, seq_results, output_file)


if __name__ == "__main__":
    main()
