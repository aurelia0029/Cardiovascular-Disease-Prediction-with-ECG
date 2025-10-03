"""
Compare Personalized vs Generalized Model Performance

This script compares the performance of:
1. Personalized models (one per patient, trained on that patient's data)
2. Generalized models (one model trained on all patients)

Analysis includes:
- Per-patient performance comparison
- Aggregate statistics
- Statistical significance testing
- Visualization of differences
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from glob import glob


def load_personalized_results(results_dir='./personalized_models_output'):
    """
    Load most recent personalized model results.
    """
    json_files = glob(os.path.join(results_dir, 'personalized_results_*.json'))

    if not json_files:
        raise FileNotFoundError(f"No personalized results found in {results_dir}")

    # Get most recent file
    latest_file = max(json_files, key=os.path.getmtime)

    with open(latest_file, 'r') as f:
        results = json.load(f)

    print(f"Loaded personalized results from: {latest_file}")
    return results


def load_generalized_results(results_file=None):
    """
    Load generalized model results from train_scdh_only.py or train_scdh_nsr.py

    Note: This is a placeholder. Actual implementation depends on where
    generalized model results are saved.
    """
    # This needs to be implemented based on your actual generalized model output
    # For now, return None to indicate it needs implementation

    print("Note: Generalized model loading not yet implemented")
    print("Please specify the path to your generalized model results")

    return None


def extract_patient_metrics(personalized_results):
    """
    Extract per-patient metrics from personalized results.
    """
    patients = []
    metrics = {
        'accuracy': [],
        'roc_auc': [],
        'onset_precision': [],
        'onset_recall': [],
        'onset_f1': [],
        'normal_precision': [],
        'normal_recall': [],
        'normal_f1': []
    }

    successful = personalized_results['successful_patients']

    for patient_id in successful:
        result = personalized_results['patient_results'][patient_id]
        patient_metrics = result['metrics']

        patients.append(patient_id)
        metrics['accuracy'].append(patient_metrics['accuracy'])

        if patient_metrics.get('both_classes', False):
            metrics['roc_auc'].append(patient_metrics['roc_auc'])
            metrics['onset_precision'].append(patient_metrics['onset_precision'])
            metrics['onset_recall'].append(patient_metrics['onset_recall'])
            metrics['onset_f1'].append(patient_metrics['onset_f1'])
            metrics['normal_precision'].append(patient_metrics['normal_precision'])
            metrics['normal_recall'].append(patient_metrics['normal_recall'])
            metrics['normal_f1'].append(patient_metrics['normal_f1'])
        else:
            # If only one class, use NaN
            metrics['roc_auc'].append(np.nan)
            metrics['onset_precision'].append(np.nan)
            metrics['onset_recall'].append(np.nan)
            metrics['onset_f1'].append(np.nan)
            metrics['normal_precision'].append(np.nan)
            metrics['normal_recall'].append(np.nan)
            metrics['normal_f1'].append(np.nan)

    # Convert to DataFrame
    df = pd.DataFrame(metrics, index=patients)
    return df


def plot_comparison_boxplots(personalized_df, generalized_df=None, output_dir='./comparison_output'):
    """
    Create boxplot comparisons between personalized and generalized models.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics_to_plot = ['accuracy', 'roc_auc', 'onset_f1', 'onset_recall', 'onset_precision']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Prepare data
        data_to_plot = []
        labels = []

        # Personalized models
        personalized_values = personalized_df[metric].dropna()
        if len(personalized_values) > 0:
            data_to_plot.append(personalized_values)
            labels.append('Personalized')

        # Generalized models (if available)
        if generalized_df is not None and metric in generalized_df.columns:
            generalized_values = generalized_df[metric].dropna()
            if len(generalized_values) > 0:
                data_to_plot.append(generalized_values)
                labels.append('Generalized')

        # Plot
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

            # Color boxes
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)

            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.grid(axis='y', alpha=0.3)

            # Add mean line
            for i, data in enumerate(data_to_plot):
                mean_val = np.mean(data)
                ax.plot([i+0.8, i+1.2], [mean_val, mean_val], 'r--', linewidth=2)

    # Remove extra subplot
    fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_boxplots.png'), dpi=300)
    plt.close()

    print(f"Comparison boxplots saved to {output_dir}/comparison_boxplots.png")


def plot_patient_comparison_bars(personalized_df, generalized_df=None, output_dir='./comparison_output'):
    """
    Create side-by-side bar charts comparing personalized vs generalized for each patient.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot accuracy comparison
    fig, ax = plt.subplots(figsize=(14, 6))

    patients = personalized_df.index
    x = np.arange(len(patients))
    width = 0.35

    ax.bar(x - width/2, personalized_df['accuracy'], width,
           label='Personalized', alpha=0.8, color='steelblue')

    if generalized_df is not None and 'accuracy' in generalized_df.columns:
        ax.bar(x + width/2, generalized_df['accuracy'], width,
               label='Generalized', alpha=0.8, color='coral')

    ax.set_xlabel('Patient ID')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy: Personalized vs Generalized Models')
    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'patient_accuracy_comparison.png'), dpi=300)
    plt.close()

    print(f"Patient comparison bars saved to {output_dir}/patient_accuracy_comparison.png")


def statistical_comparison(personalized_df, generalized_df=None):
    """
    Perform statistical tests to compare personalized vs generalized models.
    """
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON")
    print("="*80)

    metrics = ['accuracy', 'roc_auc', 'onset_f1', 'onset_recall', 'onset_precision']

    for metric in metrics:
        print(f"\n{metric.upper()}:")

        personalized_values = personalized_df[metric].dropna()

        print(f"  Personalized Models (n={len(personalized_values)}):")
        print(f"    Mean: {personalized_values.mean():.4f}")
        print(f"    Std:  {personalized_values.std():.4f}")
        print(f"    Min:  {personalized_values.min():.4f}")
        print(f"    Max:  {personalized_values.max():.4f}")

        if generalized_df is not None and metric in generalized_df.columns:
            generalized_values = generalized_df[metric].dropna()

            print(f"  Generalized Model (n={len(generalized_values)}):")
            print(f"    Mean: {generalized_values.mean():.4f}")
            print(f"    Std:  {generalized_values.std():.4f}")

            # Paired t-test (if same patients)
            if len(personalized_values) == len(generalized_values):
                t_stat, p_value = stats.ttest_rel(personalized_values, generalized_values)
                print(f"  Paired t-test:")
                print(f"    t-statistic: {t_stat:.4f}")
                print(f"    p-value: {p_value:.4f}")

                if p_value < 0.05:
                    if personalized_values.mean() > generalized_values.mean():
                        print(f"    ✓ Personalized significantly BETTER (p < 0.05)")
                    else:
                        print(f"    ✗ Generalized significantly BETTER (p < 0.05)")
                else:
                    print(f"    No significant difference (p ≥ 0.05)")

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((personalized_values.std()**2 + generalized_values.std()**2) / 2)
            cohen_d = (personalized_values.mean() - generalized_values.mean()) / pooled_std
            print(f"  Effect size (Cohen's d): {cohen_d:.4f}")

            if abs(cohen_d) < 0.2:
                effect = "negligible"
            elif abs(cohen_d) < 0.5:
                effect = "small"
            elif abs(cohen_d) < 0.8:
                effect = "medium"
            else:
                effect = "large"

            print(f"    Interpretation: {effect} effect")


def generate_summary_table(personalized_df, generalized_df=None, output_dir='./comparison_output'):
    """
    Generate a summary comparison table.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = ['accuracy', 'roc_auc', 'onset_precision', 'onset_recall', 'onset_f1']

    summary = []

    for metric in metrics:
        row = {'Metric': metric.replace('_', ' ').title()}

        personalized_values = personalized_df[metric].dropna()
        row['Personalized Mean'] = f"{personalized_values.mean():.4f}"
        row['Personalized Std'] = f"{personalized_values.std():.4f}"
        row['Personalized N'] = len(personalized_values)

        if generalized_df is not None and metric in generalized_df.columns:
            generalized_values = generalized_df[metric].dropna()
            row['Generalized Mean'] = f"{generalized_values.mean():.4f}"
            row['Generalized Std'] = f"{generalized_values.std():.4f}"
            row['Generalized N'] = len(generalized_values)

            # Difference
            diff = personalized_values.mean() - generalized_values.mean()
            row['Difference'] = f"{diff:+.4f}"
        else:
            row['Generalized Mean'] = "N/A"
            row['Generalized Std'] = "N/A"
            row['Generalized N'] = "N/A"
            row['Difference'] = "N/A"

        summary.append(row)

    summary_df = pd.DataFrame(summary)

    # Save to CSV
    csv_file = os.path.join(output_dir, 'comparison_summary.csv')
    summary_df.to_csv(csv_file, index=False)

    print(f"\nSummary table saved to {csv_file}")
    print("\n" + summary_df.to_string(index=False))


def main():
    """
    Main comparison function.
    """
    print("="*80)
    print("PERSONALIZED vs GENERALIZED MODEL COMPARISON")
    print("="*80)

    output_dir = './comparison_output'
    os.makedirs(output_dir, exist_ok=True)

    # Load personalized results
    print("\nLoading personalized model results...")
    personalized_results = load_personalized_results()
    personalized_df = extract_patient_metrics(personalized_results)

    print(f"Loaded metrics for {len(personalized_df)} patients")

    # Load generalized results (if available)
    print("\nLoading generalized model results...")
    generalized_df = load_generalized_results()

    if generalized_df is None:
        print("\nWARNING: Generalized model results not available.")
        print("Only analyzing personalized model performance.")
        print("\nTo enable comparison, implement load_generalized_results()")
        print("to load results from train_scdh_only.py or train_scdh_nsr.py")

    # Generate visualizations
    print("\nGenerating comparison visualizations...")
    plot_comparison_boxplots(personalized_df, generalized_df, output_dir)
    plot_patient_comparison_bars(personalized_df, generalized_df, output_dir)

    # Statistical comparison
    if generalized_df is not None:
        statistical_comparison(personalized_df, generalized_df)

    # Summary table
    print("\nGenerating summary table...")
    generate_summary_table(personalized_df, generalized_df, output_dir)

    print(f"\n✓ Comparison completed!")
    print(f"✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
