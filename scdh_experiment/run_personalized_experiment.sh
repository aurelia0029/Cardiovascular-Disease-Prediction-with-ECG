#!/bin/bash

# Personalized VF Prediction Experiment Runner
# This script runs the complete personalized model experiment pipeline

echo "=========================================================================="
echo "PERSONALIZED PATIENT VF PREDICTION EXPERIMENT"
echo "=========================================================================="
echo ""
echo "This experiment builds individual models for each SCDH patient"
echo "using 2/3 of their data for training and 1/3 for testing."
echo ""
echo "Motivation: Test if personalized models outperform generalized models"
echo "by learning individual-specific patterns."
echo ""
echo "=========================================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "exp_personalized_models.py" ]; then
    echo "ERROR: exp_personalized_models.py not found"
    echo "Please run this script from the scdh_experiment directory"
    exit 1
fi

# Check for required files
echo "Checking dependencies..."

if [ ! -f "data_loader.py" ]; then
    echo "ERROR: data_loader.py not found"
    exit 1
fi

if [ ! -f "feature_extraction.py" ]; then
    echo "ERROR: feature_extraction.py not found"
    exit 1
fi

echo "✓ All dependencies found"
echo ""

# Check for SCDH database
echo "Checking for SCDH database..."

SDDB_FOUND=0
for path in "../../sddb" "../sddb" "./sddb" "/app/sddb"; do
    if [ -d "$path" ]; then
        echo "✓ Found SCDH database at: $path"
        SDDB_FOUND=1
        break
    fi
done

if [ $SDDB_FOUND -eq 0 ]; then
    echo "WARNING: SCDH database (sddb/) not found"
    echo "Please ensure the database is available before running"
fi

echo ""
echo "=========================================================================="
echo "STEP 1: Running Personalized Model Experiment"
echo "=========================================================================="
echo ""

# Run the main experiment
python3 exp_personalized_models.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Experiment failed"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "STEP 2: Generating Comparison Analysis (Optional)"
echo "=========================================================================="
echo ""

read -p "Do you want to compare with generalized models? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running comparison analysis..."
    python3 compare_personalized_vs_generalized.py

    if [ $? -ne 0 ]; then
        echo "WARNING: Comparison failed (may need generalized results)"
    fi
fi

echo ""
echo "=========================================================================="
echo "EXPERIMENT COMPLETE"
echo "=========================================================================="
echo ""
echo "Results location:"
echo "  - Personalized results: ./personalized_models_output/"
echo "  - Comparison results: ./comparison_output/ (if run)"
echo ""
echo "Key files:"
echo "  - personalized_results_*.json - Summary metrics"
echo "  - personalized_results_*.pkl - Full results with predictions"
echo "  - plots/*.png - Visualization charts"
echo ""
echo "Next steps:"
echo "  1. Review the summary statistics printed above"
echo "  2. Check the plots in personalized_models_output/plots/"
echo "  3. Compare with generalized model results (if available)"
echo "  4. Analyze individual patient results for insights"
echo ""
echo "✓ Done!"
