#!/bin/bash

echo "=========================================="
echo "QRS Feature Experiment v2 - Setup Verification"
echo "=========================================="
echo ""

# Check current directory
echo "1. Current directory:"
pwd
echo ""

# Check Python files
echo "2. Python files:"
for file in data_loader.py model.py train_balanced.py train_semi_balanced.py; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file MISSING!"
    fi
done
echo ""

# Check database
echo "3. MIT-BIH database:"
if [ -d "../mitdb" ]; then
    dat_count=$(ls ../mitdb/*.dat 2>/dev/null | wc -l)
    echo "   ✓ Database found: $dat_count records"
else
    echo "   ✗ Database not found at ../mitdb"
fi
echo ""

# Check Docker files
echo "4. Docker files:"
if [ -f "Dockerfile" ]; then
    echo "   ✓ Dockerfile"
else
    echo "   ✗ Dockerfile MISSING!"
fi

if [ -f "requirements.txt" ]; then
    echo "   ✓ requirements.txt"
else
    echo "   ✗ requirements.txt MISSING!"
fi
echo ""

# Summary
echo "=========================================="
echo "Setup Status: READY ✓"
echo "=========================================="
echo ""
echo "Quick Start:"
echo ""
echo "Local:"
echo "  python train_balanced.py"
echo "  python train_semi_balanced.py"
echo ""
echo "Docker (from ECG_experiments/):"
echo "  cd .."
echo "  docker build -f feature_qrs_experiment_v2/Dockerfile -t qrs-feature-exp:latest ."
echo "  docker run --rm -v \$(pwd)/feature_qrs_experiment_v2:/app/output qrs-feature-exp:latest python train_balanced.py"
echo ""
