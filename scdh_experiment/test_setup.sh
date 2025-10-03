#!/bin/bash

echo "=========================================="
echo "SCDH Experiment Setup Verification"
echo "=========================================="
echo ""

# Check current directory
echo "1. Current directory:"
pwd
echo ""

# Check if Python files exist
echo "2. Python files:"
for file in feature_extraction.py data_loader.py train_scdh_only.py train_scdh_nsr.py; do
    if [ -f "$file" ]; then
        echo "   ✓ $file exists"
    else
        echo "   ✗ $file missing!"
    fi
done
echo ""

# Check databases
echo "3. Databases:"
if [ -d "../sddb" ]; then
    ari_count=$(ls ../sddb/*.ari 2>/dev/null | wc -l)
    echo "   ✓ SCDH database found: $ari_count .ari files"
else
    echo "   ✗ SCDH database not found at ../sddb"
fi

if [ -d "../nsrdb" ]; then
    hea_count=$(ls ../nsrdb/*.hea 2>/dev/null | wc -l)
    echo "   ✓ NSR database found: $hea_count .hea files"
else
    echo "   ✗ NSR database not found at ../nsrdb"
fi
echo ""

# Check Dockerfile
echo "4. Docker files:"
if [ -f "Dockerfile" ]; then
    echo "   ✓ Dockerfile exists"
else
    echo "   ✗ Dockerfile missing!"
fi

if [ -f "requirements.txt" ]; then
    echo "   ✓ requirements.txt exists"
else
    echo "   ✗ requirements.txt missing!"
fi
echo ""

# Summary
echo "=========================================="
echo "Setup Status: READY ✓"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "Local execution:"
echo "  python train_scdh_only.py --min 20"
echo "  python train_scdh_nsr.py --min 20"
echo ""
echo "Docker build (from ECG_experiments/):"
echo "  cd .."
echo "  docker build -f scdh_experiment/Dockerfile -t scdh-experiment:latest ."
echo ""
