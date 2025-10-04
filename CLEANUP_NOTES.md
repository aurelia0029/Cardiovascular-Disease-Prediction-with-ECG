# Cleanup Notes

## Files to Remove

The following files are unnecessary and should be removed:

### 1. Unnecessary __init__.py
```bash
# This file is not needed as experiments are run independently
rm /home/aurelia0029/MIT-BIH/ECG_experiments/__init__.py
```

**Why remove?**
- Each experiment runs independently (not imported as Python packages)
- Docker containers run each experiment in isolation
- No cross-experiment imports needed

### 2. __pycache__ directories (if any exist)
```bash
# Remove all Python cache directories
find /home/aurelia0029/MIT-BIH/ECG_experiments -type d -name "__pycache__" -exec rm -rf {} +
```

**Why remove?**
- Auto-generated Python bytecode cache
- Should never be committed to version control
- Already ignored by .dockerignore files

### 3. Compiled Python files (if any exist)
```bash
# Remove all .pyc, .pyo files
find /home/aurelia0029/MIT-BIH/ECG_experiments -type f -name "*.pyc" -delete
find /home/aurelia0029/MIT-BIH/ECG_experiments -type f -name "*.pyo" -delete
```

## What to Keep

### Keep these .dockerignore files ✅
- `cnn_lstm_experiment/.dockerignore`
- `window_prediction_experiment/.dockerignore`
- `true_prediction_experiment/.dockerignore`
- `scdh_experiment/.dockerignore`
- `feature_qrs_experiment/.dockerignore`

These correctly ignore cache files during Docker builds.

### Keep the new .gitignore ✅
- `ECG_experiments/.gitignore`

This prevents accidentally committing:
- `__pycache__/` directories
- Compiled `.pyc` files
- Trained model files (`.pth`, `.pkl`)
- Output files (`.png`, `.json`)
- Dataset files (too large for git)

## Git Best Practices

After cleanup, ensure git is tracking the right files:

```bash
cd /home/aurelia0029/MIT-BIH/ECG_experiments

# Check current status
git status

# If __pycache__ or .pyc files show up, remove from git:
git rm -r --cached __pycache__
git rm --cached **/*.pyc

# Add the new .gitignore
git add .gitignore

# Commit the cleanup
git commit -m "Add .gitignore and remove unnecessary Python cache files"
```

## What Should Be in Git

**Include:**
- `*.py` - All Python source code
- `*.md` - Documentation (README, PATH_STANDARDS, etc.)
- `requirements.txt` - Dependencies
- `Dockerfile` - Docker configurations
- `.dockerignore` - Docker build exclusions
- `.gitignore` - Git exclusions

**Exclude:**
- `__pycache__/` - Python cache
- `*.pyc, *.pyo` - Compiled Python
- `*.pth, *.pt, *.pkl` - Trained models (too large)
- `*.npy, *.npz` - NumPy arrays (too large)
- Dataset files (`.dat`, `.hea`, `.atr`)
- Output files (`.png`, `.json`, `.log`)
- Virtual environments (`venv/`, `MIT_env/`)

## Summary

✅ Created: `ECG_experiments/.gitignore`
⚠️  To remove manually: `ECG_experiments/__init__.py`
✅ Already protected: All experiments have `.dockerignore`
✅ No `__pycache__` directories currently exist
