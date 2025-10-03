# Docker Instructions for SCDH Experiments

## Directory Structure

Your directory should look like this:

```
ECG_experiments/
├── sddb/                    # SCDH database (Sudden Cardiac Death Holter DB)
│   ├── 30.ari
│   ├── 30.dat
│   ├── 30.hea
│   └── ... (all SCDH files)
│
├── nsrdb/                   # NSR database (Normal Sinus Rhythm DB)
│   ├── 16265.dat
│   ├── 16265.hea
│   └── ... (all NSR files)
│
└── scdh_experiment/
    ├── feature_extraction.py
    ├── data_loader.py
    ├── train_scdh_only.py
    ├── train_scdh_nsr.py
    ├── requirements.txt
    ├── Dockerfile
    └── README.md
```

## Building the Docker Image

**IMPORTANT**: You must build from the `ECG_experiments/` directory (NOT from inside `scdh_experiment/`)

### Step 1: Navigate to ECG_experiments directory

```bash
cd /path/to/MIT-BIH/ECG_experiments
```

### Step 2: Build the image

```bash
docker build -f scdh_experiment/Dockerfile -t scdh-experiment:latest .
```

**Explanation**:
- `-f scdh_experiment/Dockerfile`: Use the Dockerfile in scdh_experiment/
- `-t scdh-experiment:latest`: Tag the image as "scdh-experiment:latest"
- `.`: Build context is current directory (ECG_experiments/)

This allows the Dockerfile to copy:
- `sddb/` → `/app/sddb` inside container
- `nsrdb/` → `/app/nsrdb` inside container
- `scdh_experiment/*.py` → `/app/*.py` inside container

### Build Output

You should see:
```
[+] Building ...
 => COPY sddb /app/sddb       # Copying SCDH database
 => COPY nsrdb /app/nsrdb     # Copying NSR database
 => COPY scdh_experiment/...  # Copying Python scripts
```

---

## Running Experiments

### Experiment 1: SCDH Only

```bash
docker run --rm \
    -v $(pwd)/scdh_experiment:/app/output \
    scdh-experiment:latest \
    python train_scdh_only.py --min 20
```

**Parameters**:
- `--rm`: Remove container after execution
- `-v $(pwd)/scdh_experiment:/app/output`: Mount output directory
- `--min 20`: Extract data 20 minutes before/after VF onset

### Experiment 2: SCDH + NSR

```bash
docker run --rm \
    -v $(pwd)/scdh_experiment:/app/output \
    scdh-experiment:latest \
    python train_scdh_nsr.py --min 20
```

### Different Time Windows

Test with different time windows (5, 10, 15, 20, 25, 30 minutes):

```bash
for min_val in 5 10 15 20 25 30; do
    docker run --rm \
        -v $(pwd)/scdh_experiment:/app/output \
        scdh-experiment:latest \
        python train_scdh_only.py --min $min_val --output scdh_only_${min_val}min.json
done
```

---

## Troubleshooting

### Error: "COPY failed: file not found"

**Problem**: Building from wrong directory

**Solution**: Make sure you're in `ECG_experiments/` directory:
```bash
pwd  # Should show: .../MIT-BIH/ECG_experiments
docker build -f scdh_experiment/Dockerfile -t scdh-experiment:latest .
```

### Error: "No such file or directory: '../sddb'"

**Problem**: Running Python scripts directly without Docker from wrong directory

**Solution**: If running without Docker, make sure you're in `scdh_experiment/`:
```bash
cd ECG_experiments/scdh_experiment
python train_scdh_only.py --min 20
```

### Verify Databases Are Copied

After building, check the image contains databases:

```bash
docker run --rm scdh-experiment:latest ls -la /app/sddb | head -5
docker run --rm scdh-experiment:latest ls -la /app/nsrdb | head -5
```

You should see .dat, .hea, and .ari files.

---

## Inside the Container

### File Paths Inside Container

When scripts run inside Docker, paths are:
- SCDH database: `/app/sddb/`
- NSR database: `/app/nsrdb/`
- Python scripts: `/app/*.py`
- Output: `/app/output/` (mounted to host)

### Automatic Path Detection

The `data_loader.py` automatically detects whether it's running in Docker or locally:

1. **Docker environment**: Looks for `/app/sddb` and `/app/nsrdb`
2. **Local environment**: Looks for `../sddb` and `../nsrdb` (parent directory)
3. **Current directory**: Looks for `sddb` and `nsrdb` in current directory

This means you don't need to specify `--data_dir` or `--scdh_dir` / `--nsr_dir` arguments - the scripts will automatically find the databases!

**Example**:
```python
# This works both in Docker and locally:
python train_scdh_only.py --min 20

# Paths are auto-detected:
# - In Docker: /app/sddb
# - Locally: ../sddb
```
