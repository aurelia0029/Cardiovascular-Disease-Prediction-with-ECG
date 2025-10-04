# Dockerfile for QRS Feature Extraction Experiments

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY feature_qrs_experiment_v2/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy experiment files
COPY feature_qrs_experiment_v2/data_loader.py .
COPY feature_qrs_experiment_v2/model.py .
COPY feature_qrs_experiment_v2/train_balanced.py .
COPY feature_qrs_experiment_v2/train_semi_balanced.py .

# Copy MIT-BIH database
COPY mitdb /app/mitdb

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["/bin/bash"]
