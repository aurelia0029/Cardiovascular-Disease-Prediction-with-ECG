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
COPY feature_qrs_experiment/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy experiment files
COPY feature_qrs_experiment/data_loader.py .
COPY feature_qrs_experiment/model.py .
COPY feature_qrs_experiment/train_balanced.py .
COPY feature_qrs_experiment/train_semi_balanced.py .

# Copy MIT-BIH database
COPY dataset/mitdb /app/mitdb

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["/bin/bash"]
