# Dockerfile for True Prediction Experiment

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY true_prediction_experiment/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all experiment files
COPY true_prediction_experiment/data_loader.py .
COPY true_prediction_experiment/dataset.py .
COPY true_prediction_experiment/data_processing.py .
COPY true_prediction_experiment/model.py .
COPY true_prediction_experiment/train.py .
COPY true_prediction_experiment/evaluate.py .

# Copy MIT-BIH dataset
COPY dataset/mitdb /app/mitdb

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["/bin/bash"]
