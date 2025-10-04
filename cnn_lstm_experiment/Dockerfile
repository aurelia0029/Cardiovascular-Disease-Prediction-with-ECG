# Dockerfile for CNN-LSTM ECG Arrhythmia Classification Experiment

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY cnn_lstm_experiment/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all experiment files
COPY cnn_lstm_experiment/data_loader.py .
COPY cnn_lstm_experiment/dataset.py .
COPY cnn_lstm_experiment/model.py .
COPY cnn_lstm_experiment/train.py .
COPY cnn_lstm_experiment/evaluate.py .

# Copy MIT-BIH dataset
COPY dataset/mitdb /app/mitdb

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["/bin/bash"]
