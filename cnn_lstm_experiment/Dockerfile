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
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all experiment files
COPY data_loader.py .
COPY dataset.py .
COPY model.py .
COPY train.py .
COPY evaluate.py .

# Create directories for data and outputs
RUN mkdir -p /app/mitdb /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["/bin/bash"]
