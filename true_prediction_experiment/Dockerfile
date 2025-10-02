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
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all experiment files
COPY data_loader.py .
COPY dataset.py .
COPY data_processing.py .
COPY model.py .
COPY train.py .
COPY evaluate.py .

# Copy MIT-BIH dataset from parent directory
COPY ../../mitdb /app/mitdb

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["/bin/bash"]
