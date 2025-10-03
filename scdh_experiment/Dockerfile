# Dockerfile for SCDH Experiments

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY scdh_experiment/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy experiment files
COPY scdh_experiment/feature_extraction.py .
COPY scdh_experiment/data_loader.py .
COPY scdh_experiment/train_scdh_only.py .
COPY scdh_experiment/train_scdh_nsr.py .

# Copy databases from parent directory
COPY sddb /app/sddb
COPY nsrdb /app/nsrdb

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["/bin/bash"]
