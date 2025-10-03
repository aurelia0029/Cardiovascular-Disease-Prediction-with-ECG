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
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy experiment files
COPY feature_extraction.py .
COPY data_loader.py .
COPY train_scdh_only.py .
COPY train_scdh_nsr.py .

# Copy databases from parent directories
COPY ../../sddb /app/sddb
COPY ../../nsrdb /app/nsrdb

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["/bin/bash"]
