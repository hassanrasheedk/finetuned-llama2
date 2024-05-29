# Use the official CUDA runtime base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1

# Install necessary packages and Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a directory for the application
WORKDIR /app

# Copy the requirements file and the config file
COPY requirements.txt config.py /app/

# Install the required Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Set the entrypoint for the container to run the prediction script
ENTRYPOINT ["python3", "predict.py"]