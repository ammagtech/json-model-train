FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Working directory
WORKDIR /app

# Install PyTorch with CUDA 12.6
RUN pip install --no-cache-dir torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
RUN pip install --no-cache-dir \
    runpod \
    transformers==4.39.0 \
    datasets \
    accelerate \
    scipy \
    numpy \
    tokenizers \
    safetensors

# Copy all project files (make sure train.py, inference.py, rp_handler.py are here)
COPY . .

# Expose RunPod handler
CMD ["python3", "rp_handler.py"]
