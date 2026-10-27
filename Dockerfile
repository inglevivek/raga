
# Use official NVIDIA CUDA runtime base (CUDA already included — avoids downloading nvidia-*-cu12 wheels)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Suppress interactive prompts (tzdata etc.) during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.12 via deadsnakes PPA (Ubuntu 22.04 default repos only have 3.10/3.11)
RUN apt-get update && apt-get install -y software-properties-common curl build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
        python3.12 \
        python3.12-dev \
    && rm -rf /var/lib/apt/lists/*



# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Make python3.12 the default python / python3
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Install uv
RUN pip3.12 install uv


WORKDIR /Raga

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
# Increase HTTP timeout for large downloads
ENV UV_HTTP_TIMEOUT=120

# Copy dependencies first
COPY requirements.txt .

# Install PyTorch CUDA build — CUDA libs already in base image so only torch wheel downloads
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
RUN uv pip install --system -r requirements.txt

# Copy application code
COPY . .

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
