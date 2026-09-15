# Base image: PyTorch runtime (CUDA 12.1, cuDNN 8)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system packages for rasterio, GDAL, H3, and database connectivity
RUN apt-get update && apt-get install -y \
    software-properties-common \
    gdal-bin \
    libgdal-dev \
    build-essential \
    curl \
    postgresql-client \
    redis-tools \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (includes PostgreSQL, Redis, Prometheus)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose ports: FastAPI (8000), Streamlit (8503), Prometheus (9090)
EXPOSE 8000 8503 9090

# Default command: FastAPI server
# Can override with: docker run -it <image> streamlit run dashboard/app.py --server.port 8503
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
