# RoboSumo PyTorch Docker Image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    libosmesa6-dev \
    libglfw3 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
ENV PYTHONBREAKPOINT=ipdb.set_trace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire robosumo_torch directory
COPY . .

# Set PYTHONPATH to include the current directory
ENV PYTHONPATH=/app:$PYTHONPATH

# Set MuJoCo rendering to use offscreen/OSMesa mode
ENV MUJOCO_GL=osmesa

# Create output directory
RUN mkdir -p /app/out

# Default command
CMD ["python", "play.py"]

