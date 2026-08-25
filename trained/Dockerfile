FROM python:3.10-slim

# System deps:
#   ffmpeg, libsndfile1 - audio I/O
#   build-essential     - compiles pesq's Cython extension (no prebuilt wheel exists)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps before copying code so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, trained model, and static assets
COPY api.py .
COPY model.pth .
COPY static/ ./static/

EXPOSE 5055

CMD ["python", "api.py"]
