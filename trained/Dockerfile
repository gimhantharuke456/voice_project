FROM python:3.10-slim

# System deps: ffmpeg for audio I/O, libsndfile1 for soundfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps before copying code so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained model
COPY api/        ./api/
COPY step4_preprocessor.py step5_model.py ./
COPY models/cdae_best.pth ./models/cdae_best.pth

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "api.app:app"]
