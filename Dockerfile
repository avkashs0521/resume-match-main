# Use a lightweight official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies if needed
# RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the SentenceTransformer model to ensure the image is portable
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', device='cpu')"

# Copy the rest of the application
COPY . .

# Clean up local environment artifacts if any
RUN rm -rf __pycache__ .pytest_cache

# Set FastAPI as the entry point
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "7860"]