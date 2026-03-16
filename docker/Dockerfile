# Stage 1: Builder
FROM python:3.9-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final Runtime
FROM python:3.9-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Update PATH
ENV PATH=/root/.local/bin:$PATH

# Environment variables for model caching
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV TORCH_HOME=/app/cache/torch

EXPOSE 8000

# Start the application
CMD ["uvicorn", "src.api.gateway:app", "--host", "0.0.0.0", "--port", "8000"]
