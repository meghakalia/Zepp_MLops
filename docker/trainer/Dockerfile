FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[dev]"

# Copy all source code (standalone)
COPY src/ src/

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.pipelines.train_flow"]
