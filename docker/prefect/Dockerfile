FROM prefecthq/prefect:2-python3.12

WORKDIR /app

# Install project dependencies
COPY pyproject.toml .

# Ensure modern packaging tooling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir ".[dev]"

# Copy all source code (standalone - no external mounts needed)
COPY src/ src/

ENV PYTHONPATH=/app

CMD ["prefect", "server", "start", "--host", "0.0.0.0"]



