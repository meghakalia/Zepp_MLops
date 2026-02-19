FROM python:3.12-slim

WORKDIR /app

# Install build tools needed for tdigest (Cython extension)
RUN apt-get update -qq && apt-get install -y -qq gcc && rm -rf /var/lib/apt/lists/*

# Install only inference dependencies (CPU-only torch, no GPU)

RUN pip install --no-cache-dir \
    torch \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    fastapi \
    uvicorn \
    mlflow \
    numpy \
    scikit-learn \
    python-dotenv \
    pandas \
    openpyxl \
    matplotlib \
    tqdm \
    statsmodels \
    tdigest \
    xlsxwriter

# Copy only what inference needs (standalone)
COPY src/__init__.py src/__init__.py
COPY src/training/__init__.py src/training/__init__.py
COPY src/training/model.py src/training/model.py
COPY src/training/dataset_delta_v2.py src/training/dataset_delta_v2.py
COPY src/training/utils.py src/training/utils.py
COPY src/inference/ src/inference/
COPY src/data_ingestion/ src/data_ingestion/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
