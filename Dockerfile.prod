# syntax=docker/dockerfile:1.4

ARG PYTHON_VERSION=3.9.2
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory and non-root user
WORKDIR /app

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install
COPY --chown=appuser:appuser requirements.txt /app/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt


# Set Hugging Face cache
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TRANSFORMERS_CACHE=$HF_HOME/transformers
RUN mkdir -p $TRANSFORMERS_CACHE && chmod -R 777 $HF_HOME

# Copy application source code
COPY --chown=appuser:appuser . /app/

# Change to non-root user
USER appuser

CMD ["gunicorn", "--bind=0.0.0.0:5000", "--timeout=600", "app:app"]

