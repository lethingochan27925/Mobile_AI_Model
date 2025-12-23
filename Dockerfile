# --------------------------
# BUILD STAGE
# --------------------------
FROM python:3.10-slim AS build

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/requirements.txt

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY src /app

# --------------------------
# RUN STAGE
# --------------------------
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv

COPY --from=build /app /app

ENV PYTHONUNBUFFERED=1

# Azure Web App requires container to listen on $PORT
ENV PORT=80
EXPOSE 80

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
