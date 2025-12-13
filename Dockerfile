FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["bash"]
