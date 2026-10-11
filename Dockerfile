FROM python:3.11-slim

# CPU wheel by default (small). For a GPU box build with:
#   docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/root/.cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install torch --index-url ${TORCH_INDEX} && pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["python", "main.py", "dashboard"]
