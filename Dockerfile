# Containerised run of the pipeline (Track B engineering: containerisation optional).
#   docker build -t cmpe253-hmda .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" cmpe253-hmda          # make all
#   docker run --rm -p 8501:8501 -v "$PWD/results:/app/results" cmpe253-hmda python -m streamlit run app/dashboard.py --server.address 0.0.0.0
# (written for Docker Desktop on the author's laptop; this image was not built in the development sandbox, which has no Docker daemon)
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends make libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config.yaml Makefile ./
COPY src ./src
COPY app ./app
ENV PYTHONUNBUFFERED=1
CMD ["make", "all"]
