FROM python:3.11-slim

WORKDIR /app

# Set proxy settings as environment variables
ARG http_proxy
ARG https_proxy
ARG no_proxy

ENV http_proxy=${http_proxy} https_proxy=${https_proxy} 
ENV HTTP_PROXY=${http_proxy} HTTPS_PROXY=${https_proxy} 
ENV no_proxy=localhost,127.0.0.1,tfserving
ENV NO_PROXY=localhost,127.0.0.1,tfserving

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py /app/predict.py

# Default command (can be overridden by docker compose run ...)
ENTRYPOINT ["python", "/app/predict.py"]

