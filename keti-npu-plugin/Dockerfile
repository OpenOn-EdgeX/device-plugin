# KETI NPU Plugin Dockerfile
# For KubeEdge Edge Node NPU Resource Management

FROM python:3.11-slim

LABEL maintainer="KETI"
LABEL description="KETI NPU Plugin for KubeEdge with Device Plugin"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY cmd/ ./cmd/
COPY pkg/ ./pkg/
COPY config/ ./config/

# Generate gRPC proto files
RUN python -m grpc_tools.protoc \
    -I/app/pkg/device_plugin/proto \
    --python_out=/app/pkg/device_plugin \
    --grpc_python_out=/app/pkg/device_plugin \
    /app/pkg/device_plugin/proto/api.proto && \
    # Fix import statement for package structure (absolute -> relative)
    sed -i 's/^import api_pb2 as api__pb2$/from . import api_pb2 as api__pb2/' \
    /app/pkg/device_plugin/api_pb2_grpc.py

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "print('healthy')" || exit 1

# Entry point
ENTRYPOINT ["python", "-u", "cmd/main.py"]
