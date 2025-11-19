#!/bin/bash
# Generate Python gRPC code from proto file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m grpc_tools.protoc \
    -I"$SCRIPT_DIR/proto" \
    --python_out="$SCRIPT_DIR" \
    --grpc_python_out="$SCRIPT_DIR" \
    "$SCRIPT_DIR/proto/api.proto"

echo "Generated Python gRPC code in $SCRIPT_DIR"
