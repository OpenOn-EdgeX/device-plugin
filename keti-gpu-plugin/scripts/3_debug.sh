#!/bin/bash
#
# 3. Debug Script - View logs from KETI GPU Plugin
#
# KubeEdge 환경에서는 Cloud에서 Edge 노드 로그 직접 조회 불가
# - Cloud에서 실행: Pod 상태만 확인
# - Edge 노드에서 실행: 실시간 로그 스트리밍
#

set -e

# Configuration
NAMESPACE="${NAMESPACE:-edge-system}"
LABEL_SELECTOR="app=keti-gpu-plugin"
CONTAINER_NAME="gpu-plugin"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  KETI GPU Plugin - Debug              ${NC}"
echo -e "${GREEN}========================================${NC}"

# Detect environment: Edge node or Cloud
is_edge_node() {
    # Check if crictl or docker is available and EdgeCore is running
    if command -v crictl &> /dev/null || command -v docker &> /dev/null; then
        if pgrep -x "edgecore" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

if is_edge_node; then
    echo -e "${YELLOW}Detected: Edge Node${NC}"
    echo ""

    # Try crictl first (containerd), then docker
    if command -v crictl &> /dev/null; then
        echo -e "${YELLOW}Using crictl...${NC}"
        CONTAINER_ID=$(crictl ps --name "$CONTAINER_NAME" -q 2>/dev/null | head -1)

        if [ -n "$CONTAINER_ID" ]; then
            echo -e "${GREEN}Container ID: ${CONTAINER_ID}${NC}"
            echo ""
            echo -e "${GREEN}========================================${NC}"
            echo -e "${GREEN}  Streaming Logs (Ctrl+C to exit)      ${NC}"
            echo -e "${GREEN}========================================${NC}"
            echo ""
            crictl logs -f "$CONTAINER_ID"
        else
            echo -e "${RED}Container not found. Checking all containers:${NC}"
            crictl ps -a
        fi
    elif command -v docker &> /dev/null; then
        echo -e "${YELLOW}Using docker...${NC}"
        CONTAINER_ID=$(docker ps --filter "name=$CONTAINER_NAME" -q 2>/dev/null | head -1)

        if [ -n "$CONTAINER_ID" ]; then
            echo -e "${GREEN}Container ID: ${CONTAINER_ID}${NC}"
            echo ""
            echo -e "${GREEN}========================================${NC}"
            echo -e "${GREEN}  Streaming Logs (Ctrl+C to exit)      ${NC}"
            echo -e "${GREEN}========================================${NC}"
            echo ""
            docker logs -f "$CONTAINER_ID"
        else
            echo -e "${RED}Container not found. Checking all containers:${NC}"
            docker ps -a
        fi
    else
        echo -e "${RED}No container runtime found (crictl/docker)${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Detected: Cloud/Master Node${NC}"
    echo -e "${YELLOW}Note: KubeEdge does not support direct log streaming from Cloud to Edge${NC}"
    echo ""

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}Error: kubectl not found${NC}"
        exit 1
    fi

    # Show pod status
    echo -e "${YELLOW}Pod Status:${NC}"
    kubectl get pods -n "$NAMESPACE" -l "$LABEL_SELECTOR" -o wide
    echo ""

    # Show pod details
    echo -e "${YELLOW}Pod Details:${NC}"
    kubectl describe pod -n "$NAMESPACE" -l "$LABEL_SELECTOR" | grep -A 20 "Events:"
    echo ""

    echo -e "${GREEN}========================================${NC}"
    echo -e "${YELLOW}To view real-time logs, run this script on the Edge node:${NC}"
    echo ""
    echo "  # Copy script to edge node"
    echo "  scp scripts/3_debug.sh edge-node:/tmp/"
    echo ""
    echo "  # Or run directly on edge node:"
    echo "  crictl logs -f \$(crictl ps --name scheduler -q)"
    echo ""
    echo -e "${GREEN}========================================${NC}"
fi
