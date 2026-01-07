#!/bin/bash
#
# 2. Deploy Script - Deploy KETI NPU Plugin to Kubernetes/KubeEdge
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NAMESPACE="${NAMESPACE:-edge-system}"
REGISTRY="${REGISTRY:-ketidevit2}"
IMAGE_NAME="${IMAGE_NAME:-keti-npu-plugin}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  KETI NPU Plugin - Deploy             ${NC}"
echo -e "${GREEN}========================================${NC}"

cd "$PROJECT_DIR"

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl not found${NC}"
    exit 1
fi

# Check cluster connection
echo -e "${YELLOW}Checking cluster connection...${NC}"
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}Error: Cannot connect to Kubernetes cluster${NC}"
    exit 1
fi

# Create namespace if not exists
echo -e "${YELLOW}Checking namespace...${NC}"
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo -e "${YELLOW}Creating namespace: $NAMESPACE${NC}"
    kubectl create namespace "$NAMESPACE"
else
    echo -e "${GREEN}Namespace $NAMESPACE already exists${NC}"
fi

echo -e "${YELLOW}Deploying DaemonSet...${NC}"
kubectl apply -f deploy/daemonset.yaml

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Deploy Complete!                     ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Check status:"
echo "  kubectl get daemonset -n $NAMESPACE keti-npu-plugin"
echo "  kubectl get pods -n $NAMESPACE -l app=keti-npu-plugin"
echo ""
echo "View logs:"
echo "  kubectl logs -n $NAMESPACE -l app=keti-npu-plugin -f"
