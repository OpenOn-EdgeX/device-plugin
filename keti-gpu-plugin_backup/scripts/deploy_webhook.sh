#!/bin/bash
#
# Deploy Script - Deploy KETI GPU Webhook
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NAMESPACE="${NAMESPACE:-edge-system}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  KETI GPU Webhook - Deploy              ${NC}"
echo -e "${GREEN}==========================================${NC}"

cd "$PROJECT_DIR"

# Ensure namespace exists
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

echo -e "${YELLOW}Step 1: Deploying RBAC and Service...${NC}"
kubectl apply -f deploy/webhook/webhook-config.yaml

echo ""
echo -e "${YELLOW}Step 2: Generating TLS certificates...${NC}"
bash deploy/webhook/generate-certs.sh ${NAMESPACE}

echo ""
echo -e "${YELLOW}Step 3: Deploying Webhook...${NC}"
kubectl apply -f deploy/webhook/deployment.yaml

echo ""
echo -e "${YELLOW}Waiting for webhook to be ready...${NC}"
kubectl rollout status deployment/keti-gpu-webhook -n ${NAMESPACE} --timeout=60s

echo ""
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  Webhook Deployment Complete!           ${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""
echo "Check webhook status:"
echo "  kubectl get pods -n ${NAMESPACE} -l app=keti-gpu-webhook"
echo "  kubectl logs -n ${NAMESPACE} -l app=keti-gpu-webhook"
echo ""
echo "Test with a GPU pod:"
echo "  kubectl apply -f scripts/test/workloads/workload-a.yaml"
echo "  kubectl describe pod workload-a"
