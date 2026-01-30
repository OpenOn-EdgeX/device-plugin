#!/bin/bash
#
# Test Script for KETI GPU Plugin
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  KETI GPU Plugin - Test               ${NC}"
echo -e "${GREEN}========================================${NC}"

case "$1" in
    "single")
        echo -e "${YELLOW}Deploying single GPU pod...${NC}"
        kubectl apply -f "$SCRIPT_DIR/test-gpu-pod.yaml"
        echo ""
        echo "Check allocation:"
        echo "  kubectl logs -n edge-system -l app=keti-gpu-plugin -f"
        ;;
    "multi")
        echo -e "${YELLOW}Deploying multiple GPU pods...${NC}"
        kubectl apply -f "$SCRIPT_DIR/test-multi-gpu-pods.yaml"
        echo ""
        echo "Expected allocations:"
        echo "  ai-inference-1: memory=2048MB, cores=30%"
        echo "  ai-inference-2: memory=3072MB, cores=40%"
        echo "  ai-inference-3: memory=1024MB, cores=20%"
        echo "  Total: memory=6144MB, cores=90%"
        ;;
    "overcommit")
        echo -e "${YELLOW}Deploying overcommit test pods...${NC}"
        kubectl apply -f "$SCRIPT_DIR/test-overcommit.yaml"
        echo ""
        echo "Expected: Warning about insufficient GPU cores (60% + 60% = 120%)"
        ;;
    "clean")
        echo -e "${YELLOW}Cleaning up test pods...${NC}"
        kubectl delete pod -l test=single-pod --ignore-not-found
        kubectl delete pod -l test=multi-pod --ignore-not-found
        kubectl delete pod -l test=overcommit --ignore-not-found
        echo -e "${GREEN}Cleanup complete${NC}"
        ;;
    "status")
        echo -e "${YELLOW}Scheduler status:${NC}"
        kubectl logs -n edge-system -l app=keti-gpu-plugin --tail=20
        ;;
    *)
        echo "Usage: $0 {single|multi|overcommit|clean|status}"
        echo ""
        echo "  single     - Deploy single GPU pod test"
        echo "  multi      - Deploy multiple GPU pods test"
        echo "  overcommit - Deploy overcommit test (cores > 100%)"
        echo "  clean      - Delete all test pods"
        echo "  status     - Show scheduler logs"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
