#!/bin/bash
#
# Build Script - Build Docker image for KETI GPU Webhook
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REGISTRY="${REGISTRY:-ketidevit2}"
IMAGE_NAME="${IMAGE_NAME:-keti-gpu-webhook}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  KETI GPU Webhook - Build               ${NC}"
echo -e "${GREEN}==========================================${NC}"

cd "$PROJECT_DIR"

echo -e "${YELLOW}Building image: ${FULL_IMAGE}${NC}"

# Build Docker image
docker build -t "$FULL_IMAGE" -f Dockerfile.webhook .

echo ""
echo -e "${YELLOW}Pushing image to Docker Hub...${NC}"
docker push "$FULL_IMAGE"

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  Build & Push Complete!                 ${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""
echo "Image: $FULL_IMAGE"
echo ""
echo "Next steps:"
echo "  1. Generate certificates: ./deploy/webhook/generate-certs.sh"
echo "  2. Deploy webhook: kubectl apply -f deploy/webhook/"
