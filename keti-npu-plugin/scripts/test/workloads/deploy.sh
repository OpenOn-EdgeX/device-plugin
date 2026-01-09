#!/bin/bash
# Deploy NPU test workloads
# Usage: ./deploy.sh [test]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== NPU Test Workload Deployment ==="
echo ""

deploy_workload() {
    local name=$1
    echo "Deploying ${name}..."
    kubectl apply -f "${SCRIPT_DIR}/${name}.yaml"
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 [npu-test|all]"
    echo ""
    echo "Examples:"
    echo "  $0 npu-test   # Deploy NPU test workload"
    echo "  $0 all        # Deploy all workloads"
    exit 0
fi

for arg in "$@"; do
    case $arg in
        npu-test)
            deploy_workload "npu-test"
            ;;
        all)
            deploy_workload "npu-test"
            ;;
        *)
            echo "Unknown workload: $arg"
            ;;
    esac
done

echo ""
echo "Waiting for pods to start..."
sleep 3

echo ""
echo "=== Pod Status ==="
kubectl get pods -l app=npu-test-workload -o wide

echo ""
echo "=== Monitor logs with: ==="
echo "kubectl logs -f npu-test"
