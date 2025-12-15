#!/bin/bash
# Deploy SM test workloads
# Usage: ./deploy.sh [a|b|c|d|all]
# Example: ./deploy.sh a b   - Deploy workload-a and workload-b
#          ./deploy.sh all   - Deploy all workloads

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== SM Partitioning Test Deployment ==="
echo ""

# Check if vai-accelerator-binary configmap exists
if ! kubectl get configmap vai-accelerator-binary &>/dev/null; then
    echo "ERROR: vai-accelerator-binary ConfigMap not found!"
    echo "Please create it first with the vai_accelerator.so library"
    exit 1
fi

deploy_workload() {
    local name=$1
    echo "Deploying workload-${name}..."
    kubectl apply -f "${SCRIPT_DIR}/workload-${name}.yaml"
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 [a|b|c|d|all]"
    echo ""
    echo "Workload SM allocations:"
    echo "  a - 30% (56 SMs)"
    echo "  b - 40% (75 SMs)"
    echo "  c - 50% (94 SMs)"
    echo "  d - 70% (131 SMs)"
    echo ""
    echo "Examples:"
    echo "  $0 a b     # Deploy workload-a and workload-b"
    echo "  $0 all     # Deploy all workloads"
    exit 0
fi

for arg in "$@"; do
    case $arg in
        a|b|c|d)
            deploy_workload "$arg"
            ;;
        all)
            deploy_workload "a"
            deploy_workload "b"
            deploy_workload "c"
            deploy_workload "d"
            ;;
        *)
            echo "Unknown workload: $arg (use a, b, c, d, or all)"
            ;;
    esac
done

echo ""
echo "Waiting for pods to start..."
sleep 3

echo ""
echo "=== Pod Status ==="
kubectl get pods -l app=sm-test-workload -o wide

echo ""
echo "=== Monitor logs with: ==="
echo "kubectl logs -f workload-a"
echo "kubectl logs -f workload-b"
echo "kubectl logs -f workload-c"
echo "kubectl logs -f workload-d"
echo ""
echo "Or watch all: watch 'kubectl logs --tail=5 workload-a workload-b workload-c workload-d 2>/dev/null'"
