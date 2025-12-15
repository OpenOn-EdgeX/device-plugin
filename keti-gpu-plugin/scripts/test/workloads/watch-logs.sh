#!/bin/bash
# Watch logs from all running SM test workloads in real-time

echo "=== Real-time SM Workload Logs ==="
echo "Press Ctrl+C to exit"
echo ""

# Get running workload pods
PODS=$(kubectl get pods -l app=sm-test-workload -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)

if [ -z "$PODS" ]; then
    echo "No SM test workloads running"
    echo "Deploy with: ./deploy.sh a b"
    exit 1
fi

echo "Watching: $PODS"
echo "=========================================="
echo ""

# Use stern if available, otherwise kubectl
if command -v stern &> /dev/null; then
    stern -l app=sm-test-workload --tail 100
else
    # Fallback: watch kubectl logs
    while true; do
        clear
        echo "=== SM Workload Logs ($(date +%H:%M:%S)) ==="
        echo ""
        for pod in $PODS; do
            echo "--- $pod ---"
            kubectl logs --tail=3 "$pod" 2>/dev/null | grep -E "^\[workload" || echo "(waiting...)"
            echo ""
        done
        sleep 2
    done
fi
