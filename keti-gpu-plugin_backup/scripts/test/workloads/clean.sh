#!/bin/bash
# Clean up all SM test workloads

echo "=== Cleaning SM Test Workloads ==="

# Delete all workloads
kubectl delete pod workload-a workload-b workload-c workload-d --ignore-not-found=true 2>/dev/null

# Also delete by label
kubectl delete pods -l app=sm-test-workload --ignore-not-found=true 2>/dev/null

echo ""
echo "Waiting for pods to terminate..."
sleep 2

# Show remaining pods
echo ""
echo "=== Remaining GPU Pods ==="
kubectl get pods -o wide | grep -E "workload|gpu" || echo "No GPU workloads running"

echo ""
echo "Clean complete!"
