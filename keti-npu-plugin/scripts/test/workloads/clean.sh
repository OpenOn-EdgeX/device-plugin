#!/bin/bash
# Clean up NPU test workloads

echo "=== Cleaning NPU Test Workloads ==="

kubectl delete pod -l app=npu-test-workload --ignore-not-found

echo ""
echo "=== Cleanup Complete ==="
kubectl get pods -l app=npu-test-workload 2>/dev/null || echo "No NPU test pods found"
