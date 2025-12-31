# NPU Device Plugin

NPU (Neural Processing Unit) resource management for Kubernetes.

## Status

- [x] Directory structure initialized
- [ ] Device discovery implementation
- [ ] Resource allocation logic
- [ ] Kubernetes device plugin interface

## Current State

```
NPU Devices: 1 (placeholder)
Status: Not implemented
```

## Directory Structure

```
npu/
├── pkg/
│   └── npu/
│       └── device.go       # NPU device discovery
├── scripts/
│   └── test/
│       └── workloads/      # Test workload YAMLs
├── deploy/
│   └── daemonset.yaml      # Kubernetes deployment
└── README.md
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NPU_DEVICE_COUNT` | Number of NPU devices | 1 |
| `NPU_MEMORY_LIMIT` | Memory limit per device | - |

## Usage (Planned)

```yaml
resources:
  limits:
    keti.com/npu: 1
```
