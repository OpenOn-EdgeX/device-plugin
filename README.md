# KETI Device Plugin

Kubernetes device plugins for heterogeneous accelerators (GPU, NPU).

## Directory Structure

```
device-plugin/
├── keti-gpu-plugin/               # NVIDIA GPU support
│   ├── vai_accelerator/           # SM partitioning library
│   ├── scripts/test/workloads/    # Test workload YAMLs
│   ├── pkg/                       # Go packages
│   └── deploy/                    # Kubernetes deployments
│
├── keti-npu-plugin/               # NPU support (placeholder)
│   ├── pkg/npu/                   # NPU device management
│   ├── scripts/test/workloads/    # Test workload YAMLs
│   └── deploy/                    # Kubernetes deployments
│
├── keti-ppo-agent/                # PPO-based resource scheduler
│   ├── pkg/agent/                 # PPO Agent (Actor-Critic)
│   ├── pkg/api/                   # REST API server
│   └── deploy/                    # Kubernetes deployments
│
└── README.md
```

## Components

### keti-gpu-plugin

| Component | Status | Description |
|-----------|--------|-------------|
| vai_accelerator.so | ✅ Ready | SM partitioning via cuCtxCreate_v3 |
| ConfigMap | ✅ Ready | vai-accelerator-binary |
| Test workloads | ✅ Ready | workload-a,b,c,d |

**Pod Name:** `keti-gpu-plugin-xxxxx`

**Environment Variables:**
- `BLESS_LIMIT_PCT`: SM limit percentage (1-99)
- `LD_PRELOAD`: Path to libvai_accelerator.so

### keti-npu-plugin

| Component | Status | Description |
|-----------|--------|-------------|
| Device discovery | 🔲 Skeleton | Placeholder (1 device) |
| Resource allocation | 🔲 Skeleton | Not implemented |
| Device plugin | 🔲 Skeleton | DaemonSet template only |

**Pod Name:** `keti-npu-plugin-xxxxx`

**Environment Variables:**
- `NPU_DEVICE_COUNT`: Number of NPU devices (default: 1)

### keti-ppo-agent

| Component | Status | Description |
|-----------|--------|-------------|
| PPO Agent | ✅ Ready | Actor-Critic based GPU resource allocation |
| REST API | ✅ Ready | /allocate, /partition, /health endpoints |
| MPS Integration | ✅ Ready | NVIDIA MPS partition lookup |

**Pod Name:** `keti-ppo-agent-xxxxx`

**Environment Variables:**
- `API_PORT`: Agent API port (default: 8080)
- `CUDA_MPS_PIPE_DIRECTORY`: MPS pipe path
- `GPU_UUID`: Target GPU UUID

## Quick Start

### GPU Workloads

```bash
cd keti-gpu-plugin/scripts/test/workloads
./deploy.sh a b
kubectl logs -f workload-a
./clean.sh
```

### NPU Workloads

```bash
kubectl apply -f keti-npu-plugin/scripts/test/workloads/npu-test.yaml
kubectl logs npu-test
```

## Kubernetes Pods

```
$ kubectl get pods -n kube-system | grep keti

keti-gpu-plugin-xxxxx    Running
keti-npu-plugin-xxxxx    Running
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐      ┌─────────────────────┐          │
│  │   GPU Node          │      │   NPU Node          │          │
│  │                     │      │                     │          │
│  │  ┌───────────────┐  │      │  ┌───────────────┐  │          │
│  │  │keti-gpu-plugin│  │      │  │keti-npu-plugin│  │          │
│  │  │ (vai_accel.so)│  │      │  │ (placeholder) │  │          │
│  │  └───────────────┘  │      │  └───────────────┘  │          │
│  │                     │      │                     │          │
│  │  ┌───────────────┐  │      │  ┌───────────────┐  │          │
│  │  │ NVIDIA GPU    │  │      │  │ NPU Device    │  │          │
│  │  │ RTX PRO 6000  │  │      │  │ (TBD)         │  │          │
│  │  │ 188 SMs       │  │      │  │               │  │          │
│  │  └───────────────┘  │      │  └───────────────┘  │          │
│  └─────────────────────┘      └─────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
# device-plugin

## Contributors

| Name | Role |
|------|------|
| corehun | Project Lead |
| DrakwonJ | Developer |
| joingi99 | Developer |
| KilJuHyun | Developer |
| Kim-HanGyeol | Developer |
| Nick-Stokes | Developer |
| Sim-suyeon | Developer |
## Configuration
