"""
Device Plugin Module - Kubernetes/KubeEdge Device Plugin for KETI GPU

kubelet/EdgeCore가 컨테이너 시작 전에 호출
- 순서 보장: Allocate() 응답 후에야 컨테이너 시작
- 환경변수 주입으로 KETI-Core에 할당 정보 전달
- EdgeCore MetaServer 조회로 Pod annotation 획득
- KETI Allocator 연동으로 실제 SM 할당
"""

import os
import logging
import grpc
import time
import threading
import requests
from concurrent import futures
from typing import List, Dict, Optional

from ..config import (
    RESOURCE_NAME, RESOURCE_GPUMEM, RESOURCE_GPUCORES,
    DEVICE_PLUGIN_PATH, KUBELET_SOCKET, PLUGIN_SOCKET_NAME, PLUGIN_SOCKET_PATH,
    PLUGIN_API_VERSION, VGPU_UNITS, DEFAULT_MEMORY_MB, DEFAULT_CORES_PERCENT,
    EDGECORE_METASERVER, METASERVER_TIMEOUT
)
from ..allocator import KETIAllocatorClient

logger = logging.getLogger(__name__)

# Device health
HEALTHY = "Healthy"
UNHEALTHY = "Unhealthy"

# Import generated protobuf code
try:
    from . import api_pb2
    from . import api_pb2_grpc
except ImportError:
    logger.warning("Proto files not generated yet. Run generate_proto.sh first.")
    api_pb2 = None
    api_pb2_grpc = None


class KETIDevicePlugin:
    """
    KETI GPU Device Plugin

    - kubelet/EdgeCore에 등록
    - GPU 디바이스 목록 제공
    - Allocate() 호출 시 할당 결정 및 환경변수 주입
    - KETI Allocator와 연동하여 실제 SM 할당
    """

    def __init__(self, scheduler=None, allocator: KETIAllocatorClient = None):
        """
        Args:
            scheduler: Resource Scheduler for internal allocation logic
            allocator: KETI Allocator client for actual SM allocation
        """
        self.scheduler = scheduler
        self.allocator = allocator or KETIAllocatorClient()
        self.server = None
        self.running = False

        # GPU devices (virtual vGPU units)
        self.devices = []
        self._init_devices()

        # 할당 추적 (release를 위해)
        self.allocations: Dict[str, dict] = {}

        logger.info(f"KETIDevicePlugin initialized with {len(self.devices)} virtual GPU units")
        logger.info(f"KETI Allocator: {'enabled' if self.allocator.enabled else 'disabled'}")

    def _init_devices(self):
        """Initialize virtual GPU devices"""
        num_vgpu_units = VGPU_UNITS

        self.devices = []
        for i in range(num_vgpu_units):
            device = {
                'id': f"keti-gpu-{i}",
                'health': HEALTHY,
                'cores_percent': 100 // num_vgpu_units,
            }
            self.devices.append(device)

        logger.info(f"Initialized {num_vgpu_units} vGPU units")

    def start(self):
        """Start the Device Plugin gRPC server"""
        if api_pb2 is None or api_pb2_grpc is None:
            logger.error("Proto files not generated. Cannot start Device Plugin.")
            return False

        # Remove old socket if exists
        if os.path.exists(PLUGIN_SOCKET_PATH):
            os.remove(PLUGIN_SOCKET_PATH)

        # Create gRPC server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        # Add Device Plugin service
        api_pb2_grpc.add_DevicePluginServicer_to_server(
            DevicePluginServicer(self), self.server
        )

        # Listen on Unix socket
        self.server.add_insecure_port(f"unix://{PLUGIN_SOCKET_PATH}")
        self.server.start()
        self.running = True

        logger.info(f"Device Plugin server started on {PLUGIN_SOCKET_PATH}")

        # Register with kubelet
        self._register_with_kubelet()

        return True

    def _register_with_kubelet(self):
        """Register with kubelet/EdgeCore"""
        if not os.path.exists(KUBELET_SOCKET):
            logger.warning(f"Kubelet socket not found: {KUBELET_SOCKET}")
            logger.warning("Will retry registration when kubelet is ready")
            threading.Thread(target=self._retry_registration, daemon=True).start()
            return

        self._do_register()

    def _retry_registration(self):
        """Retry registration with kubelet"""
        while self.running:
            if os.path.exists(KUBELET_SOCKET):
                try:
                    self._do_register()
                    return
                except Exception as e:
                    logger.error(f"Registration failed: {e}")
            time.sleep(5)

    def _do_register(self):
        """Actually register with kubelet"""
        try:
            channel = grpc.insecure_channel(f"unix://{KUBELET_SOCKET}")
            stub = api_pb2_grpc.RegistrationStub(channel)

            request = api_pb2.RegisterRequest(
                version=PLUGIN_API_VERSION,
                endpoint=PLUGIN_SOCKET_NAME,
                resource_name=RESOURCE_NAME,
                options=api_pb2.DevicePluginOptions(
                    pre_start_required=False,
                    get_preferred_allocation_available=False,
                )
            )

            stub.Register(request)
            logger.info(f"Registered with kubelet: {RESOURCE_NAME}")

        except Exception as e:
            logger.error(f"Failed to register with kubelet: {e}")
            raise

    def stop(self):
        """Stop the Device Plugin server"""
        self.running = False

        # Release all allocations
        for tenant in list(self.allocations.keys()):
            self._release_allocation(tenant)

        if self.server:
            self.server.stop(grace=5)
            logger.info("Device Plugin server stopped")

        if os.path.exists(PLUGIN_SOCKET_PATH):
            os.remove(PLUGIN_SOCKET_PATH)

    def get_devices(self):
        """Get list of devices for ListAndWatch"""
        return self.devices

    # =========================================================================
    # MetaServer 조회
    # =========================================================================

    def _query_pending_pods(self) -> List[dict]:
        """
        EdgeCore MetaServer에서 Pending 상태의 Pod 목록 조회
        """
        try:
            url = f"{EDGECORE_METASERVER}/api/v1/pods"
            response = requests.get(url, timeout=METASERVER_TIMEOUT)

            if response.status_code != 200:
                logger.warning(f"MetaServer query failed: {response.status_code}")
                return []

            data = response.json()
            pods = data.get('items', [])

            # Pending 상태이고 GPU 리소스를 요청하는 Pod 필터링
            pending_pods = []
            for pod in pods:
                status = pod.get('status', {}).get('phase', '')

                if status in ['Pending', 'ContainerCreating', '']:
                    containers = pod.get('spec', {}).get('containers', [])
                    for container in containers:
                        limits = container.get('resources', {}).get('limits', {})
                        if RESOURCE_NAME in limits:
                            pending_pods.append(pod)
                            break

            logger.debug(f"Found {len(pending_pods)} pending GPU pods")
            return pending_pods

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to query MetaServer: {e}")
            return []
        except Exception as e:
            logger.error(f"Error querying pending pods: {e}")
            return []

    def _find_pod_by_gpu_request(self, num_units: int) -> dict:
        """
        요청된 GPU 유닛 수와 일치하는 Pod 찾기
        """
        pending_pods = self._query_pending_pods()

        for pod in pending_pods:
            containers = pod.get('spec', {}).get('containers', [])
            for container in containers:
                limits = container.get('resources', {}).get('limits', {})
                gpu_request = limits.get(RESOURCE_NAME, 0)

                if isinstance(gpu_request, str):
                    gpu_request = int(gpu_request)

                if gpu_request == num_units:
                    pod_name = pod.get('metadata', {}).get('name', 'unknown')
                    namespace = pod.get('metadata', {}).get('namespace', 'default')
                    logger.info(f"Found matching pod: {namespace}/{pod_name} (units={num_units})")
                    return pod

        logger.warning(f"No matching pod found for {num_units} GPU units")
        return {}

    def _get_pod_gpu_config(self, pod: dict) -> dict:
        """
        Pod에서 GPU 설정 추출 (annotation에서 읽기)
        """
        result = {
            'memory_mb': None,
            'cores_percent': None,
            'pod_key': None,
            'pod_name': None,
            'namespace': None,
        }

        if not pod:
            return result

        pod_name = pod.get('metadata', {}).get('name', '')
        namespace = pod.get('metadata', {}).get('namespace', 'default')
        result['pod_key'] = f"{namespace}/{pod_name}"
        result['pod_name'] = pod_name
        result['namespace'] = namespace

        annotations = pod.get('metadata', {}).get('annotations', {})

        # nvidia.com/gpumem (annotation)
        gpumem_str = annotations.get(RESOURCE_GPUMEM, '')
        if gpumem_str:
            result['memory_mb'] = self._parse_memory(str(gpumem_str))
            logger.info(f"Pod annotation {RESOURCE_GPUMEM}: {gpumem_str} -> {result['memory_mb']}MB")

        # nvidia.com/gpucores (annotation)
        gpucores_str = annotations.get(RESOURCE_GPUCORES, '')
        if gpucores_str:
            result['cores_percent'] = self._parse_cores(str(gpucores_str))
            logger.info(f"Pod annotation {RESOURCE_GPUCORES}: {gpucores_str} -> {result['cores_percent']}%")

        return result

    def _parse_memory(self, value: str) -> Optional[int]:
        """Parse memory value to MB"""
        if not value:
            return None

        value = value.strip().upper()

        try:
            if value.isdigit():
                return int(value)

            if value.endswith("GI") or value.endswith("GB"):
                return int(value[:-2]) * 1024
            elif value.endswith("MI") or value.endswith("MB"):
                return int(value[:-2])
            elif value.endswith("G"):
                return int(value[:-1]) * 1024
            elif value.endswith("M"):
                return int(value[:-1])
            else:
                return int(value)
        except ValueError:
            logger.warning(f"Invalid memory value: {value}")
            return None

    def _parse_cores(self, value: str) -> Optional[int]:
        """Parse cores value to percentage"""
        if not value:
            return None

        value = value.strip()

        try:
            if value.endswith("%"):
                return int(value[:-1])
            elif "." in value:
                return int(float(value) * 100)
            else:
                return int(value)
        except ValueError:
            logger.warning(f"Invalid cores value: {value}")
            return None

    # =========================================================================
    # 할당 로직
    # =========================================================================

    def allocate(self, device_ids: List[str]) -> dict:
        """
        Allocate devices - called before container start

        1. EdgeCore MetaServer 조회로 요청 Pod 찾기
        2. Pod annotation에서 GPU 설정 읽기
        3. KETI Allocator에 SM 할당 요청
        4. 환경변수 반환
        """
        logger.info(f"Allocate called for devices: {device_ids}")

        num_units = len(device_ids)

        # 1. EdgeCore MetaServer에서 요청 Pod 찾기
        pod = self._find_pod_by_gpu_request(num_units)

        # 2. Pod annotation에서 GPU 설정 읽기
        pod_config = self._get_pod_gpu_config(pod)

        # 3. 기본값 설정
        default_cores_percent = num_units * (100 // len(self.devices))
        default_memory_mb = DEFAULT_MEMORY_MB * num_units

        memory_mb = pod_config.get('memory_mb') or default_memory_mb
        cores_percent = pod_config.get('cores_percent') or default_cores_percent
        pod_key = pod_config.get('pod_key')

        # 4. Scheduler에 할당 결정 위임
        if self.scheduler:
            allocation = self.scheduler.allocate(
                device_ids=device_ids,
                num_units=num_units,
                total_units=len(self.devices),
                requested_memory_mb=memory_mb,
                requested_cores_percent=cores_percent,
                pod_key=pod_key
            )
            memory_mb = allocation['memory_mb']
            cores_percent = allocation['cores_percent']

        # 5. KETI Allocator에 SM 할당 요청
        allocator_result = self._request_keti_allocation(
            tenant=pod_key or f"unknown-{time.time()}",
            cores_percent=cores_percent
        )

        # 할당 추적
        if pod_key:
            self.allocations[pod_key] = {
                'device_ids': device_ids,
                'memory_mb': memory_mb,
                'cores_percent': cores_percent,
                'allocator_result': allocator_result,
            }

        logger.info(f"Allocation: memory={memory_mb}MB, cores={cores_percent}% (pod={pod_key})")

        # Environment variables for KETI-Core
        envs = {
            "KETI_GPU_MEMORY_LIMIT": str(memory_mb),
            "KETI_GPU_CORES_LIMIT": str(cores_percent),
            "KETI_GPU_DEVICE_IDS": ",".join(device_ids),
            "NVIDIA_VISIBLE_DEVICES": "0",
        }

        # KETI Allocator SM 정보 추가
        if allocator_result.get("ok") and allocator_result.get("sm_count"):
            envs["KETI_GPU_SM_COUNT"] = str(allocator_result["sm_count"])

        # KETI vai_accelerator.so SM 분할 (MPS 대신 CUDA Context 기반)
        # BLESS_LIMIT_PCT: vai_accelerator.so가 사용하는 SM 비율 환경변수
        if cores_percent and cores_percent > 0:
            envs["BLESS_LIMIT_PCT"] = str(int(cores_percent))
            envs["LD_PRELOAD"] = "/var/lib/keti/libvai_accelerator.so"
            logger.info(f"KETI SM limit set: BLESS_LIMIT_PCT={int(cores_percent)}%, LD_PRELOAD=vai_accelerator.so")

        # vai_accelerator.so 마운트
        mounts = []
        if os.path.exists("/var/lib/keti/libvai_accelerator.so"):
            mounts.append({
                "container_path": "/var/lib/keti/libvai_accelerator.so",
                "host_path": "/var/lib/keti/libvai_accelerator.so",
                "read_only": True,
            })

        return {
            "envs": envs,
            "mounts": mounts,
            "devices": [
                {
                    "container_path": "/dev/nvidia0",
                    "host_path": "/dev/nvidia0",
                    "permissions": "rw",
                },
                {
                    "container_path": "/dev/nvidiactl",
                    "host_path": "/dev/nvidiactl",
                    "permissions": "rw",
                },
                {
                    "container_path": "/dev/nvidia-uvm",
                    "host_path": "/dev/nvidia-uvm",
                    "permissions": "rw",
                },
            ],
        }

    def _request_keti_allocation(self, tenant: str, cores_percent: int) -> dict:
        """
        KETI Allocator에 SM 할당 요청
        """
        if not self.allocator.enabled:
            return {"ok": True, "skipped": True}

        result = self.allocator.allocate(
            tenant=tenant,
            sm_percent=cores_percent,
            gpu_id=0
        )

        if result.get("ok"):
            logger.info(f"KETI Allocator: allocated {result.get('sm_count')} SMs for {tenant}")
        else:
            logger.warning(f"KETI Allocator failed for {tenant}: {result.get('error')}")

        return result

    def _release_allocation(self, tenant: str):
        """
        할당 해제
        """
        if tenant in self.allocations:
            del self.allocations[tenant]

        if self.allocator.enabled:
            result = self.allocator.release(tenant)
            if result.get("ok"):
                logger.info(f"KETI Allocator: released {tenant}")


# =============================================================================
# gRPC Servicer
# =============================================================================

if api_pb2_grpc is not None:
    class DevicePluginServicer(api_pb2_grpc.DevicePluginServicer):
        """gRPC Servicer for Device Plugin"""

        def __init__(self, plugin: KETIDevicePlugin):
            self.plugin = plugin

        def GetDevicePluginOptions(self, request, context):
            return api_pb2.DevicePluginOptions(
                pre_start_required=False,
                get_preferred_allocation_available=False,
            )

        def ListAndWatch(self, request, context):
            logger.info("ListAndWatch called")

            while self.plugin.running:
                devices = []
                for dev in self.plugin.get_devices():
                    devices.append(api_pb2.Device(
                        ID=dev['id'],
                        health=dev['health'],
                    ))

                yield api_pb2.ListAndWatchResponse(devices=devices)
                time.sleep(10)

        def Allocate(self, request, context):
            logger.info("Allocate called")

            container_responses = []

            for container_req in request.container_requests:
                device_ids = list(container_req.devicesIDs)
                allocation = self.plugin.allocate(device_ids)

                # 디버깅: 반환할 내용 로깅
                logger.info(f"[gRPC Response] envs: {allocation['envs']}")
                logger.info(f"[gRPC Response] mounts: {allocation.get('mounts', [])}")
                logger.info(f"[gRPC Response] devices: {allocation.get('devices', [])}")

                response = api_pb2.ContainerAllocateResponse(
                    envs=allocation['envs'],
                    mounts=[
                        api_pb2.Mount(
                            container_path=m.get('container_path', ''),
                            host_path=m.get('host_path', ''),
                            read_only=m.get('read_only', False),
                        ) for m in allocation.get('mounts', [])
                    ],
                    devices=[
                        api_pb2.DeviceSpec(
                            container_path=d['container_path'],
                            host_path=d['host_path'],
                            permissions=d['permissions'],
                        ) for d in allocation.get('devices', [])
                    ],
                )
                container_responses.append(response)

            return api_pb2.AllocateResponse(container_responses=container_responses)

        def PreStartContainer(self, request, context):
            return api_pb2.PreStartContainerResponse()

        def GetPreferredAllocation(self, request, context):
            container_responses = []
            for container_req in request.container_requests:
                available = list(container_req.available_deviceIDs)
                size = container_req.allocation_size
                preferred = available[:size] if size <= len(available) else available

                container_responses.append(
                    api_pb2.ContainerPreferredAllocationResponse(deviceIDs=preferred)
                )

            return api_pb2.PreferredAllocationResponse(container_responses=container_responses)


__all__ = ["KETIDevicePlugin", "RESOURCE_NAME"]
