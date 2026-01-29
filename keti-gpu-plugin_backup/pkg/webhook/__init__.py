"""
KETI GPU Webhook - Mutating Admission Webhook for GPU Pod Injection

EdgeCore가 Device Plugin의 envs/mounts를 지원하지 않아서
API Server 레벨에서 Pod spec에 직접 주입합니다.

동작:
1. Pod 생성 요청 가로챔
2. annotation 확인 (nvidia.com/gpucores, nvidia.com/gpumem)
3. LD_PRELOAD, BLESS_LIMIT_PCT 환경변수 추가
4. vai_accelerator.so 볼륨 마운트 추가
"""

import os
import json
import base64
import logging
import copy
import requests
from flask import Flask, request, jsonify
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# Kubernetes API for node IP lookup
try:
    from kubernetes import client, config
    try:
        config.load_incluster_config()
        K8S_AVAILABLE = True
    except:
        try:
            config.load_kube_config()
            K8S_AVAILABLE = True
        except:
            K8S_AVAILABLE = False
except ImportError:
    K8S_AVAILABLE = False
    logger.warning("kubernetes client not available, will use nodeName directly")

# Configuration
VAI_ACCELERATOR_PATH = os.environ.get('VAI_ACCELERATOR_PATH', '/var/lib/keti/libvai_accelerator.so')
VAI_KETI_DIR = os.environ.get('VAI_KETI_DIR', '/var/lib/keti')  # Directory for shared state
ANNOTATION_GPUCORES = os.environ.get('ANNOTATION_GPUCORES', 'nvidia.com/gpucores')
ANNOTATION_GPUMEM = os.environ.get('ANNOTATION_GPUMEM', 'nvidia.com/gpumem')
GPU_RESOURCE_NAME = os.environ.get('GPU_RESOURCE_NAME', 'nvidia.com/gpu')

# Original value annotation prefix (원본 값 저장용)
ORIGINAL_ANNOTATION_PREFIX = 'keti.io/original-'

# Resource modification settings
# nvidia.com/gpu 요청을 실제 디바이스 수로 변환할지 여부
MODIFY_GPU_REQUEST = os.environ.get('MODIFY_GPU_REQUEST', 'true').lower() == 'true'
# 변환 후 GPU 디바이스 수 (기본: 1)
GPU_DEVICE_COUNT = int(os.environ.get('GPU_DEVICE_COUNT', '1'))

# PPO Agent settings
PPO_AGENT_ENABLED = os.environ.get('PPO_AGENT_ENABLED', 'true').lower() == 'true'
PPO_AGENT_PORT = int(os.environ.get('PPO_AGENT_PORT', '8080'))
PPO_AGENT_TIMEOUT = float(os.environ.get('PPO_AGENT_TIMEOUT', '2.0'))  # seconds
# PPO Agent는 각 Edge 노드에서 DaemonSet으로 실행 (HostNetwork)
# Webhook은 Pod의 nodeName으로 직접 호출


class GPUWebhook:
    """Mutating Admission Webhook for GPU Pod Injection"""

    def __init__(self):
        self.app = Flask(__name__)
        self._setup_routes()
        self._node_ip_cache: Dict[str, str] = {}  # nodeName -> IP cache
        logger.info("GPUWebhook initialized")
        logger.info(f"  VAI_ACCELERATOR_PATH: {VAI_ACCELERATOR_PATH}")
        logger.info(f"  ANNOTATION_GPUCORES: {ANNOTATION_GPUCORES}")
        logger.info(f"  GPU_RESOURCE_NAME: {GPU_RESOURCE_NAME}")
        logger.info(f"  PPO_AGENT_ENABLED: {PPO_AGENT_ENABLED}")
        logger.info(f"  K8S_AVAILABLE: {K8S_AVAILABLE}")
        if PPO_AGENT_ENABLED:
            logger.info(f"  PPO_AGENT_PORT: {PPO_AGENT_PORT} (uses node IP:port)")

    def _setup_routes(self):
        """Setup Flask routes"""
        self.app.add_url_rule('/mutate', 'mutate', self.mutate, methods=['POST'])
        self.app.add_url_rule('/health', 'health', self.health, methods=['GET'])
        self.app.add_url_rule('/ready', 'ready', self.ready, methods=['GET'])

    def health(self):
        """Health check endpoint"""
        return jsonify({"status": "healthy"})

    def ready(self):
        """Readiness check endpoint"""
        return jsonify({"status": "ready"})

    def mutate(self):
        """
        Mutating webhook endpoint

        Receives AdmissionReview, modifies Pod spec, returns patched AdmissionReview
        """
        try:
            admission_review = request.get_json()

            if not admission_review:
                return self._admission_response(None, False, "Empty request")

            uid = admission_review.get('request', {}).get('uid', '')
            pod = admission_review.get('request', {}).get('object', {})

            # Check if this is a Pod
            kind = admission_review.get('request', {}).get('kind', {}).get('kind', '')
            if kind != 'Pod':
                return self._admission_response(uid, True, "Not a Pod, skipping")

            # Check if Pod needs GPU injection
            if not self._needs_injection(pod):
                logger.debug(f"Pod does not need GPU injection")
                return self._admission_response(uid, True, "No GPU annotation found")

            # Generate patches
            patches = self._generate_patches(pod)

            if patches:
                logger.info(f"Injecting GPU envs into Pod: {len(patches)} patches")
                patch_bytes = json.dumps(patches).encode('utf-8')
                patch_base64 = base64.b64encode(patch_bytes).decode('utf-8')
                return self._admission_response(uid, True, "GPU envs injected", patch_base64)
            else:
                return self._admission_response(uid, True, "No patches needed")

        except Exception as e:
            logger.error(f"Error processing admission request: {e}")
            return self._admission_response('', False, str(e))

    def _get_node_ip(self, node_name: str) -> Optional[str]:
        """
        Kubernetes Node의 Internal IP 조회

        Args:
            node_name: 노드 이름

        Returns:
            Internal IP 또는 None
        """
        # 캐시 확인
        if node_name in self._node_ip_cache:
            return self._node_ip_cache[node_name]

        if not K8S_AVAILABLE:
            logger.warning(f"K8S client not available, cannot resolve IP for {node_name}")
            return None

        try:
            v1 = client.CoreV1Api()
            node = v1.read_node(node_name)

            for addr in node.status.addresses:
                if addr.type == "InternalIP":
                    self._node_ip_cache[node_name] = addr.address
                    logger.info(f"Resolved {node_name} -> {addr.address}")
                    return addr.address

            logger.warning(f"No InternalIP found for node {node_name}")
            return None

        except Exception as e:
            logger.warning(f"Failed to get IP for node {node_name}: {e}")
            return None

    def _call_ppo_agent(self, node_name: str, requested_cores: int, requested_memory: int) -> Optional[Tuple[int, int]]:
        """
        PPO Agent에게 자원 할당 추천 요청

        Args:
            node_name: Pod가 스케줄된 노드 이름 (PPO Agent 호출용)
            requested_cores: 요청한 GPU 코어 %
            requested_memory: 요청한 메모리 MB

        Returns:
            (allocated_cores, allocated_memory) or None if failed
        """
        if not PPO_AGENT_ENABLED:
            return None

        if not node_name:
            logger.warning("No nodeName specified, skipping PPO Agent call")
            return None

        # Node IP 조회
        node_ip = self._get_node_ip(node_name)
        if not node_ip:
            logger.warning(f"Could not resolve IP for node {node_name}, skipping PPO Agent call")
            return None

        try:
            # PPO Agent는 각 Edge 노드에서 HostNetwork로 실행
            # Node IP:port로 직접 호출
            url = f"http://{node_ip}:{PPO_AGENT_PORT}/allocate"
            payload = {
                "requested_cores": requested_cores,
                "requested_memory": requested_memory
            }
            logger.info(f"Calling PPO Agent at {url} (node: {node_name})")

            response = requests.post(
                url,
                json=payload,
                timeout=PPO_AGENT_TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                allocated_cores = data.get('allocated_cores', requested_cores)
                allocated_memory = data.get('allocated_memory', requested_memory)
                confidence = data.get('confidence', 0)
                reason = data.get('reason', '')

                logger.info(f"PPO Agent response: {requested_cores}%/{requested_memory}MB -> "
                           f"{allocated_cores}%/{allocated_memory}MB (confidence={confidence:.2f}, {reason})")

                return (allocated_cores, allocated_memory)
            else:
                logger.warning(f"PPO Agent returned {response.status_code}: {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.warning(f"PPO Agent timeout after {PPO_AGENT_TIMEOUT}s")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"PPO Agent connection error: {e}")
            return None
        except Exception as e:
            logger.error(f"PPO Agent call failed: {e}")
            return None

    def _needs_injection(self, pod: dict) -> bool:
        """Check if Pod needs GPU environment injection"""
        annotations = pod.get('metadata', {}).get('annotations', {})

        # Check for GPU cores annotation
        if ANNOTATION_GPUCORES in annotations:
            return True

        # Check for GPU memory annotation
        if ANNOTATION_GPUMEM in annotations:
            return True

        # Check if Pod requests GPU resources
        containers = pod.get('spec', {}).get('containers', [])
        for container in containers:
            limits = container.get('resources', {}).get('limits', {})
            if GPU_RESOURCE_NAME in limits:
                return True

        return False

    def _generate_original_annotation_patches(self, pod: dict) -> list:
        """
        원본 요청 값을 annotation에 저장하는 패치 생성

        저장되는 값:
        - keti.io/original-gpu-request: GPU 리소스 요청 수 (예: "6")
        - keti.io/original-gpucores: SM 코어 % (예: "60")
        - keti.io/original-gpumem: 메모리 MB (예: "8192")
        """
        patches = []
        annotations = pod.get('metadata', {}).get('annotations', {})

        # annotation이 없으면 생성
        if not pod.get('metadata', {}).get('annotations'):
            patches.append({
                "op": "add",
                "path": "/metadata/annotations",
                "value": {}
            })

        original_values = {}

        # 1. GPU 리소스 요청 값 저장
        containers = pod.get('spec', {}).get('containers', [])
        for container in containers:
            limits = container.get('resources', {}).get('limits', {})
            if GPU_RESOURCE_NAME in limits:
                gpu_request = str(limits[GPU_RESOURCE_NAME])
                original_values['gpu-request'] = gpu_request
                break

        # 2. gpucores annotation 값 저장
        if ANNOTATION_GPUCORES in annotations:
            cores_value = annotations[ANNOTATION_GPUCORES]
            original_values['gpucores'] = cores_value.replace('%', '').strip()

        # 3. gpumem annotation 값 저장
        if ANNOTATION_GPUMEM in annotations:
            mem_value = annotations[ANNOTATION_GPUMEM]
            original_values['gpumem'] = mem_value.strip()

        # 패치 생성 (이미 저장된 값은 덮어쓰지 않음)
        for key, value in original_values.items():
            annotation_key = f"{ORIGINAL_ANNOTATION_PREFIX}{key}"
            # 이미 원본이 저장되어 있으면 스킵 (중복 mutation 방지)
            if annotation_key not in annotations:
                # JSON Patch에서 /는 ~1로 escape 필요
                escaped_key = annotation_key.replace('/', '~1')
                patches.append({
                    "op": "add",
                    "path": f"/metadata/annotations/{escaped_key}",
                    "value": value
                })
                logger.info(f"Saving original value: {annotation_key}={value}")

        return patches

    def _generate_resource_patches(self, pod: dict) -> list:
        """
        GPU 리소스 요청 값을 수정하는 패치 생성

        예시:
        - 원본: nvidia.com/gpu: 6 (60% SM 의미)
        - 수정: nvidia.com/gpu: 1 (실제 디바이스 1개)

        SM 제한은 VAI_LIMIT_PCT 환경변수로 적용됨
        """
        if not MODIFY_GPU_REQUEST:
            return []

        patches = []
        containers = pod.get('spec', {}).get('containers', [])

        for i, container in enumerate(containers):
            limits = container.get('resources', {}).get('limits', {})

            if GPU_RESOURCE_NAME in limits:
                original_value = limits[GPU_RESOURCE_NAME]

                # 이미 수정된 값이면 스킵 (값이 GPU_DEVICE_COUNT와 같으면)
                try:
                    if int(original_value) == GPU_DEVICE_COUNT:
                        continue
                except (ValueError, TypeError):
                    pass

                # GPU 리소스 값을 실제 디바이스 수로 변경
                # JSON Patch에서 .은 특별한 의미가 없지만, /는 ~1로 escape
                escaped_resource = GPU_RESOURCE_NAME.replace('/', '~1')
                patches.append({
                    "op": "replace",
                    "path": f"/spec/containers/{i}/resources/limits/{escaped_resource}",
                    "value": str(GPU_DEVICE_COUNT)
                })
                logger.info(f"Modifying GPU request: {original_value} -> {GPU_DEVICE_COUNT}")

        return patches

    def _generate_patches(self, pod: dict) -> list:
        """Generate JSON patches for Pod mutation"""
        patches = []
        annotations = pod.get('metadata', {}).get('annotations', {})

        # 1. 원본 값 저장 (가장 먼저!)
        original_patches = self._generate_original_annotation_patches(pod)
        patches.extend(original_patches)

        # 2. GPU 리소스 요청 수정 (예: 6 -> 1)
        resource_patches = self._generate_resource_patches(pod)
        patches.extend(resource_patches)

        # Get GPU cores percentage from annotation or calculate from GPU request
        cores_percent = annotations.get(ANNOTATION_GPUCORES, '')
        if cores_percent:
            # Remove % if present
            cores_percent = cores_percent.replace('%', '').strip()
        else:
            # annotation 없으면 GPU 요청 값에서 계산 (예: 6 -> 60%)
            containers = pod.get('spec', {}).get('containers', [])
            for container in containers:
                limits = container.get('resources', {}).get('limits', {})
                if GPU_RESOURCE_NAME in limits:
                    try:
                        gpu_request = int(limits[GPU_RESOURCE_NAME])
                        # 1 unit = 10%
                        cores_percent = str(gpu_request * 10)
                        logger.info(f"Calculated cores_percent from GPU request: {gpu_request} -> {cores_percent}%")
                    except (ValueError, TypeError):
                        pass
                    break

        # Get GPU memory from annotation (for logging/future use)
        gpu_mem = annotations.get(ANNOTATION_GPUMEM, '')

        if not cores_percent:
            # Default to 100% if no annotation but GPU requested
            cores_percent = '100'

        # 3. PPO Agent에게 최적 할당 요청
        # Pod의 nodeName 추출 (스케줄된 노드)
        node_name = pod.get('spec', {}).get('nodeName', '')
        requested_cores = int(cores_percent) if cores_percent else 100
        requested_memory = int(gpu_mem) if gpu_mem else 16384  # 기본 16GB

        ppo_result = self._call_ppo_agent(node_name, requested_cores, requested_memory)

        if ppo_result:
            allocated_cores, allocated_memory = ppo_result

            # PPO가 추천한 값으로 annotation 수정
            if allocated_cores != requested_cores:
                escaped_key = ANNOTATION_GPUCORES.replace('/', '~1')
                patches.append({
                    "op": "replace" if ANNOTATION_GPUCORES in annotations else "add",
                    "path": f"/metadata/annotations/{escaped_key}",
                    "value": str(allocated_cores)
                })
                cores_percent = str(allocated_cores)  # 환경변수에도 적용
                logger.info(f"PPO modified gpucores: {requested_cores} -> {allocated_cores}")

            if allocated_memory != requested_memory and gpu_mem:
                escaped_key = ANNOTATION_GPUMEM.replace('/', '~1')
                patches.append({
                    "op": "replace" if ANNOTATION_GPUMEM in annotations else "add",
                    "path": f"/metadata/annotations/{escaped_key}",
                    "value": str(allocated_memory)
                })
                gpu_mem = str(allocated_memory)  # 환경변수에도 적용
                logger.info(f"PPO modified gpumem: {requested_memory} -> {allocated_memory}")

        logger.info(f"GPU injection: cores={cores_percent}%, mem={gpu_mem}")

        # Process each container
        containers = pod.get('spec', {}).get('containers', [])
        for i, container in enumerate(containers):
            # Check if container requests GPU
            limits = container.get('resources', {}).get('limits', {})
            if GPU_RESOURCE_NAME not in limits:
                continue

            # Add environment variables
            env_patches = self._generate_env_patches(i, container, cores_percent, gpu_mem)
            patches.extend(env_patches)

            # Add volume mounts
            mount_patches = self._generate_mount_patches(i, container)
            patches.extend(mount_patches)

        # Add volume for vai_accelerator.so
        if patches:
            volume_patches = self._generate_volume_patches(pod)
            patches.extend(volume_patches)

        return patches

    def _generate_env_patches(self, container_idx: int, container: dict, cores_percent: str, gpu_mem: str = '') -> list:
        """Generate patches for environment variables"""
        patches = []
        existing_env = container.get('env', [])

        # Check if env array exists
        if not existing_env:
            # Create env array
            patches.append({
                "op": "add",
                "path": f"/spec/containers/{container_idx}/env",
                "value": []
            })

        # vai_accelerator.so가 사용하는 환경변수
        env_vars = [
            {"name": "LD_PRELOAD", "value": VAI_ACCELERATOR_PATH},
            {"name": "KETI_SM_LIMIT", "value": cores_percent},  # SM 제한 %
        ]

        # Memory 제한 추가
        if gpu_mem:
            env_vars.append({"name": "KETI_MEM_LIMIT", "value": gpu_mem})

        # Add each env var
        for env_var in env_vars:
            # Check if already exists
            exists = any(e.get('name') == env_var['name'] for e in existing_env)
            if not exists:
                if existing_env:
                    patches.append({
                        "op": "add",
                        "path": f"/spec/containers/{container_idx}/env/-",
                        "value": env_var
                    })
                else:
                    patches.append({
                        "op": "add",
                        "path": f"/spec/containers/{container_idx}/env/-",
                        "value": env_var
                    })

        return patches

    def _generate_mount_patches(self, container_idx: int, container: dict) -> list:
        """Generate patches for volume mounts"""
        patches = []
        existing_mounts = container.get('volumeMounts', [])

        # Check if volumeMounts array exists
        if not existing_mounts:
            patches.append({
                "op": "add",
                "path": f"/spec/containers/{container_idx}/volumeMounts",
                "value": []
            })

        # Check if mount already exists (check both old file path and new dir path)
        mount_exists = any(
            m.get('mountPath') in (VAI_ACCELERATOR_PATH, VAI_KETI_DIR)
            for m in existing_mounts
        )

        if not mount_exists:
            # Mount the entire directory instead of just the file
            # This allows shared state file creation for multi-Pod coordination
            mount = {
                "name": "vai-accelerator",
                "mountPath": VAI_KETI_DIR,
                "readOnly": False  # Need write access for shared state file
            }
            patches.append({
                "op": "add",
                "path": f"/spec/containers/{container_idx}/volumeMounts/-",
                "value": mount
            })

        return patches

    def _generate_volume_patches(self, pod: dict) -> list:
        """Generate patches for volumes"""
        patches = []
        existing_volumes = pod.get('spec', {}).get('volumes', [])

        # Check if volumes array exists
        if not existing_volumes:
            patches.append({
                "op": "add",
                "path": "/spec/volumes",
                "value": []
            })

        # Check if volume already exists
        volume_exists = any(
            v.get('name') == 'vai-accelerator'
            for v in existing_volumes
        )

        if not volume_exists:
            # Mount the directory to allow shared state file creation
            volume = {
                "name": "vai-accelerator",
                "hostPath": {
                    "path": VAI_KETI_DIR,
                    "type": "Directory"  # Changed from File to Directory
                }
            }
            patches.append({
                "op": "add",
                "path": "/spec/volumes/-",
                "value": volume
            })

        return patches

    def _admission_response(self, uid: str, allowed: bool, message: str, patch: str = None) -> dict:
        """Build AdmissionReview response"""
        response = {
            "apiVersion": "admission.k8s.io/v1",
            "kind": "AdmissionReview",
            "response": {
                "uid": uid,
                "allowed": allowed,
            }
        }

        if message:
            response["response"]["status"] = {"message": message}

        if patch:
            response["response"]["patchType"] = "JSONPatch"
            response["response"]["patch"] = patch

        return jsonify(response)

    def run(self, host='0.0.0.0', port=8443, ssl_context=None):
        """Run the webhook server"""
        logger.info(f"Starting GPU Webhook server on {host}:{port}")
        self.app.run(host=host, port=port, ssl_context=ssl_context)


def create_app():
    """Create Flask app for WSGI servers"""
    webhook = GPUWebhook()
    return webhook.app


__all__ = ["GPUWebhook", "create_app"]
