"""
Configuration Module

환경변수 및 설정 관리
"""

import os

# =============================================================================
# Device Plugin 설정
# =============================================================================
RESOURCE_NAME = os.environ.get('KETI_RESOURCE_NAME', 'nvidia.com/gpu')
RESOURCE_GPUMEM = os.environ.get('KETI_RESOURCE_GPUMEM', 'nvidia.com/gpumem')
RESOURCE_GPUCORES = os.environ.get('KETI_RESOURCE_GPUCORES', 'nvidia.com/gpucores')

DEVICE_PLUGIN_PATH = "/var/lib/kubelet/device-plugins"
KUBELET_SOCKET = f"{DEVICE_PLUGIN_PATH}/kubelet.sock"
PLUGIN_SOCKET_NAME = "keti-gpu.sock"
PLUGIN_SOCKET_PATH = f"{DEVICE_PLUGIN_PATH}/{PLUGIN_SOCKET_NAME}"
PLUGIN_API_VERSION = "v1beta1"

# =============================================================================
# vGPU 설정
# =============================================================================
VGPU_UNITS = int(os.environ.get('KETI_VGPU_UNITS', '10'))
DEFAULT_MEMORY_MB = int(os.environ.get('KETI_DEFAULT_MEMORY_MB', '1024'))
DEFAULT_CORES_PERCENT = int(os.environ.get('KETI_DEFAULT_CORES_PERCENT', '10'))

# =============================================================================
# EdgeCore MetaServer 설정
# =============================================================================
EDGECORE_METASERVER = os.environ.get('EDGECORE_METASERVER', 'http://127.0.0.1:10550')
METASERVER_TIMEOUT = int(os.environ.get('METASERVER_TIMEOUT', '15'))

# =============================================================================
# KETI Allocator 설정
# =============================================================================
KETI_ALLOCATOR_URL = os.environ.get('KETI_ALLOCATOR_URL', 'http://127.0.0.1:7070')
KETI_ALLOCATOR_ENABLED = os.environ.get('KETI_ALLOCATOR_ENABLED', 'false').lower() == 'true'
KETI_ALLOCATOR_TIMEOUT = int(os.environ.get('KETI_ALLOCATOR_TIMEOUT', '5'))

# =============================================================================
# 로깅 설정
# =============================================================================
LOG_LEVEL = os.environ.get('KETI_LOG_LEVEL', 'INFO')


def get_config_summary() -> dict:
    """현재 설정 요약"""
    return {
        "resource_name": RESOURCE_NAME,
        "vgpu_units": VGPU_UNITS,
        "default_memory_mb": DEFAULT_MEMORY_MB,
        "metaserver": EDGECORE_METASERVER,
        "allocator_url": KETI_ALLOCATOR_URL,
        "allocator_enabled": KETI_ALLOCATOR_ENABLED,
    }
