"""
Configuration Module for KETI NPU Plugin

환경변수 및 설정 관리
"""

import os

# =============================================================================
# Device Plugin 설정
# =============================================================================
RESOURCE_NAME = os.environ.get('KETI_RESOURCE_NAME', 'keti.io/npu')

DEVICE_PLUGIN_PATH = "/var/lib/kubelet/device-plugins"
KUBELET_SOCKET = f"{DEVICE_PLUGIN_PATH}/kubelet.sock"
PLUGIN_SOCKET_NAME = "keti-npu.sock"
PLUGIN_SOCKET_PATH = f"{DEVICE_PLUGIN_PATH}/{PLUGIN_SOCKET_NAME}"
PLUGIN_API_VERSION = "v1beta1"

# =============================================================================
# NPU 설정
# =============================================================================
NPU_DEVICE_COUNT = int(os.environ.get('NPU_DEVICE_COUNT', '1'))
DEFAULT_MEMORY_MB = int(os.environ.get('KETI_DEFAULT_MEMORY_MB', '16384'))  # 16GB default

# =============================================================================
# EdgeCore MetaServer 설정
# =============================================================================
EDGECORE_METASERVER = os.environ.get('EDGECORE_METASERVER', 'http://127.0.0.1:10550')
METASERVER_TIMEOUT = int(os.environ.get('METASERVER_TIMEOUT', '15'))

# =============================================================================
# 로깅 설정
# =============================================================================
LOG_LEVEL = os.environ.get('KETI_LOG_LEVEL', 'INFO')

# Device health status
HEALTHY = "Healthy"
UNHEALTHY = "Unhealthy"


def get_config_summary() -> dict:
    """현재 설정 요약"""
    return {
        "resource_name": RESOURCE_NAME,
        "npu_device_count": NPU_DEVICE_COUNT,
        "default_memory_mb": DEFAULT_MEMORY_MB,
        "metaserver": EDGECORE_METASERVER,
    }
