"""
KETI NPU Device Plugin

Kubernetes Device Plugin gRPC 서버 구현
kubelet과 통신하여 NPU 리소스 관리
"""

import os
import time
import logging
import threading
from concurrent import futures
from typing import List, Dict, Optional

import grpc

from ..config import (
    RESOURCE_NAME, DEVICE_PLUGIN_PATH, KUBELET_SOCKET,
    PLUGIN_SOCKET_NAME, PLUGIN_SOCKET_PATH, PLUGIN_API_VERSION,
    HEALTHY, UNHEALTHY
)
from ..npu import NPUManager

logger = logging.getLogger(__name__)

# Proto imports (generated at runtime)
try:
    from . import api_pb2
    from . import api_pb2_grpc
except ImportError:
    api_pb2 = None
    api_pb2_grpc = None
    logger.warning("Proto files not generated yet")


class KETINPUDevicePlugin:
    """
    KETI NPU Device Plugin

    역할:
    - kubelet/EdgeCore와 gRPC 통신
    - NPU 디바이스 등록 및 할당
    """

    def __init__(self, npu_manager: NPUManager):
        self.npu_manager = npu_manager
        self.server = None
        self.running = False
        self.allocations: Dict[str, dict] = {}

        logger.info(f"KETINPUDevicePlugin initialized")
        logger.info(f"Resource name: {RESOURCE_NAME}")

    def start(self) -> bool:
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
        if self.server:
            self.server.stop(grace=5)
            logger.info("Device Plugin server stopped")

    def get_devices(self) -> List[dict]:
        """Get list of NPU devices for ListAndWatch"""
        devices = []
        for npu in self.npu_manager.get_devices():
            devices.append({
                'id': npu.id,
                'health': npu.health,
            })
        return devices

    def allocate(self, device_ids: List[str]) -> dict:
        """
        Allocate NPU devices

        Args:
            device_ids: List of device IDs to allocate

        Returns:
            Allocation response with envs, mounts, devices
        """
        logger.info(f"Allocating NPU devices: {device_ids}")

        # Get device info
        devices_info = []
        for dev_id in device_ids:
            device = self.npu_manager.get_device(dev_id)
            if device:
                devices_info.append(device)

        # Environment variables
        envs = {
            "KETI_NPU_DEVICE_IDS": ",".join(device_ids),
            "NPU_VISIBLE_DEVICES": ",".join([str(i) for i, _ in enumerate(device_ids)]),
        }

        # Mounts (placeholder - add actual NPU library paths if needed)
        mounts = []

        # Device mappings
        devices = []
        for i, dev_id in enumerate(device_ids):
            # Furiosa NPU device path
            npu_dev_path = f"/dev/npu{i}"
            if os.path.exists(npu_dev_path):
                devices.append({
                    "container_path": npu_dev_path,
                    "host_path": npu_dev_path,
                    "permissions": "rw",
                })

        logger.info(f"NPU Allocation complete: {len(device_ids)} device(s)")

        return {
            "envs": envs,
            "mounts": mounts,
            "devices": devices,
        }


class DevicePluginServicer(api_pb2_grpc.DevicePluginServicer):
    """gRPC Servicer for Device Plugin"""

    def __init__(self, plugin: KETINPUDevicePlugin):
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
                        container_path=d.get('container_path', ''),
                        host_path=d.get('host_path', ''),
                        permissions=d.get('permissions', 'rw'),
                    ) for d in allocation.get('devices', [])
                ],
            )
            container_responses.append(response)

        return api_pb2.AllocateResponse(container_responses=container_responses)

    def PreStartContainer(self, request, context):
        return api_pb2.PreStartContainerResponse()


__all__ = ["KETINPUDevicePlugin", "RESOURCE_NAME"]
