"""
NPU Device Management Module

NPU 디바이스 탐지 및 관리
현재는 placeholder로 가상 디바이스 생성
"""

import os
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from ..config import NPU_DEVICE_COUNT, HEALTHY, UNHEALTHY

logger = logging.getLogger(__name__)


@dataclass
class NPUDevice:
    """NPU 디바이스 정보"""
    id: str
    uuid: str
    memory_mb: int
    health: str
    vendor: str = "Unknown"
    model: str = "Unknown"


class NPUManager:
    """
    NPU Device Manager

    역할:
    - NPU 디바이스 탐지
    - 디바이스 상태 관리
    - Device Plugin에 디바이스 정보 제공
    """

    def __init__(self):
        self.devices: List[NPUDevice] = []
        self.initialized = False

    def init(self) -> bool:
        """
        NPU 디바이스 초기화

        Returns:
            True if successful
        """
        try:
            # 실제 NPU 탐지 시도
            real_devices = self._detect_real_devices()

            if real_devices:
                self.devices = real_devices
                logger.info(f"Detected {len(self.devices)} real NPU device(s)")
            else:
                # Placeholder: 가상 디바이스 생성
                self.devices = self._create_virtual_devices()
                logger.info(f"Created {len(self.devices)} virtual NPU device(s) (placeholder)")

            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize NPU devices: {e}")
            return False

    def _detect_real_devices(self) -> List[NPUDevice]:
        """
        실제 NPU 디바이스 탐지

        TODO: 실제 NPU 하드웨어 탐지 로직 구현
        - Furiosa NPU: /dev/npu* 확인
        - 기타 NPU: 벤더별 SDK 사용
        """
        devices = []

        # Furiosa NPU 탐지 시도
        furiosa_devices = self._detect_furiosa_npu()
        if furiosa_devices:
            devices.extend(furiosa_devices)

        return devices

    def _detect_furiosa_npu(self) -> List[NPUDevice]:
        """Furiosa NPU 탐지"""
        devices = []

        # /dev/npu* 디바이스 확인
        for i in range(8):  # 최대 8개 NPU 검색
            dev_path = f"/dev/npu{i}"
            if os.path.exists(dev_path):
                device = NPUDevice(
                    id=f"npu-{i}",
                    uuid=f"FURIOSA-NPU-{i:04d}",
                    memory_mb=16 * 1024,  # 16GB
                    health=HEALTHY,
                    vendor="FuriosaAI",
                    model="Warboy"
                )
                devices.append(device)
                logger.info(f"Detected Furiosa NPU: {dev_path}")

        return devices

    def _create_virtual_devices(self) -> List[NPUDevice]:
        """가상 NPU 디바이스 생성 (placeholder)"""
        devices = []

        count = NPU_DEVICE_COUNT
        logger.info(f"Creating {count} virtual NPU device(s)")

        for i in range(count):
            device = NPUDevice(
                id=f"npu-{i}",
                uuid=f"VIRTUAL-NPU-{i:04d}",
                memory_mb=16 * 1024,  # 16GB
                health=HEALTHY,
                vendor="KETI",
                model="Virtual-NPU"
            )
            devices.append(device)

        return devices

    def get_devices(self) -> List[NPUDevice]:
        """디바이스 목록 반환"""
        return self.devices

    def get_device(self, device_id: str) -> Optional[NPUDevice]:
        """특정 디바이스 조회"""
        for dev in self.devices:
            if dev.id == device_id:
                return dev
        return None

    def get_healthy_devices(self) -> List[NPUDevice]:
        """Healthy 상태의 디바이스 목록"""
        return [d for d in self.devices if d.health == HEALTHY]

    def set_device_health(self, device_id: str, health: str):
        """디바이스 상태 변경"""
        device = self.get_device(device_id)
        if device:
            device.health = health
            logger.info(f"Device {device_id} health changed to {health}")

    def get_status(self) -> Dict:
        """NPU Manager 상태 요약"""
        return {
            "initialized": self.initialized,
            "total_devices": len(self.devices),
            "healthy_devices": len(self.get_healthy_devices()),
            "devices": [
                {
                    "id": d.id,
                    "uuid": d.uuid,
                    "memory_mb": d.memory_mb,
                    "health": d.health,
                    "vendor": d.vendor,
                    "model": d.model,
                }
                for d in self.devices
            ]
        }

    def print_status(self):
        """상태 출력"""
        status = self.get_status()
        logger.info("=" * 50)
        logger.info("NPU Device Status")
        logger.info("=" * 50)
        logger.info(f"Total: {status['total_devices']}, Healthy: {status['healthy_devices']}")
        for dev in status['devices']:
            logger.info(f"  [{dev['id']}] {dev['vendor']} {dev['model']} - {dev['memory_mb']}MB - {dev['health']}")
        logger.info("=" * 50)


__all__ = ["NPUManager", "NPUDevice"]
