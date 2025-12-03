"""
Scheduler Module - Core scheduling logic for GPU resources (KubeEdge 전용)

Device Plugin의 Allocate() 호출 시 할당 정책 결정

Annotation 기반 GPU 할당 (KETI):
- keti.io/gpu-memory: VRAM (MB 또는 Gi)
- keti.io/gpu-cores: SM 코어 사용률 (0-100%)
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Annotation keys (KETI prefix)
ANNOTATION_GPU_MEMORY = "keti.io/gpu-memory"  # VRAM (예: "3000", "3Gi")
ANNOTATION_GPU_CORES = "keti.io/gpu-cores"    # SM 코어 % (예: "30", "50")


@dataclass
class PodAllocation:
    """Pod의 GPU 할당 정보"""
    pod_key: str              # namespace/pod_name
    requested_memory_mb: int  # 요청 VRAM (MB)
    requested_cores: int      # 요청 SM 코어 (%)
    allocated_memory_mb: int  # 할당된 VRAM (MB)
    allocated_cores: int      # 할당된 SM 코어 (%)
    status: str               # pending, allocated, running, released


class ResourceScheduler:
    """
    GPU Resource Scheduler (KETI)

    핵심 역할:
    - Device Plugin의 Allocate() 호출 시 할당 결정
    - Annotation 파싱 (memory, cores)
    - 할당 정책 적용
    """

    def __init__(self):
        self.running = False
        self.allocations: Dict[str, PodAllocation] = {}
        logger.info("ResourceScheduler initialized")

    def start(self):
        """Start the scheduler"""
        self.running = True
        logger.info("ResourceScheduler started")

    def stop(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("ResourceScheduler stopped")

    def parse_memory_value(self, value: str) -> Optional[int]:
        """
        Parse memory value to MB

        Supports: "1024", "1024MB", "1024Mi", "2Gi", "2GB"
        """
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

    def parse_cores_value(self, value: str) -> Optional[int]:
        """
        Parse cores value to percentage (0-100)

        Supports: "30", "30%", "0.3"
        """
        if not value:
            return None

        value = value.strip()

        try:
            if value.endswith("%"):
                return int(value[:-1])
            elif "." in value:
                # 0.3 -> 30%
                return int(float(value) * 100)
            else:
                return int(value)
        except ValueError:
            logger.warning(f"Invalid cores value: {value}")
            return None

    def allocate(
        self,
        device_ids: list,
        num_units: int,
        total_units: int,
        requested_memory_mb: int = None,
        requested_cores_percent: int = None,
        pod_key: str = None
    ) -> dict:
        """
        Device Plugin에서 호출 - 할당 결정

        Args:
            device_ids: 할당된 디바이스 ID 목록
            num_units: 요청된 유닛 수
            total_units: 전체 유닛 수
            requested_memory_mb: Pod annotation에서 읽은 VRAM 요청량 (MB)
            requested_cores_percent: Pod annotation에서 읽은 SM 코어 요청률 (%)
            pod_key: namespace/pod_name

        Returns:
            dict with memory_mb, cores_percent
        """
        import os

        # 1. 기본값 계산 (유닛 기반)
        default_cores_percent = num_units * (100 // total_units)
        default_memory_per_unit = int(os.environ.get('KETI_DEFAULT_MEMORY_MB', '1024'))
        default_memory_mb = default_memory_per_unit * num_units

        # 2. Annotation 값 우선 적용 (Fine-Grained 할당)
        memory_mb = requested_memory_mb if requested_memory_mb else default_memory_mb
        cores_percent = requested_cores_percent if requested_cores_percent else default_cores_percent

        # 3. 리소스 제한 검증
        max_memory_mb = int(os.environ.get('KETI_MAX_MEMORY_MB', '32768'))  # 32GB 기본
        max_cores_percent = 100

        if memory_mb > max_memory_mb:
            logger.warning(f"Requested memory {memory_mb}MB exceeds max {max_memory_mb}MB, capping")
            memory_mb = max_memory_mb

        if cores_percent > max_cores_percent:
            logger.warning(f"Requested cores {cores_percent}% exceeds max {max_cores_percent}%, capping")
            cores_percent = max_cores_percent

        # 4. 할당 기록 저장
        if pod_key:
            allocation = PodAllocation(
                pod_key=pod_key,
                requested_memory_mb=requested_memory_mb or default_memory_mb,
                requested_cores=requested_cores_percent or default_cores_percent,
                allocated_memory_mb=memory_mb,
                allocated_cores=cores_percent,
                status="allocated"
            )
            self.allocations[pod_key] = allocation
            logger.info(f"Recorded allocation for {pod_key}")

        logger.info(f"Scheduler allocate: pod={pod_key}, units={num_units}, "
                   f"memory={memory_mb}MB, cores={cores_percent}%")

        return {
            "memory_mb": memory_mb,
            "cores_percent": cores_percent,
        }

    def release(self, pod_key: str) -> bool:
        """
        Pod 삭제 시 할당 해제

        Args:
            pod_key: namespace/pod_name

        Returns:
            True if released, False if not found
        """
        if pod_key in self.allocations:
            del self.allocations[pod_key]
            logger.info(f"Released allocation for {pod_key}")
            return True
        return False

    def get_total_allocated_memory(self) -> int:
        """Get total allocated VRAM across all pods"""
        return sum(a.allocated_memory_mb for a in self.allocations.values())

    def get_total_allocated_cores(self) -> int:
        """Get total allocated cores across all pods"""
        return sum(a.allocated_cores for a in self.allocations.values())

    def get_allocation(self, pod_key: str) -> Optional[PodAllocation]:
        """Get allocation for a specific pod"""
        return self.allocations.get(pod_key)

    def get_all_allocations(self) -> Dict[str, PodAllocation]:
        """Get all current allocations"""
        return self.allocations.copy()

    def get_status(self) -> dict:
        """Get scheduler status"""
        total_memory = self.get_total_allocated_memory()
        total_cores = self.get_total_allocated_cores()

        return {
            "running": self.running,
            "total_pods": len(self.allocations),
            "total_allocated_memory_mb": total_memory,
            "total_allocated_cores_percent": total_cores,
            "allocations": {k: {
                "requested_memory_mb": v.requested_memory_mb,
                "requested_cores": v.requested_cores,
                "allocated_memory_mb": v.allocated_memory_mb,
                "allocated_cores": v.allocated_cores,
                "status": v.status
            } for k, v in self.allocations.items()},
        }


__all__ = ["ResourceScheduler", "PodAllocation", "ANNOTATION_GPU_MEMORY", "ANNOTATION_GPU_CORES"]
