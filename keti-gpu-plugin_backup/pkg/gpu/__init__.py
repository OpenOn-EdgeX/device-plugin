"""
GPU Module - GPU resource tracking (HAMI-like)

- NVML 모니터링 사용하지 않음 (금마에서 담당)
- 할당 정보 추적만 담당
"""

import logging

logger = logging.getLogger(__name__)


class GPUTracker:
    """
    GPU Resource Tracker (HAMI-like)

    - NVML 사용 안함 왜냐 우리껄로 개발해야하니까. 시공간분할 KETI 
    - 할당 정보만 추적
    - 
    """

    def __init__(self):
        # 노드의 GPU 정보 (설정에서 가져오거나 환경변수로 받음)
        self.gpu_count = 1  # 기본값
        self.total_memory_mb = 0  # 나중에 설정
        self.total_cores_percent = 100

        # 현재 할당 상태
        self.allocated_memory_mb = 0
        self.allocated_cores_percent = 0

        logger.info("GPUTracker initialized (HAMI-like mode)")

    def set_gpu_info(self, gpu_count: int, total_memory_mb: int):
        """GPU 정보 설정 (외부에서 설정)"""
        self.gpu_count = gpu_count
        self.total_memory_mb = total_memory_mb
        logger.info(f"GPU info set: {gpu_count} GPU(s), {total_memory_mb}MB total memory")

    def allocate(self, memory_mb: int, cores_percent: int) -> bool:
        """리소스 할당 시도"""
        new_memory = self.allocated_memory_mb + memory_mb
        new_cores = self.allocated_cores_percent + cores_percent

        # 오버커밋 허용하지 않음
        if self.total_memory_mb > 0 and new_memory > self.total_memory_mb:
            logger.warning(f"Memory overcommit: {new_memory}MB > {self.total_memory_mb}MB")
            return False

        if new_cores > self.total_cores_percent:
            logger.warning(f"Cores overcommit: {new_cores}% > {self.total_cores_percent}%")
            return False

        self.allocated_memory_mb = new_memory
        self.allocated_cores_percent = new_cores
        return True

    def release(self, memory_mb: int, cores_percent: int):
        """리소스 해제"""
        self.allocated_memory_mb = max(0, self.allocated_memory_mb - memory_mb)
        self.allocated_cores_percent = max(0, self.allocated_cores_percent - cores_percent)

    def get_status(self) -> dict:
        """현재 할당 상태"""
        return {
            "gpu_count": self.gpu_count,
            "total_memory_mb": self.total_memory_mb,
            "allocated_memory_mb": self.allocated_memory_mb,
            "available_memory_mb": self.total_memory_mb - self.allocated_memory_mb if self.total_memory_mb > 0 else 0,
            "total_cores_percent": self.total_cores_percent,
            "allocated_cores_percent": self.allocated_cores_percent,
            "available_cores_percent": self.total_cores_percent - self.allocated_cores_percent,
        }


__all__ = ["GPUTracker"]
