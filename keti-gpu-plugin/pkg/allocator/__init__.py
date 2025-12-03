"""
KETI Allocator Module

KETI Resource Allocator와 통신하여 실제 GPU SM 할당을 수행
- /allocate: SM 할당 요청
- /release: 할당 해제
- /state: 상태 조회
"""

from .client import KETIAllocatorClient

__all__ = ["KETIAllocatorClient"]
