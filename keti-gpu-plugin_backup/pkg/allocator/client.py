"""
KETI Allocator Client

KETI Resource Allocator HTTP API와 통신
- SM(Streaming Multiprocessor) 기반 GPU 자원 할당
- 실시간 할당/해제 요청
"""

import os
import logging
import requests
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# KETI Allocator 기본 설정
DEFAULT_ALLOCATOR_URL = "http://127.0.0.1:7070"


class KETIAllocatorClient:
    """
    KETI Resource Allocator HTTP Client

    KETI-Core의 Resource Allocator와 통신하여
    실제 GPU SM 할당을 수행
    """

    def __init__(self, base_url: str = None):
        """
        Args:
            base_url: KETI Allocator URL (default: http://127.0.0.1:7070)
        """
        self.base_url = base_url or os.environ.get(
            'KETI_ALLOCATOR_URL',
            DEFAULT_ALLOCATOR_URL
        )
        self.timeout = int(os.environ.get('KETI_ALLOCATOR_TIMEOUT', '5'))
        self.enabled = os.environ.get('KETI_ALLOCATOR_ENABLED', 'true').lower() == 'true'

        if self.enabled:
            logger.info(f"KETIAllocatorClient initialized: {self.base_url}")
        else:
            logger.info("KETIAllocatorClient disabled")

    def allocate(
        self,
        tenant: str,
        sm_percent: Optional[float] = None,
        sm_count: Optional[int] = None,
        gpu_id: int = 0,
        priority: Optional[str] = None,
        goal: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        GPU SM 할당 요청

        Args:
            tenant: 테넌트 ID (일반적으로 pod_key: namespace/pod_name)
            sm_percent: SM 할당 비율 (%) - sm_count와 택일
            sm_count: SM 개수 - sm_percent와 택일
            gpu_id: 대상 GPU ID (default: 0)
            priority: 우선순위 ("HIGH", "MED", "LOW")
            goal: 목표 설정 (deadline_ms, target_qps 등)

        Returns:
            dict: 할당 결과
            {
                "ok": bool,
                "tenant": str,
                "gpu_id": int,
                "sm_count": int,
                "error": str (실패 시)
            }
        """
        if not self.enabled:
            logger.debug("Allocator disabled, skipping allocation")
            return {"ok": True, "skipped": True, "reason": "allocator_disabled"}

        # 요청 데이터 구성
        payload = {
            "tenant": tenant,
            "gpu_id": gpu_id,
        }

        if sm_percent is not None:
            payload["sm"] = f"{sm_percent}%"
        elif sm_count is not None:
            payload["sm_count"] = sm_count
        else:
            # 기본값: 10%
            payload["sm"] = "10%"
            logger.warning(f"No SM spec provided for {tenant}, using default 10%")

        if priority:
            payload["priority"] = priority

        if goal:
            payload["goal"] = goal

        try:
            url = f"{self.base_url}/allocate"
            logger.info(f"Allocating SM for {tenant}: {payload}")

            response = requests.post(url, json=payload, timeout=self.timeout)
            result = response.json()

            if response.status_code == 200 and result.get("ok"):
                logger.info(f"Allocation success: {tenant} -> GPU {result.get('gpu_id')}, "
                           f"SM count: {result.get('sm_count')}")
                return result
            else:
                error = result.get("err", f"HTTP {response.status_code}")
                logger.error(f"Allocation failed for {tenant}: {error}")
                return {"ok": False, "error": error}

        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Cannot connect to KETI Allocator at {self.base_url}: {e}")
            return {"ok": False, "error": "connection_failed", "detail": str(e)}
        except requests.exceptions.Timeout:
            logger.warning(f"KETI Allocator timeout for {tenant}")
            return {"ok": False, "error": "timeout"}
        except Exception as e:
            logger.error(f"KETI Allocator error: {e}")
            return {"ok": False, "error": str(e)}

    def release(self, tenant: str) -> Dict[str, Any]:
        """
        GPU SM 할당 해제

        Args:
            tenant: 테넌트 ID

        Returns:
            dict: 해제 결과
        """
        if not self.enabled:
            return {"ok": True, "skipped": True, "reason": "allocator_disabled"}

        try:
            url = f"{self.base_url}/release"
            logger.info(f"Releasing SM for {tenant}")

            response = requests.post(
                url,
                json={"tenant": tenant},
                timeout=self.timeout
            )
            result = response.json()

            if response.status_code == 200 and result.get("ok"):
                logger.info(f"Release success: {tenant}")
                return result
            else:
                error = result.get("err", f"HTTP {response.status_code}")
                logger.warning(f"Release failed for {tenant}: {error}")
                return {"ok": False, "error": error}

        except requests.exceptions.ConnectionError:
            logger.warning(f"Cannot connect to KETI Allocator for release")
            return {"ok": False, "error": "connection_failed"}
        except Exception as e:
            logger.error(f"KETI Allocator release error: {e}")
            return {"ok": False, "error": str(e)}

    def get_state(self) -> Dict[str, Any]:
        """
        현재 할당 상태 조회

        Returns:
            dict: GPU 및 테넌트 상태
            {
                "gpus": {gpu_id: {"total_sms", "used_sms", ...}},
                "tenants": {tenant_id: {"gpu_id", "sm_count", ...}}
            }
        """
        if not self.enabled:
            return {"ok": False, "error": "allocator_disabled"}

        try:
            url = f"{self.base_url}/state"
            response = requests.get(url, timeout=self.timeout)

            if response.status_code == 200:
                return response.json()
            else:
                return {"ok": False, "error": f"HTTP {response.status_code}"}

        except requests.exceptions.ConnectionError:
            return {"ok": False, "error": "connection_failed"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def health_check(self) -> bool:
        """
        KETI Allocator 연결 상태 확인

        Returns:
            bool: 연결 가능 여부
        """
        if not self.enabled:
            return False

        try:
            response = requests.get(
                f"{self.base_url}/state",
                timeout=2
            )
            return response.status_code == 200
        except:
            return False
