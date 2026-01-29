"""
Watcher Module - KubeEdge Pod watcher

KubeEdge 전용:
- EdgeCore MetaServer API 사용 (로컬 HTTP API)
- 오프라인에서도 로컬 캐시된 정보 사용 가능
"""

import logging
import threading
import time
import os

logger = logging.getLogger(__name__)

# EdgeCore MetaServer 설정
EDGECORE_HOST = os.environ.get('EDGECORE_HOST', '127.0.0.1')
EDGECORE_PORT = os.environ.get('EDGECORE_PORT', '10550')
EDGECORE_METASERVER_URL = f"http://{EDGECORE_HOST}:{EDGECORE_PORT}"

# Try to import requests
REQUESTS_AVAILABLE = False
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    logger.error("requests library not available - required for EdgeCore API")


class PodWatcher:
    """
    KubeEdge Pod Watcher (EdgeCore 전용)

    - EdgeCore MetaServer API 사용 (http://127.0.0.1:10550)
    - 로컬에서 Pod 정보 조회 (네트워크 독립적)
    - 오프라인 지원
    """

    def __init__(self):
        self.running = False
        self.thread = None
        self.edgecore_available = False
        self.node_name = os.environ.get('NODE_NAME', '')

        # Callbacks (set by application)
        self.on_pod_added = None    # callback(pod: dict)
        self.on_pod_deleted = None  # callback(pod_key: str)

        self._init_edgecore()
        logger.info(f"PodWatcher initialized (edgecore_available: {self.edgecore_available})")

    def _init_edgecore(self):
        """Initialize EdgeCore MetaServer connection"""
        if not REQUESTS_AVAILABLE:
            logger.error("Cannot initialize: requests library required")
            return

        if self._check_edgecore():
            self.edgecore_available = True
            logger.info(f"EdgeCore MetaServer available at {EDGECORE_METASERVER_URL}")
        else:
            logger.warning(f"EdgeCore MetaServer not available at {EDGECORE_METASERVER_URL}")

    def _check_edgecore(self) -> bool:
        """Check if EdgeCore MetaServer is available"""
        try:
            url = f"{EDGECORE_METASERVER_URL}/api/v1/pods"
            response = requests.get(url, timeout=3)
            return response.status_code in [200, 404]
        except Exception as e:
            logger.debug(f"EdgeCore MetaServer check failed: {e}")
            return False

    def start(self):
        """Start watching for pods"""
        self.running = True
        self.thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()
        logger.info("PodWatcher started")

    def stop(self):
        """Stop watching for pods"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("PodWatcher stopped")

    def _watch_loop(self):
        """Main watch loop"""
        if self.edgecore_available:
            self._edgecore_watch()
        else:
            self._wait_for_edgecore()

    def _wait_for_edgecore(self):
        """Wait for EdgeCore to become available"""
        logger.info("Waiting for EdgeCore MetaServer...")

        while self.running:
            if self._check_edgecore():
                logger.info("EdgeCore MetaServer is now available")
                self.edgecore_available = True
                self._edgecore_watch()
                return
            time.sleep(10)

    def _edgecore_watch(self):
        """
        EdgeCore MetaServer Polling

        - 로컬 HTTP API로 Pod 정보 조회
        - Polling 방식 (watch 미지원)
        - 오프라인에서도 캐시된 정보 사용
        """
        logger.info("Starting EdgeCore MetaServer watch")
        previous_pods = set()

        while self.running:
            try:
                pods = self._get_pods_from_edgecore()

                current_pods = set()
                for pod in pods:
                    pod_name = pod.get('metadata', {}).get('name', 'unknown')
                    namespace = pod.get('metadata', {}).get('namespace', 'default')
                    pod_key = f"{namespace}/{pod_name}"
                    current_pods.add(pod_key)

                    if pod_key not in previous_pods:
                        self._handle_pod_added(pod)

                for pod_key in previous_pods - current_pods:
                    self._handle_pod_deleted(pod_key)

                previous_pods = current_pods
                time.sleep(5)

            except Exception as e:
                logger.error(f"Error in EdgeCore watch: {e}")
                time.sleep(5)

    def _get_pods_from_edgecore(self) -> list:
        """Get pods from EdgeCore MetaServer"""
        try:
            if self.node_name:
                url = f"{EDGECORE_METASERVER_URL}/api/v1/pods?fieldSelector=spec.nodeName={self.node_name}"
            else:
                url = f"{EDGECORE_METASERVER_URL}/api/v1/pods"

            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                return data.get('items', [])
            else:
                logger.warning(f"EdgeCore returned status {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Failed to get pods from EdgeCore: {e}")
            return []

    def _handle_pod_added(self, pod: dict):
        """Handle pod added event"""
        metadata = pod.get('metadata', {})
        pod_name = metadata.get('name', 'unknown')
        namespace = metadata.get('namespace', 'default')

        logger.info(f"Pod added: {namespace}/{pod_name}")

        # Call callback if set
        if self.on_pod_added:
            try:
                self.on_pod_added(pod)
            except Exception as e:
                logger.error(f"Error in on_pod_added callback: {e}")

    def _handle_pod_deleted(self, pod_key: str):
        """Handle pod deleted event"""
        logger.info(f"Pod deleted: {pod_key}")

        # Call callback if set
        if self.on_pod_deleted:
            try:
                self.on_pod_deleted(pod_key)
            except Exception as e:
                logger.error(f"Error in on_pod_deleted callback: {e}")

    def get_pods(self) -> list:
        """Get current pods on this node"""
        if self.edgecore_available:
            return self._get_pods_from_edgecore()
        return []


__all__ = ["PodWatcher", "EDGECORE_METASERVER_URL"]
