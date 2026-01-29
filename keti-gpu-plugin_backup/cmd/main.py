#!/usr/bin/env python3
"""
KETI GPU Plugin for KubeEdge
Entry point for the DaemonSet application

구성요소:
1. Device Plugin - kubelet/EdgeCore와 통신 (순서 보장)
2. Resource Scheduler - GPU 할당 정책/로직 (우리 핵심!)

Watcher 제거됨 - Device Plugin이 kubelet 연동 담당
GPUTracker 제거됨 - kubelet이 디바이스 할당 상태 관리
"""

import os
import sys
import signal
import logging
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pkg.scheduler import ResourceScheduler

# Device Plugin
try:
    from pkg.device_plugin import KETIDevicePlugin, RESOURCE_NAME
    DEVICE_PLUGIN_AVAILABLE = True
except ImportError as e:
    DEVICE_PLUGIN_AVAILABLE = False
    logging.warning(f"Device Plugin not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Application:
    """Main application class"""

    def __init__(self):
        self.running = True

        # Deploy vai_accelerator.so for KETI SM partitioning
        self._deploy_vai_accelerator()

        # Initialize Resource Scheduler (핵심 로직)
        logger.info("Initializing Resource Scheduler...")
        self.scheduler = ResourceScheduler()

        # Initialize Device Plugin
        self._init_device_plugin()

    def _deploy_vai_accelerator(self):
        """Copy vai_accelerator.so to host for GPU pods to use"""
        import shutil
        src = "/app/libvai_accelerator.so"
        dst_dir = "/var/lib/keti"
        dst = f"{dst_dir}/libvai_accelerator.so"

        try:
            os.makedirs(dst_dir, exist_ok=True)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                os.chmod(dst, 0o755)
                logger.info(f"Deployed vai_accelerator.so to {dst}")
            else:
                logger.warning(f"vai_accelerator.so not found at {src}")
        except Exception as e:
            logger.warning(f"Failed to deploy vai_accelerator.so: {e}")

    def _init_device_plugin(self):
        """Initialize Device Plugin after scheduler is ready"""
        # Initialize Device Plugin (kubelet 연동)
        if not DEVICE_PLUGIN_AVAILABLE:
            raise RuntimeError("Device Plugin not available - proto files not generated?")

        logger.info("Initializing Device Plugin...")
        self.device_plugin = KETIDevicePlugin(scheduler=self.scheduler)

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def run(self):
        """Main run loop"""
        logger.info("=" * 50)
        logger.info("KETI GPU Plugin Starting...")
        logger.info("=" * 50)

        # Print node info
        node_name = os.environ.get('NODE_NAME', 'unknown')
        pod_name = os.environ.get('POD_NAME', 'unknown')
        logger.info(f"Node: {node_name}")
        logger.info(f"Pod: {pod_name}")

        # Start Device Plugin
        logger.info("Starting Device Plugin...")
        if self.device_plugin.start():
            logger.info(f"Device Plugin registered: {RESOURCE_NAME}")
        else:
            logger.error("Device Plugin failed to start")
            return

        # Start Scheduler
        self.scheduler.start()

        logger.info("=" * 50)
        logger.info("KETI GPU Plugin is running!")
        logger.info("=" * 50)
        logger.info(f"Resource name: {RESOURCE_NAME}")
        logger.info("Pods can request GPU with:")
        logger.info(f"  resources.limits.{RESOURCE_NAME}: <units>")
        logger.info("  (1 unit = 10% GPU)")

        # Main loop
        while self.running:
            try:
                self._periodic_status()
                time.sleep(30)
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)

        # Cleanup
        self._shutdown()

    def _periodic_status(self):
        """Periodic status logging"""
        status = self.scheduler.get_status()
        if status['total_pods'] > 0:
            logger.info(f"Status: {status['total_pods']} GPU pods, "
                       f"memory={status['total_allocated_memory_mb']}MB, "
                       f"cores={status['total_allocated_cores_percent']}%")

    def _shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down...")
        self.device_plugin.stop()
        self.scheduler.stop()
        logger.info("KETI GPU Plugin stopped.")


def main():
    """Entry point"""
    import traceback
    try:
        app = Application()
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
