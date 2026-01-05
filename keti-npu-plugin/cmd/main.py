#!/usr/bin/env python3
"""
KETI NPU Plugin for KubeEdge
Entry point for the DaemonSet application

구성요소:
1. Device Plugin - kubelet/EdgeCore와 통신
2. NPU Manager - NPU 디바이스 탐지 및 관리
"""

import os
import sys
import signal
import logging
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pkg.config import LOG_LEVEL, get_config_summary
from pkg.npu import NPUManager

# Device Plugin
try:
    from pkg.device_plugin import KETINPUDevicePlugin, RESOURCE_NAME
    DEVICE_PLUGIN_AVAILABLE = True
except ImportError as e:
    DEVICE_PLUGIN_AVAILABLE = False
    logging.warning(f"Device Plugin not available: {e}")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Application:
    """Main application class"""

    def __init__(self):
        self.running = True
        self.npu_manager = None
        self.device_plugin = None

        # Initialize NPU Manager
        logger.info("Initializing NPU Manager...")
        self.npu_manager = NPUManager()
        if not self.npu_manager.init():
            raise RuntimeError("Failed to initialize NPU Manager")

        # Initialize Device Plugin
        self._init_device_plugin()

    def _init_device_plugin(self):
        """Initialize Device Plugin"""
        if not DEVICE_PLUGIN_AVAILABLE:
            raise RuntimeError("Device Plugin not available - proto files not generated?")

        logger.info("Initializing Device Plugin...")
        self.device_plugin = KETINPUDevicePlugin(npu_manager=self.npu_manager)

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
        logger.info("KETI NPU Plugin Starting...")
        logger.info("=" * 50)

        # Print config summary
        config = get_config_summary()
        logger.info(f"Resource name: {config['resource_name']}")
        logger.info(f"NPU device count: {config['npu_device_count']}")
        logger.info(f"MetaServer: {config['metaserver']}")

        # Print node info
        node_name = os.environ.get('NODE_NAME', 'unknown')
        pod_name = os.environ.get('POD_NAME', 'unknown')
        logger.info(f"Node: {node_name}")
        logger.info(f"Pod: {pod_name}")

        # Print NPU device status
        self.npu_manager.print_status()

        # Start Device Plugin
        logger.info("Starting Device Plugin...")
        if self.device_plugin.start():
            logger.info(f"Device Plugin registered: {RESOURCE_NAME}")
        else:
            logger.error("Device Plugin failed to start")
            return

        logger.info("=" * 50)
        logger.info("KETI NPU Plugin is running!")
        logger.info("=" * 50)
        logger.info(f"Resource name: {RESOURCE_NAME}")
        logger.info("Pods can request NPU with:")
        logger.info(f"  resources.limits.{RESOURCE_NAME}: <count>")

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
        status = self.npu_manager.get_status()
        logger.debug(f"NPU Status: {status['total_devices']} devices, "
                    f"{status['healthy_devices']} healthy")

    def _shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down...")
        if self.device_plugin:
            self.device_plugin.stop()
        logger.info("KETI NPU Plugin stopped.")


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
