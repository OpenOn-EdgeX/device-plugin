#!/usr/bin/env python3
"""
KETI GPU Webhook Server
Mutating Admission Webhook for GPU Pod Environment Injection

이 서버는 Cloud/Master 노드에서 실행됩니다.
EdgeCore가 Device Plugin의 envs를 지원하지 않기 때문에
API Server 레벨에서 Pod spec에 직접 주입합니다.
"""

import os
import sys
import logging
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pkg.webhook import GPUWebhook

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='KETI GPU Webhook Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8443, help='Port to bind (default: 8443)')
    parser.add_argument('--cert', default='/etc/webhook/certs/tls.crt', help='TLS certificate path')
    parser.add_argument('--key', default='/etc/webhook/certs/tls.key', help='TLS key path')
    parser.add_argument('--no-tls', action='store_true', help='Disable TLS (for testing only)')
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("KETI GPU Webhook Server Starting...")
    logger.info("=" * 50)

    # SSL context
    ssl_context = None
    if not args.no_tls:
        if os.path.exists(args.cert) and os.path.exists(args.key):
            ssl_context = (args.cert, args.key)
            logger.info(f"TLS enabled: cert={args.cert}, key={args.key}")
        else:
            logger.error(f"TLS certificate not found: {args.cert} or {args.key}")
            logger.error("Use --no-tls for testing without TLS")
            sys.exit(1)
    else:
        logger.warning("TLS disabled - NOT FOR PRODUCTION USE")

    # Create and run webhook
    webhook = GPUWebhook()

    logger.info(f"Listening on {args.host}:{args.port}")
    logger.info("=" * 50)

    try:
        webhook.run(host=args.host, port=args.port, ssl_context=ssl_context)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
