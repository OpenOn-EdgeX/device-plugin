#!/bin/bash
#
# Build vai_accelerator.so with Ubuntu 22.04 compatibility (GLIBC 2.35)
# Uses Docker to ensure compatibility
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Building libvai_accelerator.so"
echo "  (Ubuntu 22.04 / GLIBC 2.35 compatible)"
echo "=========================================="

# Build using Docker with Ubuntu 22.04 CUDA image
docker run --rm \
    -v "$SCRIPT_DIR:/build" \
    -w /build \
    nvidia/cuda:12.4.1-devel-ubuntu22.04 \
    bash -c '
        echo "Building in container..."
        echo "GLIBC version: $(ldd --version | head -1)"

        # Build with explicit flags to avoid C23 features
        g++ -fPIC -O2 -std=c++17 \
            -D_GLIBCXX_USE_CXX11_ABI=1 \
            -I/usr/local/cuda/include \
            -o libvai_accelerator.so \
            vai_accelerator.cpp \
            -shared \
            -L/usr/local/cuda/lib64 \
            -lcuda -lcudart -ldl -lpthread

        echo ""
        echo "Build complete! Checking GLIBC requirements:"
        objdump -T libvai_accelerator.so | grep -E "GLIBC_2\.(3[0-9]|[4-9][0-9])" || echo "No GLIBC 2.3x+ dependencies (good!)"

        echo ""
        echo "All GLIBC versions used:"
        strings libvai_accelerator.so | grep -E "^GLIBC_" | sort -u
    '

echo ""
echo "=========================================="
echo "  Build Complete!"
echo "=========================================="
echo ""
echo "Output: $SCRIPT_DIR/libvai_accelerator.so"
echo ""
echo "Next: Copy to edge node:"
echo "  scp libvai_accelerator.so edge-gpu-232:/var/lib/keti/"
