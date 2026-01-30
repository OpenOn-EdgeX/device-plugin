#!/usr/bin/env python3
"""
GPU 워크로드 - PyTorch 행렬 연산 반복
"""
import sys
import time
import torch

def main():
    wid = sys.argv[1] if len(sys.argv) > 1 else "0"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 15  # 기본 15초

    print(f"[Workload-{wid}] Starting PyTorch GPU workload ({duration}s)...")

    device = torch.device("cuda:0")

    # 큰 행렬 생성
    size = 4096
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    start = time.time()
    iterations = 0

    while time.time() - start < duration:
        # 행렬 곱셈 (GPU 집약적)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        iterations += 1

        if iterations % 50 == 0:
            elapsed = time.time() - start
            print(f"[Workload-{wid}] {elapsed:.1f}s - {iterations} iterations")

    elapsed = time.time() - start
    print(f"[Workload-{wid}] Done! {iterations} iterations in {elapsed:.1f}s")

if __name__ == "__main__":
    main()
