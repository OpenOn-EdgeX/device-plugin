#pragma once
#include "context_manager.hpp"
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>

// Resolve real CUDA Runtime symbols once
struct RealCuda {
    decltype(&cudaLaunchKernel) launchKernel = nullptr;
    decltype(&cudaMalloc)       malloc_      = nullptr;
    decltype(&cudaFree)         free_        = nullptr;
    decltype(&cudaMallocManaged) mallocManaged = nullptr;
    decltype(&cudaStreamCreate) streamCreate = nullptr;
    decltype(&cudaStreamDestroy) streamDestroy = nullptr;
    decltype(&cudaDeviceGetAttribute) deviceGetAttribute = nullptr;

    void resolve();
};

RealCuda& real();

// RAII context switch based on current route
struct ScopedRouteCtx {
    BlessContextManager::Route r;
    ScopedRouteCtx();
    ~ScopedRouteCtx();
};