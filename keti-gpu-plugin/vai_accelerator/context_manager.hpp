#pragma once
#include <cuda.h>
#include <atomic>
#include <thread>
#include <vector>
#include <string>
#include <mutex>

// Build contexts with SM affinity using cuCtxCreate_v3 + CUexecAffinityParam.
// Requires: CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1 and MPS daemon.
struct BlessContextManager {

    CUdevice dev = 0;
    int total_sms = 0;

    struct Ctx { CUcontext ctx=nullptr; int sm_count=0; } limited, unlimited;

    enum Route { ROUTE_LIMITED=0, ROUTE_UNLIMITED=1 };
    std::atomic<Route> route{ROUTE_LIMITED};

    std::thread ctrl_thread;
    std::atomic<bool> running{false};

    // ✅ 아래 둘을 *멤버*로 선언 (init()에서 다시 선언하지 말 것!)
    std::atomic<bool> ready{false};
    int ctrl_fd = -1;

    bool init(int sm_percent);
    void teardown();
    void start_control_server(const std::string& sock);
    void stop_control_server();

    inline void pushLimited(){ cuCtxPushCurrent(limited.ctx); }
    inline void pushUnlimited(){ cuCtxPushCurrent(unlimited.ctx); }
    inline void pop(){ CUcontext c; cuCtxPopCurrent(&c); }
};


BlessContextManager& bless(); // singleton accessor