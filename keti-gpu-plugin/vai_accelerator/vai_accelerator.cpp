/**
 * vai_accelerator.so - GPU Resource Limiter (HAMi-core style)
 *
 * Token Bucket + Sleep 방식으로 SM/Core 및 Memory 제한
 * - CUexecAffinityParam 대신 커널 실행 전 토큰 확인
 * - 토큰 부족 시 sleep하여 강제 대기
 * - Memory 할당량 추적 및 제한
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <pthread.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <chrono>
#include <algorithm>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

// ============================================================================
// Shared Memory Token Bucket Structure
// 여러 Pod가 하나의 GPU를 공유할 때 토큰을 공유하여 전체 사용률 제한
// ============================================================================
// Host path that is shared across all containers via hostPath volume
// This path is already mounted in containers that use vai_accelerator.so
// 파티션별 분리: VAI_SHM_PATH 환경변수로 오버라이드 가능
//   예: VAI_SHM_PATH=/var/lib/keti/.vai_shared_state_A (파티션 A 전용)
#define VAI_SHM_PATH_DEFAULT "/var/lib/keti/.vai_shared_state"
#define VAI_SHM_MAGIC 0x56414947  // "VAIG"

struct VaiSharedState {
    uint32_t magic;                    // Magic number for validation
    std::atomic<int32_t> process_count; // Number of processes using this GPU
    std::atomic<int32_t> leader_pid;    // PID of the token replenish leader
    std::atomic<int64_t> compute_tokens; // Shared token pool
    std::atomic<int64_t> total_sm_limit; // Total SM limit (sum of all processes)
    int64_t max_tokens;                 // Maximum tokens
    std::atomic<int64_t> total_kernels; // Total kernels executed (all processes)
    pthread_mutex_t shm_mutex;          // Mutex for initialization
};

// Forward declarations
static void ensure_init();
static void rate_limiter_wait(unsigned int grid_size, unsigned int block_size);

// ---------------- Real CUDA symbols ----------------
typedef cudaError_t (*cudaLaunchKernel_fn)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
typedef CUresult (*cuLaunchKernel_fn)(CUfunction, unsigned int, unsigned int, unsigned int,
                                      unsigned int, unsigned int, unsigned int,
                                      unsigned int, CUstream, void**, void**);
typedef CUresult (*cuInit_fn)(unsigned int);
typedef CUresult (*cuDriverGetVersion_fn)(int*);

static cudaLaunchKernel_fn  real_cudaLaunchKernel  = nullptr;
static cuLaunchKernel_fn    real_cuLaunchKernel    = nullptr;
static cuInit_fn            real_cuInit            = nullptr;
static cuDriverGetVersion_fn real_cuDriverGetVersion = nullptr;
static decltype(&cudaMalloc)        real_cudaMalloc        = nullptr;
static decltype(&cudaFree)          real_cudaFree          = nullptr;
static decltype(&cudaMallocManaged) real_cudaMallocManaged = nullptr;
static decltype(&cudaMemcpy)        real_cudaMemcpy        = nullptr;
static decltype(&cudaMemcpyAsync)   real_cudaMemcpyAsync   = nullptr;
static decltype(&cudaStreamCreate)  real_cudaStreamCreate  = nullptr;
static decltype(&cudaStreamDestroy) real_cudaStreamDestroy = nullptr;
static decltype(&cudaGraphLaunch)   real_cudaGraphLaunch   = nullptr;
static decltype(&cudaDeviceSynchronize) real_cudaDeviceSynchronize = nullptr;
static decltype(&cudaStreamSynchronize) real_cudaStreamSynchronize = nullptr;

// Additional launch variants
static cudaLaunchKernel_fn real_cudaLaunchKernel_ptsz = nullptr;
static cuLaunchKernel_fn real_cuLaunchKernel_ptsz = nullptr;

// cuGetProcAddress - HAMi-core style interception
// Remove the CUDA macro that maps cuGetProcAddress -> cuGetProcAddress_v2
#undef cuGetProcAddress

typedef CUresult (*cuGetProcAddress_fn)(const char*, void**, int, cuuint64_t);
typedef CUresult (*cuGetProcAddress_v2_fn)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
static cuGetProcAddress_fn real_cuGetProcAddress = nullptr;
static cuGetProcAddress_v2_fn real_cuGetProcAddress_v2 = nullptr;

// Forward declarations for our hook functions
extern "C" CUresult cuLaunchKernel(CUfunction, unsigned int, unsigned int, unsigned int,
                                   unsigned int, unsigned int, unsigned int,
                                   unsigned int, CUstream, void**, void**);
extern "C" CUresult cuLaunchKernel_ptsz(CUfunction, unsigned int, unsigned int, unsigned int,
                                        unsigned int, unsigned int, unsigned int,
                                        unsigned int, CUstream, void**, void**);
extern "C" cudaError_t cudaMalloc(void **p, size_t n);
extern "C" cudaError_t cudaFree(void *devPtr);
extern "C" CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
                                        CUdriverProcAddressQueryResult *symbolStatus);
extern "C" CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
                                      CUdriverProcAddressQueryResult *symbolStatus);

// Prevent recursion in dlsym
static __thread int in_dlsym = 0;

// The real dlsym - we need to get this VERY early
static void* (*real_dlsym)(void*, const char*) = nullptr;

static void init_real_dlsym() {
    if (!real_dlsym) {
        // Use the internal glibc dlsym lookup
        real_dlsym = (void*(*)(void*, const char*))dlvsym(RTLD_DEFAULT, "dlsym", "GLIBC_2.2.5");
        if (!real_dlsym) {
            real_dlsym = (void*(*)(void*, const char*))dlvsym(RTLD_DEFAULT, "dlsym", "GLIBC_2.34");
        }
    }
}

// dlsym interception - HAMi-core style
extern "C" __attribute__((visibility("default")))
void* dlsym(void* handle, const char* symbol) {
    if (in_dlsym || !symbol) {
        init_real_dlsym();
        return real_dlsym ? real_dlsym(handle, symbol) : nullptr;
    }

    in_dlsym = 1;
    init_real_dlsym();

    // Log all dlsym lookups for CUDA-related symbols
    static int dlsym_log_count = 0;
    if (dlsym_log_count < 200 && symbol) {
        // Log CUDA-related symbols - broader filter
        if (strncmp(symbol, "cu", 2) == 0) {
            fprintf(stderr, "[vai] dlsym: looking up '%s'\n", symbol);
            fflush(stderr);
            dlsym_log_count++;
        }
    }

    void* result = nullptr;

    // Intercept CUDA functions
    // NOTE: We MUST intercept cuGetProcAddress_v2 at dlsym level so we can intercept
    // the function lookup mechanism itself. This is the HAMi-core approach.
    // NOTE: Do NOT intercept cudaMalloc/cudaFree here - they use versioned symbols
    // and intercepting them via dlsym can break CUDA initialization.
    if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
        fprintf(stderr, "[vai] dlsym: intercepting cuGetProcAddress_v2\n");
        result = (void*)cuGetProcAddress_v2;
    }
    else if (strcmp(symbol, "cuGetProcAddress") == 0) {
        fprintf(stderr, "[vai] dlsym: intercepting cuGetProcAddress\n");
        result = (void*)cuGetProcAddress_v2;
    }
    else if (strcmp(symbol, "cuLaunchKernel") == 0) {
        fprintf(stderr, "[vai] dlsym: intercepting cuLaunchKernel\n");
        result = (void*)cuLaunchKernel;
    }
    else if (strcmp(symbol, "cuLaunchKernel_ptsz") == 0) {
        fprintf(stderr, "[vai] dlsym: intercepting cuLaunchKernel_ptsz\n");
        result = (void*)cuLaunchKernel_ptsz;
    }
    // Also intercept cudaLaunchKernel at dlsym level
    else if (strcmp(symbol, "cudaLaunchKernel") == 0) {
        fprintf(stderr, "[vai] dlsym: intercepting cudaLaunchKernel\n");
        extern cudaError_t cudaLaunchKernel(const void*, dim3, dim3, void**, size_t, cudaStream_t);
        result = (void*)cudaLaunchKernel;
    }
    else {
        result = real_dlsym ? real_dlsym(handle, symbol) : nullptr;
    }

    in_dlsym = 0;
    return result;
}

// Load real CUDA functions from actual libraries
static void* libcuda_handle = nullptr;
static void* libcudart_handle = nullptr;

// Helper macro to use real_dlsym
#define REAL_DLSYM(handle, name) (real_dlsym ? real_dlsym(handle, name) : nullptr)

static void resolve_real() {
    init_real_dlsym();  // Ensure real_dlsym is initialized

    // Try to open CUDA libraries if not already open
    // Note: Use RTLD_NOLOAD first to check if already loaded, then try without it
    if (!libcuda_handle) {
        libcuda_handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (!libcuda_handle) {
            libcuda_handle = dlopen("libcuda.so", RTLD_LAZY | RTLD_NOLOAD);
        }
        if (!libcuda_handle) {
            fprintf(stderr, "[vai] Warning: Could not find libcuda.so\n");
        }
    }
    if (!libcudart_handle) {
        libcudart_handle = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_NOLOAD);
        if (!libcudart_handle) {
            libcudart_handle = dlopen("libcudart.so", RTLD_LAZY | RTLD_NOLOAD);
        }
        if (!libcudart_handle) {
            fprintf(stderr, "[vai] Warning: Could not find libcudart.so\n");
        }
    }

    // Resolve from specific libraries using REAL_DLSYM to avoid recursion
    if (!real_cuLaunchKernel) {
        if (libcuda_handle) {
            real_cuLaunchKernel = (cuLaunchKernel_fn)REAL_DLSYM(libcuda_handle, "cuLaunchKernel");
        }
        if (!real_cuLaunchKernel) {
            real_cuLaunchKernel = (cuLaunchKernel_fn)REAL_DLSYM(RTLD_NEXT, "cuLaunchKernel");
        }
    }
    if (!real_cuLaunchKernel_ptsz) {
        if (libcuda_handle) {
            real_cuLaunchKernel_ptsz = (cuLaunchKernel_fn)REAL_DLSYM(libcuda_handle, "cuLaunchKernel_ptsz");
        }
        if (!real_cuLaunchKernel_ptsz) {
            real_cuLaunchKernel_ptsz = (cuLaunchKernel_fn)REAL_DLSYM(RTLD_NEXT, "cuLaunchKernel_ptsz");
        }
    }

    // CUDA Runtime functions
    if (!real_cudaLaunchKernel) {
        if (libcudart_handle) {
            real_cudaLaunchKernel = (cudaLaunchKernel_fn)REAL_DLSYM(libcudart_handle, "cudaLaunchKernel");
        }
        if (!real_cudaLaunchKernel) {
            real_cudaLaunchKernel = (cudaLaunchKernel_fn)REAL_DLSYM(RTLD_NEXT, "cudaLaunchKernel");
        }
    }
    if (!real_cudaLaunchKernel_ptsz) {
        if (libcudart_handle) {
            real_cudaLaunchKernel_ptsz = (cudaLaunchKernel_fn)REAL_DLSYM(libcudart_handle, "cudaLaunchKernel_ptsz");
        }
        if (!real_cudaLaunchKernel_ptsz) {
            real_cudaLaunchKernel_ptsz = (cudaLaunchKernel_fn)REAL_DLSYM(RTLD_NEXT, "cudaLaunchKernel_ptsz");
        }
    }

    // Memory functions
    if (!real_cudaMalloc) {
        if (libcudart_handle) {
            real_cudaMalloc = (decltype(&cudaMalloc))REAL_DLSYM(libcudart_handle, "cudaMalloc");
        }
        if (!real_cudaMalloc) {
            real_cudaMalloc = (decltype(&cudaMalloc))REAL_DLSYM(RTLD_NEXT, "cudaMalloc");
        }
    }
    if (!real_cudaFree) {
        if (libcudart_handle) {
            real_cudaFree = (decltype(&cudaFree))REAL_DLSYM(libcudart_handle, "cudaFree");
        }
        if (!real_cudaFree) {
            real_cudaFree = (decltype(&cudaFree))REAL_DLSYM(RTLD_NEXT, "cudaFree");
        }
    }
    if (!real_cudaMallocManaged) {
        if (libcudart_handle) {
            real_cudaMallocManaged = (decltype(&cudaMallocManaged))REAL_DLSYM(libcudart_handle, "cudaMallocManaged");
        }
        if (!real_cudaMallocManaged) {
            real_cudaMallocManaged = (decltype(&cudaMallocManaged))REAL_DLSYM(RTLD_NEXT, "cudaMallocManaged");
        }
    }
    if (!real_cudaMemcpy) {
        if (libcudart_handle) {
            real_cudaMemcpy = (decltype(&cudaMemcpy))REAL_DLSYM(libcudart_handle, "cudaMemcpy");
        }
        if (!real_cudaMemcpy) {
            real_cudaMemcpy = (decltype(&cudaMemcpy))REAL_DLSYM(RTLD_NEXT, "cudaMemcpy");
        }
    }
    if (!real_cudaMemcpyAsync) {
        if (libcudart_handle) {
            real_cudaMemcpyAsync = (decltype(&cudaMemcpyAsync))REAL_DLSYM(libcudart_handle, "cudaMemcpyAsync");
        }
        if (!real_cudaMemcpyAsync) {
            real_cudaMemcpyAsync = (decltype(&cudaMemcpyAsync))REAL_DLSYM(RTLD_NEXT, "cudaMemcpyAsync");
        }
    }
    if (!real_cudaStreamCreate) {
        if (libcudart_handle) {
            real_cudaStreamCreate = (decltype(&cudaStreamCreate))REAL_DLSYM(libcudart_handle, "cudaStreamCreate");
        }
        if (!real_cudaStreamCreate) {
            real_cudaStreamCreate = (decltype(&cudaStreamCreate))REAL_DLSYM(RTLD_NEXT, "cudaStreamCreate");
        }
    }
    if (!real_cudaStreamDestroy) {
        if (libcudart_handle) {
            real_cudaStreamDestroy = (decltype(&cudaStreamDestroy))REAL_DLSYM(libcudart_handle, "cudaStreamDestroy");
        }
        if (!real_cudaStreamDestroy) {
            real_cudaStreamDestroy = (decltype(&cudaStreamDestroy))REAL_DLSYM(RTLD_NEXT, "cudaStreamDestroy");
        }
    }
    if (!real_cudaGraphLaunch) {
        if (libcudart_handle) {
            real_cudaGraphLaunch = (decltype(&cudaGraphLaunch))REAL_DLSYM(libcudart_handle, "cudaGraphLaunch");
        }
        if (!real_cudaGraphLaunch) {
            real_cudaGraphLaunch = (decltype(&cudaGraphLaunch))REAL_DLSYM(RTLD_NEXT, "cudaGraphLaunch");
        }
    }
    if (!real_cudaDeviceSynchronize) {
        if (libcudart_handle) {
            real_cudaDeviceSynchronize = (decltype(&cudaDeviceSynchronize))REAL_DLSYM(libcudart_handle, "cudaDeviceSynchronize");
        }
        if (!real_cudaDeviceSynchronize) {
            real_cudaDeviceSynchronize = (decltype(&cudaDeviceSynchronize))REAL_DLSYM(RTLD_NEXT, "cudaDeviceSynchronize");
        }
    }
    if (!real_cudaStreamSynchronize) {
        if (libcudart_handle) {
            real_cudaStreamSynchronize = (decltype(&cudaStreamSynchronize))REAL_DLSYM(libcudart_handle, "cudaStreamSynchronize");
        }
        if (!real_cudaStreamSynchronize) {
            real_cudaStreamSynchronize = (decltype(&cudaStreamSynchronize))REAL_DLSYM(RTLD_NEXT, "cudaStreamSynchronize");
        }
    }

    // HAMi-core style: resolve cuGetProcAddress
    if (!real_cuGetProcAddress) {
        if (libcuda_handle) {
            real_cuGetProcAddress = (cuGetProcAddress_fn)REAL_DLSYM(libcuda_handle, "cuGetProcAddress");
        }
        if (!real_cuGetProcAddress) {
            real_cuGetProcAddress = (cuGetProcAddress_fn)REAL_DLSYM(RTLD_NEXT, "cuGetProcAddress");
        }
    }
    if (!real_cuGetProcAddress_v2) {
        if (libcuda_handle) {
            real_cuGetProcAddress_v2 = (cuGetProcAddress_v2_fn)REAL_DLSYM(libcuda_handle, "cuGetProcAddress_v2");
        }
        if (!real_cuGetProcAddress_v2) {
            real_cuGetProcAddress_v2 = (cuGetProcAddress_v2_fn)REAL_DLSYM(RTLD_NEXT, "cuGetProcAddress_v2");
        }
        // Debug: log resolution status
        static int resolve_log = 0;
        if (!resolve_log) {
            fprintf(stderr, "[vai] resolve: real_cuGetProcAddress=%p real_cuGetProcAddress_v2=%p libcuda=%p\n",
                    (void*)real_cuGetProcAddress, (void*)real_cuGetProcAddress_v2, libcuda_handle);
            resolve_log = 1;
        }
    }

    // cuInit and cuDriverGetVersion - called very early
    if (!real_cuInit) {
        if (libcuda_handle) {
            real_cuInit = (cuInit_fn)REAL_DLSYM(libcuda_handle, "cuInit");
        }
        if (!real_cuInit) {
            real_cuInit = (cuInit_fn)REAL_DLSYM(RTLD_NEXT, "cuInit");
        }
    }
    if (!real_cuDriverGetVersion) {
        if (libcuda_handle) {
            real_cuDriverGetVersion = (cuDriverGetVersion_fn)REAL_DLSYM(libcuda_handle, "cuDriverGetVersion");
        }
        if (!real_cuDriverGetVersion) {
            real_cuDriverGetVersion = (cuDriverGetVersion_fn)REAL_DLSYM(RTLD_NEXT, "cuDriverGetVersion");
        }
    }
}

// ---------------- VAI State ----------------
namespace vai {
    static std::atomic<bool> inited{false};
    static pthread_mutex_t   init_mu = PTHREAD_MUTEX_INITIALIZER;

    // GPU info
    static int total_sms = 0;
    static size_t total_memory = 0;

    // === Shared Memory for Multi-Pod Token Sharing ===
    static VaiSharedState* shared_state = nullptr;       // Shared memory pointer
    static int shm_fd = -1;                              // Shared memory file descriptor
    static bool is_leader = false;                       // Is this process the token replenish leader?
    static bool use_shared_memory = true;                // Use shared memory (can be disabled)
    static char shm_path[256] = VAI_SHM_PATH_DEFAULT;   // Shared state 파일 경로 (파티션별 분리 가능)

    // === Token Bucket for SM/Core limiting ===
    // These are now primarily used for local fallback if shared memory fails
    static std::atomic<long long> compute_tokens{0};     // 현재 토큰 (로컬 fallback)
    static long long max_tokens = 0;                     // 최대 토큰 (SM 제한 기반)
    static int sm_limit_pct = 100;                       // SM 제한 퍼센트 (이 프로세스의 제한)
    static std::atomic<long long> kernel_count{0};       // 실행된 커널 수

    // Token replenish thread
    static std::thread token_thread;
    static std::atomic<bool> token_running{false};
    static const int TOKEN_CYCLE_MS = 10;                // 10ms 주기로 토큰 재충전

    // === Memory limiting ===
    static std::atomic<size_t> memory_used{0};           // 현재 사용 중인 메모리
    static size_t memory_limit = 0;                      // 메모리 제한 (bytes)
    static int memory_limit_mb = 0;                      // 메모리 제한 (MB, 설정값)

    // Rate limit sleep config
    static struct timespec sleep_interval = {0, 1000000}; // 1ms sleep

    // Effective SM limit (adjusted based on total demand)
    // If total > 100%, each process gets proportional share
    static int effective_sm_limit_pct = 100;
}

// ---------------- Shared Memory Functions ----------------

// Initialize shared memory for multi-Pod token sharing
// Uses a file in /var/lib/keti/ which is mounted as hostPath in all containers
static bool init_shared_memory() {
    // Check if shared memory is disabled
    if (const char* e = getenv("VAI_DISABLE_SHARED_MEMORY")) {
        if (strcmp(e, "1") == 0 || strcmp(e, "true") == 0) {
            vai::use_shared_memory = false;
            fprintf(stderr, "[vai] Shared memory disabled by environment\n");
            return false;
        }
    }

    // 파티션별 shared state 경로 설정 (VAI_SHM_PATH 환경변수)
    if (const char* e = getenv("VAI_SHM_PATH")) {
        snprintf(vai::shm_path, sizeof(vai::shm_path), "%s", e);
        fprintf(stderr, "[vai] Using custom shared state path: %s\n", vai::shm_path);
    }

    // Check if the parent directory exists (mounted from host)
    struct stat st;
    if (stat("/var/lib/keti", &st) != 0 || !S_ISDIR(st.st_mode)) {
        fprintf(stderr, "[vai] WARNING: /var/lib/keti not available, shared memory disabled\n");
        vai::use_shared_memory = false;
        return false;
    }

    // Try to open existing shared state file first
    vai::shm_fd = open(vai::shm_path, O_RDWR, 0666);
    bool created = false;

    if (vai::shm_fd < 0) {
        // Create new shared state file
        vai::shm_fd = open(vai::shm_path, O_CREAT | O_RDWR, 0666);
        if (vai::shm_fd < 0) {
            fprintf(stderr, "[vai] WARNING: Failed to create shared state file: %s\n", strerror(errno));
            vai::use_shared_memory = false;
            return false;
        }

        // Set size
        if (ftruncate(vai::shm_fd, sizeof(VaiSharedState)) < 0) {
            fprintf(stderr, "[vai] WARNING: Failed to set shared state file size: %s\n", strerror(errno));
            close(vai::shm_fd);
            unlink(vai::shm_path);
            vai::shm_fd = -1;
            vai::use_shared_memory = false;
            return false;
        }
        created = true;
    }

    // Map shared memory from file
    vai::shared_state = (VaiSharedState*)mmap(nullptr, sizeof(VaiSharedState),
                                               PROT_READ | PROT_WRITE, MAP_SHARED,
                                               vai::shm_fd, 0);
    if (vai::shared_state == MAP_FAILED) {
        fprintf(stderr, "[vai] WARNING: Failed to map shared state file: %s\n", strerror(errno));
        close(vai::shm_fd);
        if (created) unlink(vai::shm_path);
        vai::shm_fd = -1;
        vai::shared_state = nullptr;
        vai::use_shared_memory = false;
        return false;
    }

    // Initialize if we created it
    if (created || vai::shared_state->magic != VAI_SHM_MAGIC) {
        // Initialize mutex for shared memory
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&vai::shared_state->shm_mutex, &attr);
        pthread_mutexattr_destroy(&attr);

        vai::shared_state->magic = VAI_SHM_MAGIC;
        vai::shared_state->process_count.store(0);
        vai::shared_state->leader_pid.store(0);
        vai::shared_state->compute_tokens.store(0);
        vai::shared_state->total_sm_limit.store(0);
        vai::shared_state->max_tokens = 100 * 1000;  // 100% * 1초 = 기본값
        vai::shared_state->total_kernels.store(0);

        fprintf(stderr, "[vai] Created new shared memory\n");
    }

    // Register this process
    pthread_mutex_lock(&vai::shared_state->shm_mutex);

    int old_count = vai::shared_state->process_count.fetch_add(1);
    vai::shared_state->total_sm_limit.fetch_add(vai::sm_limit_pct);

    // Update max_tokens based on total SM limit (capped at 100%)
    int64_t total_limit = vai::shared_state->total_sm_limit.load();
    if (total_limit > 100) total_limit = 100;  // Cap at 100%
    vai::shared_state->max_tokens = total_limit * 1000;  // tokens = limit% * 1초

    // Become leader if no leader or leader is dead
    pid_t current_leader = vai::shared_state->leader_pid.load();
    if (current_leader == 0 || kill(current_leader, 0) != 0) {
        vai::shared_state->leader_pid.store(getpid());
        vai::is_leader = true;
        // Initialize tokens if becoming new leader
        if (old_count == 0) {
            vai::shared_state->compute_tokens.store(vai::shared_state->max_tokens);
        }
    }

    // Calculate effective limit based on total demand
    // If total <= 100%, use original limit
    // If total > 100%, proportionally reduce to fit in 100%
    if (total_limit <= 100) {
        vai::effective_sm_limit_pct = vai::sm_limit_pct;
    } else {
        // Proportional share: my_limit * 100 / total
        vai::effective_sm_limit_pct = (vai::sm_limit_pct * 100) / total_limit;
        if (vai::effective_sm_limit_pct < 1) vai::effective_sm_limit_pct = 1;  // Minimum 1%
    }

    // Update local token bucket with effective limit
    vai::max_tokens = (long long)vai::effective_sm_limit_pct * 1000;
    vai::compute_tokens.store(vai::max_tokens);

    pthread_mutex_unlock(&vai::shared_state->shm_mutex);

    fprintf(stderr, "[vai] Shared memory: process_count=%d, total_sm_limit=%lld%%, my_limit=%d%%, effective=%d%%, is_leader=%s, pid=%d\n",
            vai::shared_state->process_count.load(),
            (long long)vai::shared_state->total_sm_limit.load(),
            vai::sm_limit_pct,
            vai::effective_sm_limit_pct,
            vai::is_leader ? "yes" : "no",
            getpid());

    return true;
}

// Cleanup shared memory on exit
static void cleanup_shared_memory() {
    if (!vai::shared_state) return;

    pthread_mutex_lock(&vai::shared_state->shm_mutex);

    // Unregister this process
    vai::shared_state->process_count.fetch_sub(1);
    vai::shared_state->total_sm_limit.fetch_sub(vai::sm_limit_pct);

    // Update max_tokens
    int64_t total_limit = vai::shared_state->total_sm_limit.load();
    if (total_limit > 100) total_limit = 100;
    if (total_limit < 0) total_limit = 0;
    vai::shared_state->max_tokens = total_limit * 1000;

    // If we were leader, clear it so another process can take over
    if (vai::is_leader) {
        vai::shared_state->leader_pid.store(0);
        vai::is_leader = false;
    }

    int remaining = vai::shared_state->process_count.load();

    pthread_mutex_unlock(&vai::shared_state->shm_mutex);

    // Unmap
    munmap(vai::shared_state, sizeof(VaiSharedState));
    vai::shared_state = nullptr;

    // Close
    if (vai::shm_fd >= 0) {
        close(vai::shm_fd);
        vai::shm_fd = -1;
    }

    // Remove shared state file if we're the last process
    if (remaining <= 0) {
        unlink(vai::shm_path);
        fprintf(stderr, "[vai] Removed shared state file (last process): %s\n", vai::shm_path);
    }
}

// ---------------- Token Bucket Implementation ----------------

// Token 재충전 스레드 (개별 Token Bucket + 비율 조정)
// 각 프로세스가 자신의 effective_limit 만큼 토큰을 재충전
static void token_replenish_thread() {
    int recalc_counter = 0;

    while (vai::token_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // Periodically recalculate effective limit (every 100ms)
        // In case other processes join or leave
        if (vai::use_shared_memory && vai::shared_state && ++recalc_counter >= 100) {
            recalc_counter = 0;
            int64_t total_limit = vai::shared_state->total_sm_limit.load();
            int new_effective;

            if (total_limit <= 100) {
                new_effective = vai::sm_limit_pct;
            } else {
                new_effective = (vai::sm_limit_pct * 100) / total_limit;
                if (new_effective < 1) new_effective = 1;
            }

            // Update if changed
            if (new_effective != vai::effective_sm_limit_pct) {
                vai::effective_sm_limit_pct = new_effective;
                vai::max_tokens = (long long)vai::effective_sm_limit_pct * 1000;
                fprintf(stderr, "[vai] Effective limit updated: %d%% (total=%lld%%)\n",
                        vai::effective_sm_limit_pct, (long long)total_limit);
            }
        }

        // Refill local token bucket with effective rate
        // 각 프로세스가 자신의 비율만큼 토큰 재충전
        long long refill_rate = vai::effective_sm_limit_pct;
        long long current = vai::compute_tokens.load();
        long long new_val = current + refill_rate;
        if (new_val > vai::max_tokens) new_val = vai::max_tokens;
        vai::compute_tokens.store(new_val);
    }
}

// 커널 실행 전 토큰 확인 및 대기 (개별 Token Bucket + 비율 조정)
// 각 프로세스가 자신의 effective_limit에 맞는 로컬 토큰 버킷 사용
static void rate_limiter_wait(unsigned int grid_size, unsigned int block_size) {
    // HAMi-core style: 각 커널은 토큰 소비
    long long total_threads = (long long)grid_size * block_size;
    long long kernel_cost;

    // 비용 계산: 커널 효율성 기반 차등 비용
    // cuBLAS 등 최적화 라이브러리 커널은 SM을 매우 효율적으로 사용
    // 순수 CUDA 커널(16M+ threads)은 상대적으로 비효율적
    if (total_threads < 15000000) {
        // cuBLAS 포함 대부분의 커널: 매우 높은 비용
        kernel_cost = 600;
    } else {
        // 16M+ threads 대형 커널만: 낮은 비용
        kernel_cost = 30;
    }

    int wait_count = 0;
    const int max_waits = 1000;  // 최대 1초 대기 (1ms * 1000)

    // 각 프로세스는 자신의 로컬 토큰 버킷 사용
    // effective_sm_limit_pct가 비율에 맞게 조정됨
CHECK_TOKEN:
    long long current = vai::compute_tokens.load(std::memory_order_acquire);

    if (current >= kernel_cost) {
        // 토큰 차감 시도 (CAS로 atomic)
        if (vai::compute_tokens.compare_exchange_weak(current, current - kernel_cost)) {
            return;  // 성공, 커널 실행 허용
        }
        goto CHECK_TOKEN;  // CAS 실패, 재시도
    }

    // 토큰 부족 - sleep 후 재시도
    if (wait_count < max_waits) {
        nanosleep(&vai::sleep_interval, nullptr);
        wait_count++;
        goto CHECK_TOKEN;
    }

    // 최대 대기 시간 초과 - 경고 후 실행 허용 (완전 차단 방지)
    static int timeout_warn_count = 0;
    if (timeout_warn_count < 5) {
        fprintf(stderr, "[vai] WARN: token wait timeout, allowing kernel (effective=%d%%)\n",
                vai::effective_sm_limit_pct);
        timeout_warn_count++;
    }
}

// Memory 할당 전 확인
static bool check_memory_limit(size_t requested) {
    if (vai::memory_limit == 0) return true;  // 제한 없음

    size_t current = vai::memory_used.load();
    if (current + requested > vai::memory_limit) {
        fprintf(stderr, "[vai] Memory limit exceeded: used=%zuMB + requested=%zuMB > limit=%zuMB\n",
                current / (1024*1024), requested / (1024*1024), vai::memory_limit / (1024*1024));
        return false;
    }
    return true;
}

// ---------------- Initialization ----------------
static void ensure_init() {
    if (vai::inited.load()) return;
    pthread_mutex_lock(&vai::init_mu);
    if (vai::inited.load()) { pthread_mutex_unlock(&vai::init_mu); return; }

    resolve_real();

    // CUDA 초기화 with error checking
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[vai] ERROR: cuInit failed with %d\n", err);
        pthread_mutex_unlock(&vai::init_mu);
        return;
    }

    CUdevice dev;
    err = cuDeviceGet(&dev, 0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[vai] ERROR: cuDeviceGet failed with %d\n", err);
        pthread_mutex_unlock(&vai::init_mu);
        return;
    }

    err = cuDeviceGetAttribute(&vai::total_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[vai] ERROR: cuDeviceGetAttribute failed with %d\n", err);
        vai::total_sms = 1;  // Fallback
    }

    size_t free_mem, total_mem;
    err = cuMemGetInfo(&free_mem, &total_mem);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[vai] ERROR: cuMemGetInfo failed with %d\n", err);
        total_mem = 0;
    }
    vai::total_memory = total_mem;

    // SM 제한 설정 (KETI_SM_LIMIT 환경변수, % 단위)
    vai::sm_limit_pct = 100;
    if (const char* e = getenv("KETI_SM_LIMIT")) {
        int v = atoi(e);
        if (v > 0 && v <= 100) vai::sm_limit_pct = v;
    }

    // Local Token bucket 초기화 (fallback용)
    // max_tokens = 1초 worth of tokens at current limit
    vai::max_tokens = (long long)vai::sm_limit_pct * 1000;  // ~1초 버스트 허용
    vai::compute_tokens.store(vai::max_tokens);

    // Memory 제한 설정 (KETI_MEM_LIMIT 환경변수, MB 단위)
    vai::memory_limit_mb = 0;
    if (const char* e = getenv("KETI_MEM_LIMIT")) {
        vai::memory_limit_mb = atoi(e);
        vai::memory_limit = (size_t)vai::memory_limit_mb * 1024 * 1024;
    }

    // ★ Shared Memory 초기화 (Multi-Pod 지원)
    // sm_limit_pct가 설정된 후에 호출해야 함
    bool shm_ok = init_shared_memory();

    // Token 재충전 스레드 시작
    vai::token_running.store(true);
    vai::token_thread = std::thread(token_replenish_thread);

    fprintf(stderr, "[vai] HAMi-style init: total_sms=%d, sm_limit=%d%%, max_tokens=%lld, shared_mem=%s\n",
            vai::total_sms, vai::sm_limit_pct, vai::max_tokens,
            shm_ok ? "enabled" : "disabled");
    if (vai::memory_limit > 0) {
        fprintf(stderr, "[vai] Memory limit: %dMB (total: %zuMB)\n",
                vai::memory_limit_mb, vai::total_memory / (1024*1024));
    }

    // Debug: check which functions were resolved
    fprintf(stderr, "[vai] Resolved: cudaLaunchKernel=%p, cuLaunchKernel=%p\n",
            (void*)real_cudaLaunchKernel, (void*)real_cuLaunchKernel);
    fprintf(stderr, "[vai] Resolved: cudaLaunchKernel_ptsz=%p, cuLaunchKernel_ptsz=%p\n",
            (void*)real_cudaLaunchKernel_ptsz, (void*)real_cuLaunchKernel_ptsz);

    vai::inited.store(true);
    pthread_mutex_unlock(&vai::init_mu);
}

// ---------------- CUDA Interceptors ----------------

// Internal implementation
static cudaError_t cudaLaunchKernel_impl(const void *hostFunc,
                                          dim3 gridDim, dim3 blockDim,
                                          void **args, size_t sharedMem,
                                          cudaStream_t stream)
{
    static int runtime_launch_log_count = 0;
    if (runtime_launch_log_count < 5) {
        fprintf(stderr, "[vai] cudaLaunchKernel_impl hook called! grid=%dx%dx%d block=%dx%dx%d\n",
                gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
        runtime_launch_log_count++;
    }

    resolve_real();
    ensure_init();

    // Rate limiting: 토큰 확인 및 대기
    unsigned int total_blocks = gridDim.x * gridDim.y * gridDim.z;
    unsigned int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    rate_limiter_wait(total_blocks, threads_per_block);

    vai::kernel_count.fetch_add(1);
    if (vai::use_shared_memory && vai::shared_state) {
        vai::shared_state->total_kernels.fetch_add(1);
    }

    return real_cudaLaunchKernel(hostFunc, gridDim, blockDim, args, sharedMem, stream);
}

// Export unversioned symbol
extern "C" __attribute__((visibility("default")))
cudaError_t cudaLaunchKernel(const void *hostFunc,
                              dim3 gridDim, dim3 blockDim,
                              void **args, size_t sharedMem,
                              cudaStream_t stream)
{
    return cudaLaunchKernel_impl(hostFunc, gridDim, blockDim, args, sharedMem, stream);
}

extern "C" CUresult cuLaunchKernel(CUfunction f,
                                   unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                                   unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                                   unsigned int sharedMemBytes,
                                   CUstream hStream,
                                   void **kernelParams, void **extra)
{
    static int launch_log_count = 0;
    if (launch_log_count < 5) {
        fprintf(stderr, "[vai] cuLaunchKernel hook called! grid=%dx%dx%d block=%dx%dx%d\n",
                gridX, gridY, gridZ, blockX, blockY, blockZ);
        launch_log_count++;
    }

    resolve_real();
    ensure_init();

    // Rate limiting
    unsigned int total_blocks = gridX * gridY * gridZ;
    unsigned int threads_per_block = blockX * blockY * blockZ;
    rate_limiter_wait(total_blocks, threads_per_block);

    vai::kernel_count.fetch_add(1);
    if (vai::use_shared_memory && vai::shared_state) {
        vai::shared_state->total_kernels.fetch_add(1);
    }

    return real_cuLaunchKernel(f, gridX, gridY, gridZ,
                               blockX, blockY, blockZ,
                               sharedMemBytes, hStream, kernelParams, extra);
}

// Per-thread stream variants (CUDA 12.x)
extern "C" cudaError_t cudaLaunchKernel_ptsz(const void *hostFunc,
                                              dim3 gridDim, dim3 blockDim,
                                              void **args, size_t sharedMem,
                                              cudaStream_t stream)
{
    resolve_real();
    ensure_init();

    unsigned int total_blocks = gridDim.x * gridDim.y * gridDim.z;
    unsigned int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    rate_limiter_wait(total_blocks, threads_per_block);

    vai::kernel_count.fetch_add(1);
    if (vai::use_shared_memory && vai::shared_state) {
        vai::shared_state->total_kernels.fetch_add(1);
    }

    // Use the _ptsz variant if available, otherwise fall back to regular
    if (real_cudaLaunchKernel_ptsz) {
        return real_cudaLaunchKernel_ptsz(hostFunc, gridDim, blockDim, args, sharedMem, stream);
    }
    return real_cudaLaunchKernel(hostFunc, gridDim, blockDim, args, sharedMem, stream);
}

extern "C" CUresult cuLaunchKernel_ptsz(CUfunction f,
                                        unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                                        unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                                        unsigned int sharedMemBytes,
                                        CUstream hStream,
                                        void **kernelParams, void **extra)
{
    resolve_real();
    ensure_init();

    unsigned int total_blocks = gridX * gridY * gridZ;
    unsigned int threads_per_block = blockX * blockY * blockZ;
    rate_limiter_wait(total_blocks, threads_per_block);

    vai::kernel_count.fetch_add(1);
    if (vai::use_shared_memory && vai::shared_state) {
        vai::shared_state->total_kernels.fetch_add(1);
    }

    if (real_cuLaunchKernel_ptsz) {
        return real_cuLaunchKernel_ptsz(f, gridX, gridY, gridZ,
                                        blockX, blockY, blockZ,
                                        sharedMemBytes, hStream, kernelParams, extra);
    }
    return real_cuLaunchKernel(f, gridX, gridY, gridZ,
                               blockX, blockY, blockZ,
                               sharedMemBytes, hStream, kernelParams, extra);
}

// cuLaunchKernelEx - CUDA 12.x extended launch API
typedef CUresult (*cuLaunchKernelEx_fn)(const CUlaunchConfig*, CUfunction, void**, void**);
static cuLaunchKernelEx_fn real_cuLaunchKernelEx = nullptr;

extern "C" __attribute__((visibility("default")))
CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
                          void **kernelParams, void **extra)
{
    static int launch_ex_log_count = 0;
    if (launch_ex_log_count < 10 && config) {
        fprintf(stderr, "[vai] cuLaunchKernelEx hook called! grid=%dx%dx%d block=%dx%dx%d\n",
                config->gridDimX, config->gridDimY, config->gridDimZ,
                config->blockDimX, config->blockDimY, config->blockDimZ);
        fflush(stderr);
        launch_ex_log_count++;
    }

    resolve_real();
    ensure_init();

    // Rate limiting based on config dimensions
    if (config) {
        unsigned int total_blocks = config->gridDimX * config->gridDimY * config->gridDimZ;
        unsigned int threads_per_block = config->blockDimX * config->blockDimY * config->blockDimZ;
        rate_limiter_wait(total_blocks, threads_per_block);
    }

    vai::kernel_count.fetch_add(1);
    if (vai::use_shared_memory && vai::shared_state) {
        vai::shared_state->total_kernels.fetch_add(1);
    }

    // Resolve real function if not yet done
    if (!real_cuLaunchKernelEx) {
        init_real_dlsym();
        void* libcuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (libcuda && real_dlsym) {
            real_cuLaunchKernelEx = (cuLaunchKernelEx_fn)real_dlsym(libcuda, "cuLaunchKernelEx");
        }
        if (!real_cuLaunchKernelEx && real_dlsym) {
            real_cuLaunchKernelEx = (cuLaunchKernelEx_fn)real_dlsym(RTLD_NEXT, "cuLaunchKernelEx");
        }
    }

    if (real_cuLaunchKernelEx) {
        return real_cuLaunchKernelEx(config, f, kernelParams, extra);
    }

    // Fallback to cuLaunchKernel if cuLaunchKernelEx is not available
    if (config && real_cuLaunchKernel) {
        return real_cuLaunchKernel(f,
                                   config->gridDimX, config->gridDimY, config->gridDimZ,
                                   config->blockDimX, config->blockDimY, config->blockDimZ,
                                   config->sharedMemBytes, config->hStream,
                                   kernelParams, extra);
    }

    return CUDA_ERROR_NOT_INITIALIZED;
}

// Memory allocation with limit checking
extern "C" cudaError_t cudaMalloc(void **p, size_t n) {
    static int malloc_log_count = 0;
    if (malloc_log_count < 5) {
        fprintf(stderr, "[vai] cudaMalloc hook called! size=%zu bytes (%zuMB)\n",
                n, n / (1024*1024));
        malloc_log_count++;
    }

    resolve_real();
    ensure_init();

    // Memory 제한 확인
    if (!check_memory_limit(n)) {
        fprintf(stderr, "[vai] cudaMalloc REJECTED: memory limit exceeded\n");
        return cudaErrorMemoryAllocation;
    }

    cudaError_t ret = real_cudaMalloc(p, n);
    if (ret == cudaSuccess) {
        vai::memory_used.fetch_add(n);
    }
    return ret;
}

extern "C" cudaError_t cudaFree(void *devPtr) {
    resolve_real();
    ensure_init();

    // 메모리 크기를 정확히 알 수 없으므로 대략적으로 추적
    // (실제 HAMi-core는 할당 시 크기를 map에 저장)
    cudaError_t ret = real_cudaFree(devPtr);
    // Note: 정확한 추적을 위해서는 할당 시 크기를 저장해야 함
    return ret;
}

extern "C" cudaError_t cudaMallocManaged(void **p, size_t n, unsigned int f) {
    resolve_real();
    ensure_init();

    if (!check_memory_limit(n)) {
        return cudaErrorMemoryAllocation;
    }

    cudaError_t ret = real_cudaMallocManaged(p, n, f);
    if (ret == cudaSuccess) {
        vai::memory_used.fetch_add(n);
    }
    return ret;
}

// Pass-through functions
extern "C" cudaError_t cudaMemcpy(void *d, const void *s, size_t c, cudaMemcpyKind k) {
    resolve_real();
    ensure_init();
    return real_cudaMemcpy(d, s, c, k);
}

extern "C" cudaError_t cudaMemcpyAsync(void *d, const void *s, size_t c, cudaMemcpyKind k, cudaStream_t st) {
    resolve_real();
    ensure_init();
    return real_cudaMemcpyAsync(d, s, c, k, st);
}

extern "C" cudaError_t cudaStreamCreate(cudaStream_t *st) {
    resolve_real();
    ensure_init();
    return real_cudaStreamCreate(st);
}

extern "C" cudaError_t cudaStreamDestroy(cudaStream_t st) {
    resolve_real();
    ensure_init();
    return real_cudaStreamDestroy(st);
}

extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t g, cudaStream_t st) {
    resolve_real();
    ensure_init();
    return real_cudaGraphLaunch(g, st);
}

extern "C" cudaError_t cudaDeviceSynchronize() {
    resolve_real();
    ensure_init();
    return real_cudaDeviceSynchronize();
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t st) {
    resolve_real();
    ensure_init();
    return real_cudaStreamSynchronize(st);
}

// ---------------- Query APIs ----------------
extern "C" __attribute__((visibility("default")))
int vai_get_sm_limit_pct() {
    ensure_init();
    return vai::sm_limit_pct;
}

extern "C" __attribute__((visibility("default")))
long long vai_get_kernel_count() {
    // Return shared kernel count if available
    if (vai::use_shared_memory && vai::shared_state) {
        return vai::shared_state->total_kernels.load();
    }
    return vai::kernel_count.load();
}

extern "C" __attribute__((visibility("default")))
size_t vai_get_memory_used() {
    return vai::memory_used.load();
}

extern "C" __attribute__((visibility("default")))
size_t vai_get_memory_limit() {
    return vai::memory_limit;
}

extern "C" __attribute__((visibility("default")))
long long vai_get_current_tokens() {
    return vai::compute_tokens.load();
}

// ---------------- cuInit / cuDriverGetVersion Interception ----------------
// These are called very early in CUDA initialization

extern "C" __attribute__((visibility("default")))
CUresult cuInit(unsigned int flags) {
    static int init_logged = 0;
    if (!init_logged) {
        fprintf(stderr, "[vai] cuInit hook called with flags=%u\n", flags);
        init_logged = 1;
    }
    resolve_real();
    if (real_cuInit) {
        CUresult ret = real_cuInit(flags);
        static int ret_logged = 0;
        if (!ret_logged) {
            fprintf(stderr, "[vai] real_cuInit returned %d\n", ret);
            ret_logged = 1;
        }
        return ret;
    }
    fprintf(stderr, "[vai] ERROR: real_cuInit is NULL!\n");
    return CUDA_ERROR_NOT_INITIALIZED;
}

extern "C" __attribute__((visibility("default")))
CUresult cuDriverGetVersion(int* driverVersion) {
    static int ver_logged = 0;
    if (!ver_logged) {
        fprintf(stderr, "[vai] cuDriverGetVersion hook called\n");
        ver_logged = 1;
    }
    resolve_real();
    if (real_cuDriverGetVersion) {
        return real_cuDriverGetVersion(driverVersion);
    }
    return CUDA_ERROR_NOT_INITIALIZED;
}

// ---------------- HAMi-core Style: cuGetProcAddress Interception ----------------
// This is the KEY to intercepting CUDA driver functions!
// When CUDA runtime looks up driver functions, it uses cuGetProcAddress.
// By intercepting this, we can return our hook functions.

// Note: CUDA header defines cuGetProcAddress as macro -> cuGetProcAddress_v2
// We only need to implement cuGetProcAddress_v2 which is the actual function

// The actual function that gets called (v2 is the real implementation)
extern "C" __attribute__((visibility("default")))
CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
                              CUdriverProcAddressQueryResult *symbolStatus) {
    static int entry_count = 0;
    // Log more entries to see all symbol lookups
    if (entry_count < 100) {
        fprintf(stderr, "[vai] cuGetProcAddress_v2 ENTRY: symbol=%s\n", symbol ? symbol : "(null)");
        fflush(stderr);
        entry_count++;
    }

    resolve_real();

    // First, get the real function pointer
    CUresult ret = CUDA_SUCCESS;
    if (real_cuGetProcAddress_v2) {
        ret = real_cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);
    } else if (real_cuGetProcAddress) {
        // Fallback to older version
        ret = real_cuGetProcAddress(symbol, pfn, cudaVersion, flags);
        if (symbolStatus) *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    } else {
        // No real function available - this is critical!
        // Instead of failing, try to let CUDA handle it itself by returning NOT_FOUND
        // This allows CUDA to fall back to its internal mechanisms
        static int error_count = 0;
        if (error_count < 5) {
            fprintf(stderr, "[vai] WARNING: real cuGetProcAddress not resolved for '%s'\n",
                    symbol ? symbol : "(null)");
            error_count++;
        }
        if (pfn) *pfn = nullptr;
        if (symbolStatus) *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        return CUDA_ERROR_NOT_FOUND;
    }

    // Log all function lookups to see what's being requested (only cuda/Launch related)
    if (symbol && (strstr(symbol, "Launch") || strstr(symbol, "Kernel"))) {
        fprintf(stderr, "[vai] cuGetProcAddress_v2 lookup: %s -> %p\n", symbol, pfn ? *pfn : nullptr);
    }

    if (ret == CUDA_SUCCESS && pfn && *pfn && symbol) {
        // Check if this is a function we want to intercept
        // IMPORTANT: When they ask for cuGetProcAddress, return OUR hook so that
        // all subsequent lookups go through us. This is critical because CUDA
        // caches the cuGetProcAddress function pointer and uses it for all lookups.
        if (strcmp(symbol, "cuGetProcAddress") == 0 || strcmp(symbol, "cuGetProcAddress_v2") == 0) {
            fprintf(stderr, "[vai] cuGetProcAddress intercepted: %s: returning our hook %p\n",
                    symbol, (void*)cuGetProcAddress_v2);
            fflush(stderr);
            *pfn = (void*)cuGetProcAddress_v2;
        }
        else if (strcmp(symbol, "cuLaunchKernel") == 0) {
            fprintf(stderr, "[vai] cuGetProcAddress intercepted: cuLaunchKernel: orig=%p -> hook=%p\n",
                    *pfn, (void*)cuLaunchKernel);
            fflush(stderr);
            *pfn = (void*)cuLaunchKernel;
        }
        else if (strcmp(symbol, "cuLaunchKernel_ptsz") == 0) {
            fprintf(stderr, "[vai] cuGetProcAddress intercepted: cuLaunchKernel_ptsz: orig=%p -> hook=%p\n",
                    *pfn, (void*)cuLaunchKernel_ptsz);
            fflush(stderr);
            *pfn = (void*)cuLaunchKernel_ptsz;
        }
        else if (strcmp(symbol, "cuLaunchKernelEx") == 0 || strcmp(symbol, "cuLaunchKernelEx_ptsz") == 0) {
            // Forward declaration for cuLaunchKernelEx
            extern CUresult cuLaunchKernelEx(const CUlaunchConfig*, CUfunction, void**, void**);
            fprintf(stderr, "[vai] cuGetProcAddress intercepted: %s: orig=%p -> hook=%p\n",
                    symbol, *pfn, (void*)cuLaunchKernelEx);
            fflush(stderr);
            *pfn = (void*)cuLaunchKernelEx;
        }
    }

    return ret;
}

// Also export as cuGetProcAddress for older CUDA versions
extern "C" __attribute__((visibility("default")))
CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
                           CUdriverProcAddressQueryResult *symbolStatus) {
    return cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);
}

// ---------------- CUDA Injection Entry Point ----------------
// Required for CUDA_INJECTION64_PATH mechanism
// NOTE: Do NOT call ensure_init() here! It would interfere with CUDA's own initialization.
// ensure_init() will be called lazily from our hooks when they're first invoked.
extern "C" __attribute__((visibility("default")))
int InitializeInjection(void) {
    fprintf(stderr, "[vai] CUDA Injection InitializeInjection called\n");
    // Just acknowledge the injection, don't initialize yet
    return 0;  // Success
}

// Alternative entry point name
extern "C" __attribute__((visibility("default")))
int injectionInit(void) {
    fprintf(stderr, "[vai] CUDA Injection injectionInit called\n");
    // Just acknowledge the injection, don't initialize yet
    return 0;
}

// ---------------- Constructor ----------------
__attribute__((constructor)) static void vai_ctor() {
    // Force early initialization when loaded via LD_PRELOAD
    fprintf(stderr, "[vai] Library loaded (constructor)\n");
}

// ---------------- Destructor ----------------
__attribute__((destructor)) static void vai_dtor() {
    vai::token_running.store(false);
    if (vai::token_thread.joinable()) {
        vai::token_thread.join();
    }

    // Report statistics
    long long total_kernels = vai::kernel_count.load();
    if (vai::use_shared_memory && vai::shared_state) {
        total_kernels = vai::shared_state->total_kernels.load();
    }

    fprintf(stderr, "[vai] shutdown: local_kernels=%lld, memory_used=%zuMB, was_leader=%s\n",
            vai::kernel_count.load(), vai::memory_used.load() / (1024*1024),
            vai::is_leader ? "yes" : "no");

    // ★ Shared Memory 정리
    cleanup_shared_memory();
}
