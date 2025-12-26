#!/bin/bash
#
# MPS Static Partition 초기화 스크립트
# 서버 리부팅 후 실행 필요 (MPS daemon + 파티션은 리부팅 시 초기화됨)
#
# 사용법:
#   ./setup_mps_partitions.sh              # edge-gpu-232 에서 직접 실행
#   ./setup_mps_partitions.sh remote       # control-plane에서 SSH로 원격 실행
#   ./setup_mps_partitions.sh --remote     # (동일)
#
# 수행 내용:
#   1. MPS daemon 정리 (기존 프로세스 종료)
#   2. GPU 아키텍처 감지 (Blackwell → Static Partition, 그 외 → Thread Percentage)
#   3. MPS daemon 시작
#   4. SM 파티션 생성 또는 Thread Percentage 설정
#   5. PPO Agent 재시작 (파티션 매핑 재로드)
#

set -e

# ============================================================
# 설정
# ============================================================
EDGE_NODE="10.0.4.232"
EDGE_USER="root"
EDGE_PASS="ketilinux"
MPS_PIPE_DIR="/tmp/nvidia-mps"
MPS_LOG_DIR="/tmp/nvidia-mps-log"

# 파티션 설정: chunk 수 (RTX PRO 6000 Blackwell = 22 chunks, 8 SM/chunk, 176 SM total)
PARTITION_A_CHUNKS=7   # 56 SM
PARTITION_B_CHUNKS=7   # 56 SM
PARTITION_C_CHUNKS=8   # 64 SM

# 색상
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================
# 함수
# ============================================================
log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${CYAN}[STEP]${NC} $1"; }

run_on_edge() {
    # 원격 모드면 SSH, 로컬이면 직접 실행
    if [ "$REMOTE_MODE" = true ]; then
        sshpass -p "$EDGE_PASS" ssh -o StrictHostKeyChecking=no "$EDGE_USER@$EDGE_NODE" "$1"
    else
        bash -c "$1"
    fi
}

# ============================================================
# 메인
# ============================================================
REMOTE_MODE=false
if [ "$1" = "--remote" ] || [ "$1" = "remote" ]; then
    REMOTE_MODE=true
    log_info "원격 모드: SSH를 통해 $EDGE_NODE 에 접속합니다"
    # sshpass 확인
    if ! command -v sshpass &> /dev/null; then
        log_error "sshpass가 설치되어 있지 않습니다. apt install sshpass"
        exit 1
    fi
    # SSH 연결 테스트
    if ! sshpass -p "$EDGE_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$EDGE_USER@$EDGE_NODE" "echo ok" &>/dev/null; then
        log_error "SSH 연결 실패: $EDGE_USER@$EDGE_NODE"
        exit 1
    fi
    log_info "SSH 연결 확인 완료"
fi

echo ""
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  MPS Partition 초기화                    ${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""

# --- Step 1: GPU 확인 및 아키텍처 감지 ---
log_step "Step 1: GPU 상태 및 아키텍처 확인"
GPU_INFO=$(run_on_edge "nvidia-smi --query-gpu=name,driver_version,uuid --format=csv,noheader 2>/dev/null | head -1" || true)
if [ -z "$GPU_INFO" ]; then
    log_error "GPU를 찾을 수 없습니다. 드라이버를 확인하세요."
    exit 1
fi
log_info "GPU: $GPU_INFO"

# GPU UUID 자동 감지
GPU_UUID=$(run_on_edge "nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader 2>/dev/null | head -1" || true)
GPU_UUID=$(echo "$GPU_UUID" | tr -d '[:space:]')
log_info "GPU UUID: $GPU_UUID"

# GPU 이름으로 아키텍처 감지
GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
STATIC_PARTITION_SUPPORTED=false

# Blackwell 계열: RTX PRO 6000, RTX 5090, B100, B200 등
if echo "$GPU_NAME" | grep -qiE "PRO 6000|B100|B200|B40|5090|5080|Blackwell"; then
    STATIC_PARTITION_SUPPORTED=true
    log_info "GPU 아키텍처: Blackwell → Static Partition 모드 사용"
else
    log_info "GPU 아키텍처: $(echo "$GPU_NAME") → Thread Percentage 모드 사용"
    log_warn "Static Partition(-S)은 Blackwell 이상에서만 지원됩니다"
fi
echo ""

# --- Step 2: 기존 MPS 정리 ---
log_step "Step 2: 기존 MPS daemon 정리"
run_on_edge "
    export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE_DIR
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 0.5
    pkill -9 -f nvidia-cuda-mps 2>/dev/null || true
    sleep 1
" 2>/dev/null || true

# 종료 확인
REMAINING=$(run_on_edge "pgrep -c -f nvidia-cuda-mps 2>/dev/null" || echo "0")
if [ "$REMAINING" != "0" ] && [ -n "$REMAINING" ]; then
    log_warn "MPS 프로세스 $REMAINING 개 남아있음. 강제 종료 재시도..."
    run_on_edge "pkill -9 -f nvidia-cuda-mps 2>/dev/null; sleep 1" 2>/dev/null || true
fi
log_info "기존 MPS 프로세스 정리 완료"
echo ""

# --- Step 2.5: Compute Mode 설정 ---
log_step "Step 2.5: GPU Compute Mode 설정"
# 먼저 DEFAULT로 리셋 후 EXCLUSIVE_PROCESS로 설정 (안전하게)
run_on_edge "nvidia-smi -i 0 -c DEFAULT 2>/dev/null" || true
sleep 0.5
run_on_edge "nvidia-smi -i 0 -c EXCLUSIVE_PROCESS" 2>&1
echo ""

# --- Step 3: MPS daemon 시작 ---
if [ "$STATIC_PARTITION_SUPPORTED" = true ]; then
    log_step "Step 3: MPS daemon 시작 (Static Partition 모드: -S)"
    run_on_edge "
        export CUDA_VISIBLE_DEVICES=0
        export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE_DIR
        export CUDA_MPS_LOG_DIRECTORY=$MPS_LOG_DIR
        rm -rf $MPS_PIPE_DIR $MPS_LOG_DIR
        mkdir -p $MPS_PIPE_DIR $MPS_LOG_DIR
        nvidia-cuda-mps-control -d -S
        sleep 1
    " 2>&1
else
    log_step "Step 3: MPS daemon 시작 (Standard 모드)"
    run_on_edge "
        export CUDA_VISIBLE_DEVICES=0
        export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE_DIR
        export CUDA_MPS_LOG_DIRECTORY=$MPS_LOG_DIR
        rm -rf $MPS_PIPE_DIR $MPS_LOG_DIR
        mkdir -p $MPS_PIPE_DIR $MPS_LOG_DIR
        nvidia-cuda-mps-control -d
        sleep 1
    " 2>&1
fi

# MPS daemon 시작 확인
MPS_PID=$(run_on_edge "pgrep -f 'nvidia-cuda-mps-control -d' 2>/dev/null | head -1" || true)
if [ -z "$MPS_PID" ]; then
    log_error "MPS daemon 시작 실패!"
    log_error "로그 확인: cat $MPS_LOG_DIR/control.log"
    run_on_edge "cat $MPS_LOG_DIR/control.log 2>/dev/null | tail -10" || true
    exit 1
fi
log_info "MPS daemon 시작됨 (PID: $MPS_PID)"

# MPS server 시작 대기
log_info "MPS server 시작 대기 중..."
for i in $(seq 1 5); do
    SERVER_PID=$(run_on_edge "pgrep -f nvidia-cuda-mps-server 2>/dev/null | head -1" || true)
    if [ -n "$SERVER_PID" ]; then
        log_info "MPS server 시작됨 (PID: $SERVER_PID)"
        break
    fi
    sleep 1
done
if [ -z "$SERVER_PID" ]; then
    log_warn "MPS server가 아직 시작되지 않았습니다 (파티션 생성 후 시작될 수 있음)"
fi
echo ""

# --- Step 4: 파티션/Thread Percentage 설정 ---
if [ "$STATIC_PARTITION_SUPPORTED" = true ]; then
    log_step "Step 4: SM Static Partition 생성"
    log_info "  Partition A: ${PARTITION_A_CHUNKS} chunks ($((PARTITION_A_CHUNKS * 8)) SM)"
    log_info "  Partition B: ${PARTITION_B_CHUNKS} chunks ($((PARTITION_B_CHUNKS * 8)) SM)"
    log_info "  Partition C: ${PARTITION_C_CHUNKS} chunks ($((PARTITION_C_CHUNKS * 8)) SM)"
    echo ""

    PART_A=$(run_on_edge "export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE_DIR && echo 'sm_partition add $GPU_UUID $PARTITION_A_CHUNKS' | nvidia-cuda-mps-control 2>&1")
    log_info "Partition A 생성: $PART_A"

    PART_B=$(run_on_edge "export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE_DIR && echo 'sm_partition add $GPU_UUID $PARTITION_B_CHUNKS' | nvidia-cuda-mps-control 2>&1")
    log_info "Partition B 생성: $PART_B"

    PART_C=$(run_on_edge "export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE_DIR && echo 'sm_partition add $GPU_UUID $PARTITION_C_CHUNKS' | nvidia-cuda-mps-control 2>&1")
    log_info "Partition C 생성: $PART_C"

    echo ""
    log_info "파티션 목록:"
    run_on_edge "export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE_DIR && echo lspart | nvidia-cuda-mps-control" 2>&1

    # 파티션 생성 후 MPS server 시작 확인
    log_info "파티션 생성 후 MPS server 확인 중..."
    sleep 1
    SERVER_PID=$(run_on_edge "pgrep -f nvidia-cuda-mps-server 2>/dev/null | head -1" || true)
    if [ -n "$SERVER_PID" ]; then
        log_info "MPS server 정상 동작 중 (PID: $SERVER_PID)"
    else
        log_warn "MPS server가 아직 없습니다. 클라이언트 연결 시 자동 시작됩니다."
    fi
else
    log_step "Step 4: Thread Percentage 설정 (Standard MPS)"
    log_warn "Static Partition 미지원 GPU → Thread Percentage로 SM 제한"

    # MPS server 시작 대기
    for i in $(seq 1 5); do
        SERVER_PID=$(run_on_edge "pgrep -f nvidia-cuda-mps-server 2>/dev/null | head -1" || true)
        if [ -n "$SERVER_PID" ]; then
            break
        fi
        sleep 1
    done

    if [ -n "$SERVER_PID" ]; then
        # Default thread percentage 설정 (전체 SM의 비율)
        DEFAULT_PCT=100
        run_on_edge "export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE_DIR && echo 'set_default_active_thread_percentage $DEFAULT_PCT' | nvidia-cuda-mps-control" 2>&1
        log_info "Default active thread percentage: ${DEFAULT_PCT}%"
        log_info "개별 Pod의 SM 제한은 libvai_accelerator.so (Token Bucket)으로 수행됩니다"
    else
        log_warn "MPS server 미시작. GPU 사용 시 자동 시작됩니다."
    fi
fi
echo ""

# --- Step 5: PPO Agent 재시작 ---
log_step "Step 5: PPO Agent 재시작 (파티션 매핑 재로드)"
if [ "$REMOTE_MODE" = true ]; then
    kubectl rollout restart daemonset/keti-ppo-agent -n edge-system 2>/dev/null || true
    kubectl rollout status daemonset/keti-ppo-agent -n edge-system --timeout=120s 2>/dev/null || true
else
    log_warn "Edge 노드에서 실행 중: PPO Agent를 수동으로 재시작하세요"
    log_warn "  kubectl rollout restart daemonset/keti-ppo-agent -n edge-system"
fi
echo ""

# --- Step 6: 검증 ---
log_step "Step 6: 검증"
sleep 2

# GPU 상태 확인
log_info "GPU Compute Mode:"
run_on_edge "nvidia-smi --query-gpu=compute_mode --format=csv,noheader 2>/dev/null | head -1" 2>&1

# MPS 프로세스 확인
log_info "MPS 프로세스:"
run_on_edge "ps aux | grep nvidia-cuda-mps | grep -v grep" 2>&1 || log_warn "MPS 프로세스 없음"

# PPO Agent 파티션 확인
if [ "$STATIC_PARTITION_SUPPORTED" = true ]; then
    PARTITION_CHECK=$(curl -s http://$EDGE_NODE:8080/partitions 2>/dev/null || true)
    if echo "$PARTITION_CHECK" | python3 -m json.tool 2>/dev/null | grep -q "count"; then
        log_info "PPO Agent 파티션 로드 확인:"
        echo "$PARTITION_CHECK" | python3 -m json.tool 2>/dev/null
    else
        log_warn "PPO Agent 파티션 로드 대기 중... 잠시 후 다시 확인하세요:"
        log_warn "  curl http://$EDGE_NODE:8080/partitions"
    fi
fi

# CUDA 접근 테스트
log_info "CUDA 접근 테스트:"
run_on_edge "
    export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE_DIR
    python3 -c '
import subprocess
r = subprocess.run([\"nvidia-smi\", \"--query-gpu=name,utilization.gpu\", \"--format=csv,noheader\"],
                   capture_output=True, text=True, timeout=5)
print(\"  GPU:\", r.stdout.strip())
' 2>/dev/null
" 2>&1 || true

echo ""
echo -e "${GREEN}==========================================${NC}"
if [ "$STATIC_PARTITION_SUPPORTED" = true ]; then
    echo -e "${GREEN}  MPS Static Partition 초기화 완료!       ${NC}"
    echo -e "${GREEN}==========================================${NC}"
    echo ""
    echo "파티션 구성:"
    echo "  A: ${PARTITION_A_CHUNKS} chunks ($(( PARTITION_A_CHUNKS * 8 )) SM) - 공간분할 파티션"
    echo "  B: ${PARTITION_B_CHUNKS} chunks ($(( PARTITION_B_CHUNKS * 8 )) SM) - 공간분할 파티션"
    echo "  C: ${PARTITION_C_CHUNKS} chunks ($(( PARTITION_C_CHUNKS * 8 )) SM) - 공간분할 파티션"
    echo "  합계: $(( PARTITION_A_CHUNKS + PARTITION_B_CHUNKS + PARTITION_C_CHUNKS )) / 22 chunks"
else
    echo -e "${GREEN}  MPS Standard 모드 초기화 완료!          ${NC}"
    echo -e "${GREEN}==========================================${NC}"
    echo ""
    echo "모드: Thread Percentage (SM 제한은 libvai_accelerator.so로 수행)"
fi
echo ""
echo "테스트:"
echo "  kubectl apply -f scripts/test/workloads/workload-a-1.yaml"
echo "  kubectl describe pod workload-a-1 | grep CUDA_MPS"
