# PPO (Proximal Policy Optimization) - GPU 자원 할당 에이전트

## 1. PPO란 무엇인가

PPO는 **정책 기반 강화학습(Policy Gradient)** 알고리즘이다. OpenAI의 John Schulman이 2017년에 발표했다.

**핵심 아이디어:** 정책(Policy)을 직접 업데이트하되, 한 번에 너무 크게 바꾸지 않도록 **클리핑(Clipping)**으로 제한한다.

### 왜 PPO를 쓰는가?

| 문제 | 설명 |
|------|------|
| Policy Gradient의 불안정성 | 업데이트가 너무 크면 성능 붕괴 (catastrophic forgetting) |
| TRPO의 복잡성 | Trust Region은 계산이 비싸고 구현이 복잡함 |
| PPO의 해결책 | 간단한 클리핑으로 안정적 학습. 구현도 쉬움 |

### 강화학습 기본 구조

```
┌─────────┐   action(a)    ┌─────────────┐
│  Agent   │ ─────────────→ │ Environment │
│  (PPO)   │ ←───────────── │  (GPU Node) │
└─────────┘  state(s),      └─────────────┘
             reward(r)
```

- **Agent**: PPO 신경망. 상태를 보고 행동을 결정
- **Environment**: GPU가 있는 Edge 노드. 할당 결과에 따라 보상 반환
- **State (s)**: 현재 GPU 사용률, 요청 정보 등
- **Action (a)**: GPU 코어 %, 메모리 MB 할당량
- **Reward (r)**: 할당이 얼마나 효율적이었는지

---

## 2. PPO 수도코드 (Pseudocode)

> 출처: Schulman et al., "Proximal Policy Optimization Algorithms", 2017 (Algorithm 1)

```
Algorithm: PPO-Clip

Input: 초기 정책 파라미터 θ₀, 초기 가치함수 파라미터 φ₀

for iteration = 1, 2, ... do

    ┌──────────────────────────────────────────────────────┐
    │ Step 1. 경험 수집 (T timesteps)                       │
    │                                                       │
    │   for t = 1, 2, ..., T do                             │
    │       현재 상태 sₜ 관측                                │
    │       정책 πθ에서 행동 aₜ ~ πθ(·|sₜ) 샘플링            │
    │       환경에서 보상 rₜ, 다음 상태 sₜ₊₁ 수신            │
    │       log πθ(aₜ|sₜ) 저장  ← 나중에 ratio 계산용       │
    │       V_φ(sₜ) 저장        ← 나중에 Advantage 계산용   │
    │   end for                                             │
    └──────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────┐
    │ Step 2. Advantage 계산 (GAE)                          │
    │                                                       │
    │   for t = T, T-1, ..., 1 do  (역순)                   │
    │       δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)     ← TD Error   │
    │       Âₜ = δₜ + (γλ)·Âₜ₊₁              ← GAE        │
    │   end for                                             │
    │   Rₜ = Âₜ + V(sₜ)                      ← Return     │
    └──────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────┐
    │ Step 3. K Epochs 학습                                 │
    │                                                       │
    │   for epoch = 1, 2, ..., K do                         │
    │                                                       │
    │     # Policy (Actor) 업데이트                          │
    │     rₜ(θ) = πθ_new(aₜ|sₜ) / πθ_old(aₜ|sₜ)  ← ratio │
    │     L_CLIP = min(                                     │
    │         rₜ(θ) · Âₜ,                                  │
    │         clip(rₜ(θ), 1-ε, 1+ε) · Âₜ                  │
    │     )                                                 │
    │                                                       │
    │     # Value (Critic) 업데이트                          │
    │     L_V = (V_φ(sₜ) - Rₜ)²                            │
    │                                                       │
    │     # Entropy Bonus (탐색 장려)                        │
    │     L_E = H(πθ)                                       │
    │                                                       │
    │     # 총 Loss                                         │
    │     L = -L_CLIP + c₁·L_V - c₂·L_E                    │
    │                                                       │
    │     θ ← θ - α·∇L   (gradient descent)                │
    │   end for                                             │
    └──────────────────────────────────────────────────────┘

end for
```

---

## 3. 수도코드 단계별 설명

### Step 1. 경험 수집 — "세상과 상호작용"

```
aₜ ~ πθ(·|sₜ)   →  정책 분포에서 행동을 샘플링
```

- 현재 정책(πθ)이 확률 분포를 만들고, 거기서 행동을 **랜덤 샘플링**함
- 이것이 **탐색(Exploration)**. 항상 같은 행동을 하지 않고 다양하게 시도
- 샘플링할 때 `log_prob`과 `V(s)`를 함께 저장해야 나중에 학습 가능
- T번 반복하여 경험 배치를 모음

### Step 2. Advantage 계산 — "이 행동이 평균보다 좋았나?"

```
δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)
```

- **δₜ (TD Error)**: 실제 받은 보상 + 미래 기대값 - 예측했던 값
- 양수면: "예상보다 좋았다" → 이 행동을 더 하자
- 음수면: "예상보다 나빴다" → 이 행동을 줄이자

```
Âₜ = δₜ + (γλ)·Âₜ₊₁
```

- **GAE**: TD Error를 가중 합산하여 미래까지 고려
- λ가 0이면: 현재 스텝만 봄 (편향 높음, 분산 낮음)
- λ가 1이면: 에피소드 전체를 봄 (편향 낮음, 분산 높음)
- λ=0.95: 적당한 트레이드오프

### Step 3. K Epochs 학습 — "같은 데이터로 여러 번 학습"

```
ratio = πθ_new(a|s) / πθ_old(a|s)
```

- 새 정책과 옛 정책의 **비율**
- ratio = 1: 정책 변화 없음
- ratio > 1: 이 행동의 확률이 올라감
- ratio < 1: 이 행동의 확률이 내려감

```
L_CLIP = min(ratio · Â, clip(ratio, 1-ε, 1+ε) · Â)
```

- **클리핑의 핵심**: 정책이 한 번에 너무 많이 바뀌는 것을 방지
- ε=0.2일 때, ratio는 0.8~1.2 범위로 제한됨
- Â > 0 (좋은 행동): ratio를 키우고 싶지만 1.2까지만 허용
- Â < 0 (나쁜 행동): ratio를 줄이고 싶지만 0.8까지만 허용

---

## 4. 입력 (Input) — State 벡터

PPO 네트워크에 들어가는 **상태(State)**는 5차원 벡터이다.

```
state = [s₀, s₁, s₂, s₃, s₄]
```

| 인덱스 | 이름 | 설명 | 범위 | 데이터 출처 |
|--------|------|------|------|------------|
| s₀ | requested_cores | 요청 GPU SM 코어 비율 | 0.0 ~ 1.0 | Webhook HTTP 요청 (annotation에서 추출) |
| s₁ | requested_memory | 요청 GPU 메모리 | 0.0 ~ 1.0 | Webhook HTTP 요청 (annotation에서 추출) |
| s₂ | node_gpu_util | 현재 노드 GPU 사용률 | 0.0 ~ 1.0 | `nvidia-smi --query-gpu=utilization.gpu` |
| s₃ | node_mem_util | 현재 노드 메모리 사용률 | 0.0 ~ 1.0 | `nvidia-smi --query-gpu=utilization.memory` |
| s₄ | running_pods | 현재 실행 중인 GPU Pod 수 | 0.0 ~ 1.0 | `nvidia-smi pmon` |

모든 값은 0~1로 정규화된다.

**예시:**
```
Webhook에서 코어 80%, 메모리 4000MB 요청
현재 GPU 사용률 45%, 메모리 30%, Pod 2개 실행 중

state = [0.80, 0.12, 0.45, 0.30, 0.20]
         ↑      ↑      ↑      ↑      ↑
       80/100 4000/   45%    30%   2/10
              32768
```

---

## 5. 네트워크 구조 — Actor-Critic

PPO는 **Actor-Critic** 구조를 사용한다. 두 개의 독립적인 신경망이 있다.

### Actor (정책 네트워크)

역할: 상태를 보고 "어떤 행동을 할지" 결정

```
Input [5] → Linear(64) → ReLU → Linear(64) → ReLU → Linear(2) → Sigmoid → mean [2]
                                                                             std [2] ← 학습 가능 파라미터
```

- **출력**: 행동의 평균(μ)과 표준편차(σ)
- **Gaussian Policy**: π(a|s) = N(μ(s), σ)
- 연속 행동 공간(GPU 코어%, 메모리MB)에 적합

### Critic (가치 네트워크)

역할: 현재 상태가 "얼마나 좋은 상태인지" 평가

```
Input [5] → Linear(64) → ReLU → Linear(64) → ReLU → Linear(1) → V(s)
```

- **출력**: 상태 가치 V(s). 스칼라 하나
- "이 상태에서 미래에 받을 보상의 기대값"

### 추론 vs 학습

| 모드 | 행동 선택 | log_prob | V(s) |
|------|----------|----------|------|
| 추론 (Inference) | `action = mean` (결정론적) | 저장 안 함 | 계산 안 함 |
| 학습 (Training) | `action ~ N(mean, std)` (샘플링) | 저장 | 저장 |

---

## 6. 하이퍼파라미터

### 논문 기반 (Schulman et al., 2017)

| 파라미터 | 기호 | 값 | 역할 |
|----------|------|-----|------|
| Clip Epsilon | ε | 0.2 | 정책 변화 제한 범위. ratio를 [0.8, 1.2]로 제한 |
| Discount Factor | γ | 0.99 | 미래 보상 할인율. 1에 가까울수록 미래를 중시 |
| GAE Lambda | λ | 0.95 | Bias-Variance 트레이드오프. 1이면 Monte Carlo, 0이면 TD(0) |

### OpenAI Baselines 관행

| 파라미터 | 값 | 역할 |
|----------|-----|------|
| Learning Rate (α) | 3×10⁻⁴ | 경사하강법 스텝 크기. Adam optimizer 사용 |
| K Epochs | 4 | 같은 배치 데이터로 학습 반복 횟수 |
| Entropy Coefficient (c₂) | 0.01 | 탐색 장려 정도. 클수록 더 랜덤하게 탐색 |
| Value Loss Coefficient (c₁) | 0.5 | Critic Loss 가중치 |
| Max Grad Norm | 0.5 | Gradient Clipping 임계값. 폭발 방지 |

### 환경 고유 설정

| 파라미터 | 값 | 역할 |
|----------|-----|------|
| Batch Size (T) | 32 | 학습에 필요한 최소 경험 수 (= 수도코드의 T timesteps) |
| State Dim | 5 | 상태 벡터 차원 수 |
| Action Dim | 2 | 행동 벡터 차원 수 (코어%, 메모리MB) |
| Hidden Dim | 64 | 신경망 은닉층 뉴런 수 |

---

## 7. 학습 파이프라인 (Training Pipeline)

### 전체 흐름

```
┌──────────┐     Pod 생성       ┌───────────┐    /allocate     ┌───────────┐
│ 사용자    │ ───────────────→  │  Webhook   │ ──────────────→  │ PPO Agent │
│          │                    │ (Mutating) │ ←────────────── │ (Edge)    │
└──────────┘                    └───────────┘   cores%, memMB  └───────────┘
                                                                     │
                                     ┌───────────────────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │     PPO Agent 내부 처리           │
                    │                                  │
                    │  1. state 생성 (nvidia-smi 조회)  │
                    │  2. Actor에서 action 샘플링       │
                    │  3. log_prob, V(s) 저장          │
                    │  4. 응답 반환                     │
                    │  5. 피드백 수신 → reward 계산      │
                    │  6. Experience 완성 → 버퍼 저장   │
                    │  7. 버퍼 ≥ 32 → train_step()     │
                    └─────────────────────────────────┘
```

### 단계별 상세

**Phase 1: 행동 결정 (`/allocate` 호출)**

```
Webhook → POST /allocate { cores: 80, memory: 4000 }
                    ↓
        ┌─ nvidia-smi로 현재 GPU 상태 조회
        │   gpu_util = 0.45, mem_util = 0.30, pods = 2
        ↓
        state = [0.80, 0.12, 0.45, 0.30, 0.20]
                    ↓
        Actor Network: π(a|s) = N(μ(s), σ)
                    ↓
        action ~ N(mean, std)  ← 분포에서 샘플링 (탐색)
        log_prob = log π(a|s)  ← 저장
        value = V(s)           ← 저장
                    ↓
        action → cores=60%, memory=3000MB 변환
                    ↓
        Experience(state, action, log_prob, value) 임시 생성
        → 응답 { cores: 60, memory: 3000 }
```

**Phase 2: 피드백 수신 (`/feedback` 호출)**

```
피드백 수신 ← { actual_usage: 55%, qos_met: true, pods: 3 }
                    ↓
        reward = compute_reward(60, 3000, 55%, true, 3)
          = 효율성(0.46) + QoS(1.0) - 독점(0) - 과소(0)
          = 1.46
                    ↓
        Experience.reward = 1.46
        Experience.next_state = [새로운 nvidia-smi 상태]
        Experience.done = false
                    ↓
        record_experience() → 버퍼에 저장
```

**Phase 3: 학습 (버퍼 ≥ 32)**

```
experiences >= 32 (BATCH_SIZE)
                    ↓
        ┌── GAE 계산 ──────────────────────────┐
        │  for t = T to 1 (역순):               │
        │    δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)      │
        │    Âₜ = δₜ + γλ·Âₜ₊₁                 │
        │  Rₜ = Âₜ + V(sₜ)                     │
        └───────────────────────────────────────┘
                    ↓
        ┌── K=4 Epochs 학습 ────────────────────┐
        │  ratio = exp(log_π_new - log_π_old)   │
        │  L_CLIP = min(ratio·Â, clip·Â)        │
        │  L_V = MSE(V(s), Rₜ)                  │
        │  L = -L_CLIP + 0.5·L_V - 0.01·H(π)   │
        │  θ ← θ - 3e-4 · ∇L                   │
        └───────────────────────────────────────┘
                    ↓
        경험 버퍼 클리어 (on-policy)
        모델 저장 (ppo_model.pt)
```

---

## 8. 출력 (Output) — Action 벡터

PPO 네트워크의 출력은 **2차원 행동 벡터**이다.

```
action = [a₀, a₁]    (0~1 범위, 정규화된 값)
```

| 인덱스 | 이름 | 정규화 값 | 실제 변환 | 실제 범위 |
|--------|------|----------|----------|----------|
| a₀ | cores_percent | 0.0 ~ 1.0 | a₀ × (100-10) + 10 | 10% ~ 100% |
| a₁ | memory_mb | 0.0 ~ 1.0 | a₁ × (32768-512) + 512 | 512MB ~ 32768MB |

**변환 공식:**
```
cores_percent = action[0] × (MAX_CORES - MIN_CORES) + MIN_CORES
memory_mb     = action[1] × (MAX_MEMORY - MIN_MEMORY) + MIN_MEMORY
```

**예시:**
```
Actor 출력: action = [0.56, 0.08]

cores_percent = 0.56 × (100 - 10) + 10 = 60.4 → 60%
memory_mb     = 0.08 × (32768 - 512) + 512 = 3092 → 3092MB
```

최종적으로 요청값보다 크지 않도록 `min(할당값, 요청값)` 적용.

---

## 9. Reward 함수

PPO가 "좋은 할당"과 "나쁜 할당"을 구분하는 기준이다.

```
reward = 효율성 보상 + QoS 보상 - 독점 페널티 - 과소 할당 페널티
```

| 항목 | 수식 | 가중치 | 설명 |
|------|------|--------|------|
| 효율성 | actual_usage / allocated_cores | ×0.5 | 할당 대비 실사용률. 1에 가까울수록 좋음 |
| QoS 충족 | qos_met ? +1.0 : -0.5 | — | 워크로드 성능 목표 달성 여부 |
| 독점 방지 | allocated > fair_share×1.5 ? -0.3 : 0 | — | 다른 Pod 대비 과다 점유 시 페널티 |
| 과소 할당 | allocated < 10% ? -0.2 : 0 | — | 너무 적게 할당하면 페널티 |

---

## 10. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    Control Plane                         │
│                                                         │
│  ┌──────────────┐    ┌──────────────────┐              │
│  │  API Server   │──→│  Webhook          │              │
│  │              │    │  (MutatingAdmission)│             │
│  └──────────────┘    └────────┬─────────┘              │
│                               │ HTTP (node IP:8080)     │
└───────────────────────────────┼─────────────────────────┘
                                │
        ┌───────────────────────┼──────────────────────┐
        │                       ▼                       │
        │  ┌─────────────────────────┐                 │
        │  │  PPO Agent (DaemonSet)   │   Edge Node    │
        │  │  - Actor Network         │                 │
        │  │  - Critic Network        │                 │
        │  │  - Experience Buffer     │                 │
        │  │  - nvidia-smi 직접 접근  │                 │
        │  └─────────────────────────┘                 │
        │                                               │
        │  ┌─────────────────────────┐                 │
        │  │  GPU (NVIDIA)            │                 │
        │  │  - SM Partitioning       │                 │
        │  │  - libvai_accelerator.so │                 │
        │  └─────────────────────────┘                 │
        └───────────────────────────────────────────────┘
```

- **Webhook**: Control Plane에서 실행. Pod 생성 시 GPU 환경 주입
- **PPO Agent**: 각 Edge 노드에서 DaemonSet으로 실행. HostNetwork 사용
- **통신**: Webhook → Node IP:8080으로 PPO Agent 직접 호출

---

## 11. 현재 개발 상태

### 완료된 항목

| 항목 | 상태 | 설명 |
|------|------|------|
| Actor Network | ✅ | Gaussian Policy, 64-64 은닉층 |
| Critic Network | ✅ | V(s) 출력, 64-64 은닉층 |
| GAE 계산 | ✅ | _compute_gae() 구현 완료 |
| PPO Clipping | ✅ | train_step() K=4 epochs 구현 완료 |
| Reward 함수 | ✅ | compute_reward() 효율성/QoS/공정성 |
| 경험 수집 함수 | ✅ | select_action_for_training() |
| 하이퍼파라미터 | ✅ | 논문 + OpenAI 기준값 설정 |
| DaemonSet 배포 | ✅ | Edge 노드별 HostNetwork |
| Webhook 연동 | ✅ | Node IP 기반 호출 |

### 미연결 항목 (학습 파이프라인)

| 항목 | 상태 | 설명 |
|------|------|------|
| /allocate → select_action_for_training() | ❌ | 현재 get_allocation() (추론만) 호출 |
| /feedback → compute_reward() | ❌ | 현재 TODO (로그만 출력) |
| /feedback → record_experience() | ❌ | 미연결 |
| /feedback → train_step() | ❌ | 미연결 |

### 현재 동작

```
현재: /allocate → get_allocation() → mean 그대로 반환 (샘플링 X, 학습 X)
목표: /allocate → select_action_for_training() → 분포에서 샘플링 + log_prob/value 저장
     /feedback → compute_reward() → record_experience() → train_step()
```

**결론**: PPO 알고리즘 자체는 완성되었으나, API 서버에서 학습 파이프라인이 연결되지 않아 실제 학습이 발생하지 않는 상태. 학습 파이프라인 연결이 필요함.
