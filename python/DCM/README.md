# DCM 기반 이족 보행 제어

G1 휴머노이드 로봇의 DCM(Divergent Component of Motion) 기반 3-Layer 보행 제어 구현체입니다.

## 📐 전체 구조

DCM 제어는 3개의 계층적 레이어로 구성됩니다:

```
Layer 1: Trajectory Planner (TrajectoryOptimization)
    ↓ ref_dcm, ref_dcm_vel, ref_com_pos, ref_com_vel, foot_traj
Layer 2: Simplified Model Control (SimplifiedModelControl) 
    ↓ desired_com_vel, desired_zmp
Layer 3: Whole-Body Controller (WholeBodyController)
    ↓ qpos, qvel
MuJoCo Simulation
```

### 레퍼런스 논문
- "A Benchmarking of DCM Based Architectures for Position and Velocity Controlled Walking of Humanoid Robots"
- DCM 정의: ξ = x + (1/ω)·dx (ω = √(g/z_c))

---

## 🎯 Layer 1: Trajectory Planner ([Layer1.py](Layer1.py))

### 역할
발자국 계획(footsteps)을 기반으로 DCM, CoM, 발 궤적을 생성합니다.

### 주요 메서드

#### 1. `plan_footsteps()`
```python
footsteps = [(x₀, y₀), (x₁, y₁), ..., (xₙ, yₙ)]
```
- 왼발부터 시작하여 교대로 발자국 배치
- 첫 발(i=0): x=init_xy[0], y=step_width
- 이후 발: x=init_xy[0] + i·step_length, y=±step_width (좌우 교대)

#### 2. `compute_dcm_trajectory()`
DCM End-of-Step(EOS) 기반 역방향 계산:
```python
# 역방향: 마지막 스텝부터 시작
dcm_eos[-1] = footsteps[-1]
for i in reversed:
    dcm_eos[i] = next_zmp + (dcm_eos[i+1] - next_zmp)·exp(-ω·T)

# 순방향: 각 스텝 내 궤적 생성
ξ(t) = r + (ξ_eos - r)·exp(-ω·t_remaining)
dξ(t) = ω·(ξ(t) - r)
```

#### 3. `compute_com_trajectory()`
DCM을 적분하여 CoM 궤적 생성:
```python
dx = ω·(ξ_ref - x)
x[k+1] = x[k] + dx·dt
```

#### 4. `compute_foot_trajectories()`
DSP/SSP에 따른 발 궤적:
- **DSP (Double Support Phase)**: 양발 고정
- **SSP (Single Support Phase)**: 스윙 발이 정현파 궤적으로 이동
  ```python
  progress = 0.5·(1 - cos(π·swing_phase))
  z = ground_z + step_height·sin(π·swing_phase)
  ```

---

## ⚙️ Layer 2: Simplified Model Control ([Layer2.py](Layer2.py))

### 역할
3D 동역학을 2D LIPM(Linear Inverted Pendulum Model)로 단순화하여 제어합니다.

### 제어 흐름

#### 1. DCM 계산
```python
current_dcm = x + dx/ω  # 측정값으로부터 계산
```

#### 2. DCM Instantaneous Control (Eq. 7)
목표 ZMP 계산:
```python
r_ref = ξ_ref - (1/ω)·dξ_ref + Kp·e_dcm + Ki·∫e_dcm·dt
```
- `Kp`: DCM 비례 게인 (> 1.0)
- `Ki`: DCM 적분 게인 (≥ 0.0)
- 적분항에 anti-windup 적용 (±0.05m 제한)

#### 3. ZMP-CoM Controller (Eq. 13)
목표 CoM 속도 계산:
```python
dx* = dx_ref - K_zmp·(r_ref - r) + K_com·(x_ref - x)
```
- `K_zmp`: ZMP 오차 게인 (0 < K_zmp < ω)
- `K_com`: CoM 위치 게인 (K_com > ω)

---

## 🤖 Layer 3: Whole-Body Controller ([Layer3.py](Layer3.py))

### 역할
Task Space 목표(CoM, Foot)를 Joint Space 속도(qvel)로 변환합니다.

### Task 구성 (Mink 기반 IK)

| Task | Cost | 설명 |
|------|------|------|
| **CoM Task** | 100.0 | CoM 위치 추종 |
| **Foot Position** | 200.0 | 발 위치 추종 (높은 우선순위) |
| **Foot Orientation** | 5.0 | 발 방향 유지 |
| **Torso Orientation** | 5.0 | 상체 수직 유지 |
| **Arm Posture** | 5.0 | 팔 스윙 |
| **Pelvis/Posture** | 0.0 | 기본 자세 (soft) |

### IK 풀이
```python
vel = mink.solve_ik(configuration, tasks, dt, solver="daqp", damping=1e-1)
configuration.integrate_inplace(vel, dt)
```
- QP 기반 soft task 풀이
- Configuration Limit + Collision Avoidance 적용

---

## 📊 DSP_TIME 설정 방법

### DSP (Double Support Phase) 시간 구성

```python
# config.py
STEP_TIME = 0.7      # 한 스텝 총 시간
DSP_TIME = 0.08      # 기본 DSP 시간 (양발 지지)
INIT_DSP_EXTRA = 0.12  # 첫 스텝 추가 DSP 시간
```

### 첫 스텝 DSP 확장 로직 ([Layer1.py](Layer1.py#L37))

```python
def _dsp_time_for(self, i: int) -> float:
    """i번째 스텝의 DSP 시간"""
    if i == 0 and self.init_dsp_extra > 0:
        return self.dsp_time + self.init_dsp_extra  # 0.08 + 0.12 = 0.20
    return self.dsp_time  # 0.08
```

**목적**: 정지 상태에서 안정적으로 보행을 시작하기 위해 첫 스텝의 양발 지지 시간을 연장합니다.

### 스텝 타이밍 구조

| 스텝 | DSP 시간 | SSP 시간 | 총 시간 |
|------|---------|----------|---------|
| **0번 (첫 스텝)** | 0.20s | 0.50s | 0.70 + 0.12 = **0.82s** |
| **1번 이후** | 0.08s | 0.62s | **0.70s** |

### Support Phase 판별 ([g1_wbc_dynamics_mink.py](g1_wbc_dynamics_mink.py#L33))

```python
def get_support_phase(traj_idx: int, samples_per_step: int) -> str:
    first_step_samples = samples_per_step + int(INIT_DSP_EXTRA / dt)
    if traj_idx < first_step_samples:
        step_idx = 0
        local_t = traj_idx * dt
        first_dsp = DSP_TIME + INIT_DSP_EXTRA  # 0.20s
    else:
        # 이후 스텝
        local_t = ...
        first_dsp = DSP_TIME  # 0.08s
    
    if local_t < first_dsp:
        return 'dsp'  # 양발 지지
    elif step_idx % 2 == 0:
        return 'left_support'  # 왼발 지지, 오른발 스윙
    else:
        return 'right_support'  # 오른발 지지, 왼발 스윙
```

### DSP/SSP 시간 튜닝 가이드

1. **DSP_TIME (0.08s)**
   - 너무 짧으면: 발 전환 시 불안정
   - 너무 길면: 보행 속도 저하, 로봇이 "껑충껑충" 뛰는 느낌
   - 권장 범위: 0.05 ~ 0.15s

2. **INIT_DSP_EXTRA (0.12s)**
   - 정지에서 출발 시 안정성 확보
   - CoM이 지지 영역으로 이동할 시간 제공
   - 권장 범위: 0.10 ~ 0.20s

3. **STEP_TIME (0.7s)**
   - 전체 보행 속도 결정
   - DSP_TIME + SSP_TIME = STEP_TIME
   - 빠른 보행: 0.5~0.6s / 안정적 보행: 0.7~0.8s

---

## 🚀 실행

```bash
cd /home/frlab/mujoco_demo/tutorial/g1_new/DCM
python g1_wbc_dynamics_mink.py
```

### 주요 파라미터 ([config.py](config.py))

```python
N_STEPS = 20           # 총 스텝 수
STEP_LENGTH = 0.1      # 보폭 (m)
STEP_WIDTH = 0.1185    # 발 간격 (m)
STEP_HEIGHT = 0.08     # 발 들어올림 높이 (m)
K_DCM = 2.0            # DCM 비례 게인
K_ZMP = 1.0            # ZMP 피드백 게인
K_COM = 1.0            # CoM 위치 게인
ARM_SWING_AMP = 0.15   # 팔 스윙 진폭 (rad)
```

---

## 📂 파일 구조

```
DCM/
├── config.py                    # 파라미터 설정
├── Layer1.py                    # 궤적 계획 (DCM, CoM, Foot)
├── Layer2.py                    # 간략화 모델 제어 (ZMP-CoM)
├── Layer3.py                    # 전신 제어 (IK)
├── g1_wbc_dynamics_mink.py      # 메인 시뮬레이션 루프
├── zmp_sensor.py                # ZMP 센서 계산
└── README.md                    # 이 문서
```

---

## 🔍 주요 특징

1. **첫 스텝 안정화**: `INIT_DSP_EXTRA`로 DSP 시간 확장
2. **Footstep vs Foot Target 분리**: 
   - `footsteps`: CoM/ZMP 계획용 (CoM 기준)
   - `foot_targets`: IK 목표용 (발 기준)
3. **팔 스윙**: 보행과 동기화된 정현파 (대립 위상)
4. **Support Phase 자동 판별**: DSP → Left Support → DSP → Right Support
5. **Anti-windup**: DCM 적분 오차 제한으로 안정성 확보

---

## � QP Solver 비교: qpax vs ReLU-QP

DCM 제어는 QP를 풀 필요가 없지만, [MPC+QP](../MPC+QP/)와 [Diff_MPC_Learning](../Diff_MPC_Learning/)에서는 미분 가능한 QP solver가 필요합니다.

### qpax (사용 중)

**개요**: JAX 기반 미분 가능 QP solver ([GitHub](https://github.com/kevin-tracy/qpax))

**핵심 특징**:
- **알고리즘**: Primal-Dual Interior Point Method (PDIP)
- **미분 방법**: **Implicit Function Theorem** + Custom VJP
  ```python
  @jax.custom_vjp
  def solve_qp_primal(Q, q, A, b, G, h):
      # Forward: PDIP로 QP 풀이
      # Backward: Implicit differentiation (KKT 조건 활용)
  ```
- **그래디언트 계산**: KKT 조건을 만족하는 해에서 implicit differentiation
  ```python
  # Backward pass: diff_qp() 함수 내부
  dl_dQ = 0.5·(dz⊗z + z⊗dz)  # OptNet 스타일
  dl_dq = dz
  dl_dh = -λ·dλ
  ```
- **Smoothing**: `target_kappa` 파라미터로 relaxed KKT → 안정적 그래디언트
- **정밀도**: Float64 권장 (tol ∈ [1e-12, 1e-2])
- **플랫폼**: CPU/GPU (JAX 기반)

**장점**:
- ✅ JAX 생태계와 완벽한 통합 (`jit`, `vmap`, `grad`)
- ✅ 수학적으로 엄밀한 implicit differentiation
- ✅ 그래디언트 smoothing으로 학습 안정성 확보
- ✅ 중소규모 문제에서 높은 정확도

**단점**:
- ⚠️ 대규모 문제에서 속도 제한
- ⚠️ GPU 가속이 ReLU-QP 대비 덜 최적화됨

### ReLU-QP (대안)

**개요**: GPU 가속 QP solver ([GitHub](https://github.com/RoboticExplorationLab/ReLUQP-py), [Paper](https://arxiv.org/abs/2311.18056))

**핵심 특징**:
- **알고리즘**: ADMM을 ReLU 신경망으로 재구성
  ```python
  # ADMM iteration → Neural Network Layer
  class ReLU_Layer(torch.nn.Module):
      def forward(self, input, idx):
          out = W @ input + b
          out[nx:nx+nc].clamp_(l, u)  # ReLU-like clamp
          return out
  ```
- **미분 방법**: PyTorch Autograd (unrolled differentiation)
  - ADMM의 모든 iteration을 계산 그래프로 unroll
  - 각 레이어를 통한 역전파로 그래디언트 계산
- **Adaptive ρ**: 수렴 속도 향상을 위한 동적 penalty 조정
- **정밀도**: Float32/Float64 모두 지원
- **플랫폼**: GPU 최적화 (PyTorch 기반)

**장점**:
- ✅ **대규모 문제에서 매우 빠름** (order-of-magnitude 속도 향상)
- ✅ GPU 병렬화 극대화 (batch QP 풀이)
- ✅ 실시간 MPC에 최적화됨
- ✅ PyTorch와 자연스러운 통합

**단점**:
- ⚠️ Unrolled differentiation → 메모리 사용량 높음
- ⚠️ ADMM 특성상 높은 정밀도 필요 시 iteration 증가
- ⚠️ **Implicit differentiation 미지원** (명시적 unrolling만)

### 비교표

| 특성 | qpax | ReLU-QP |
|------|------|---------|
| **프레임워크** | JAX | PyTorch |
| **알고리즘** | PDIP | ADMM → ReLU Net |
| **미분 방식** | Implicit (KKT) | Unrolled Autograd |
| **메모리** | 낮음 | 높음 (unrolling) |
| **소규모 QP** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **대규모 QP** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **정확도** | 매우 높음 | 높음 |
| **실시간 MPC** | 적합 | 매우 적합 |
| **학습 안정성** | ⭐⭐⭐⭐⭐ (smoothing) | ⭐⭐⭐⭐ |

### 왜 qpax를 사용하나?

1. **JAX 생태계**: 프로젝트 전체가 JAX 기반 (MJX 시뮬레이션 등)
2. **수학적 엄밀성**: Implicit differentiation이 이론적으로 명확
3. **학습 안정성**: `target_kappa`로 gradient smoothing 가능
4. **문제 크기**: G1 보행의 MPC horizon은 중간 규모 (수백 변수)
5. **코드 단순성**: `jax.grad()`만으로 end-to-end 미분 가능

### 추천 사항

- **중소규모 MPC (horizon ≤ 50)**: qpax 추천
- **대규모 MPC (horizon > 100)**: ReLU-QP 고려
- **실시간 요구 (< 1ms)**: ReLU-QP + GPU
- **학습 우선**: qpax (gradient smoothing)

---

## 📖 참고

- [MPC+QP](../MPC+QP/): MPC 기반 비교 구현
- [ZMP](../ZMP/): ZMP 기반 비교 구현
- [Mink 문서](https://github.com/stephane-caron/mink)
- [qpax GitHub](https://github.com/kevin-tracy/qpax)
- [ReLU-QP Paper](https://arxiv.org/abs/2311.18056)
