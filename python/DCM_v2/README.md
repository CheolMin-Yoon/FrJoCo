# DCM_v2 로직 설명

DCM(Divergent Component of Motion) 기반 휴머노이드 보행 제어 구현체.
두 가지 실행 모드를 제공한다:

- `g1_kinematic.py` — mink IK 기반 키네마틱 제어
- `g1_wbc_dynamics_qp.py` — QP 기반 역동역학 토크 제어

---

## 전체 구조

```
Layer1 (오프라인 궤적 생성)
  ↓ footsteps, DCM, CoM, 발 궤적
Layer2 (DCM PI 피드백)
  ↓ desired CoM velocity, desired ZMP
Layer3 (WBC — IK 또는 QP)
  ↓ 관절각 또는 토크
로봇
```

---

## Layer1: TrajectoryOptimization (`Layer1.py`)

시뮬레이션 시작 전에 전체 보행 궤적을 오프라인으로 생성한다.

### 1. Footstep Plan
- 초기 CoM 위치 기준으로 N_STEPS개의 발자국 좌표 생성
- 짝수 인덱스 = 왼발 (y = +STEP_WIDTH), 홀수 = 오른발 (y = -STEP_WIDTH)
- x 방향으로 STEP_LENGTH씩 전진

### 2. DCM 궤적 (역방향 → 순방향)
- 마지막 발자국에서 시작하여 역방향으로 각 스텝 끝의 DCM(dcm_eos) 계산:
  ```
  dcm_eos[i] = zmp[i+1] + (dcm_eos[i+1] - zmp[i+1]) * exp(-ω * T_step)
  ```
- 순방향으로 각 타임스텝의 DCM 보간:
  ```
  ξ(t) = zmp + (ξ_end - zmp) * exp(-ω * t_remaining)
  dξ(t) = ω * (ξ(t) - zmp)
  ```
- ω = sqrt(g / z_c) — LIPM 고유 주파수

### 3. CoM 궤적 (DCM 적분)
- DCM 정의 ξ = x + dx/ω 를 역으로 풀어 CoM 속도 계산:
  ```
  dx = ω * (ξ - x)
  x(k+1) = x(k) + dx * dt
  ```

### 4. 발 궤적
- DSP 구간: 양발 고정
- SSP 구간: 스윙 발을 다음 발자국으로 이동
  - XY: 코사인 보간 `progress = 0.5 * (1 - cos(π * phase))`
  - Z: `sin(π * phase) * STEP_HEIGHT`

### 5. CoM Shift (선택)
- 보행 시작 전 CoM을 첫 지지발(왼발) 쪽으로 코사인 보간으로 이동
- COM_SHIFT_TIME > 0일 때만 활성화, 궤적 앞에 prepend

---

## Layer2: SimplifiedModelControl (`Layer2.py`)

실시간 DCM PI 피드백 컨트롤러. 매 타임스텝 실행.

### 1. 현재 DCM 계산
```
ξ_meas = x_meas + dx_meas / ω
```

### 2. Desired ZMP (Eq. 7)
```
r_des = ξ_ref - dξ_ref/ω + Kp*(ξ_meas - ξ_ref) + Ki*∫(ξ_meas - ξ_ref)dt
```
- Kp (K_DCM): DCM 비례 게인 — DCM 오차를 ZMP로 보상
- Ki (KI_DCM): DCM 적분 게인 — 정상상태 오차 제거 (anti-windup 포함)

### 3. Desired CoM Velocity (Eq. 13)
```
dx_des = dx_ref - K_zmp*(r_des - r_meas) + K_com*(x_ref - x_meas)
```
- K_zmp: ZMP 오차 피드백 — 실제 ZMP가 목표와 다를 때 CoM 속도 보정
- K_com: CoM 위치 오차 피드백 — CoM이 기준 궤적에서 벗어날 때 보정

### ZMP 측정 (`zmp_sensor.py`)
- MuJoCo contact 배열에서 발 body에 해당하는 접촉점만 필터링
- 수직력(fn) 가중 평균으로 ZMP 계산: `zmp = Σ(pos * fn) / Σ(fn)`

---

## g1_kinematic.py — 키네마틱 모드

### 제어 흐름 (매 타임스텝)

```
1. 초기 안정화 (0~2초)
   → gravity comp + PD (초기 자세 유지)

2. 보행 시작 후
   a. Layer2: DCM PI 피드백
      - 측정: CoM pos/vel, ZMP (접촉력 기반)
      - 출력: desired CoM velocity
   
   b. 목표 설정
      - target_com = com_traj[k] + desired_com_vel * dt
      - target_lf/rf = lf_traj[k], rf_traj[k]
      - ref_torso = (target_com_x, target_com_y, TORSO_HEIGHT)
   
   c. mink IK (Layer3 역할)
      - Task 우선순위:
        1. 발 위치 (cost=500) — 가장 높은 우선순위
        2. CoM 위치 (cost=100)
        3. 토르소 위치/자세 (cost=50/5)
        4. 자세 정규화 (cost=1)
      - 최대 5회 반복, 위치 오차 < 0.1mm이면 조기 종료
      - 출력: q_ik (목표 관절각)
   
   d. 토크 계산
      tau = gravity_comp + Kp*(q_ik - q) - Kd*dq
   
   e. mj_step → IK config를 실제 자세로 동기화 (closed-loop)
```

### 특징
- mink 라이브러리의 QP 기반 IK 사용 (daqp solver)
- waist_roll, waist_pitch의 PD 게인 = 0 (자유 회전 허용)
- IK 결과를 매 스텝 실제 자세로 동기화하여 드리프트 방지

---

## g1_wbc_dynamics_qp.py — 역동역학 QP 모드

### 제어 흐름 (매 타임스텝)

```
1. 초기 안정화 (0~2초)
   → gravity comp + PD (키네마틱 모드와 동일)

2. 보행 시작 후
   a. Layer2: DCM PI 피드백 (키네마틱 모드와 동일)
   
   b. Support Phase 판별
      - DSP: 양발 접지 → 50/50 힘 분배
      - Left Support: 왼발 접지, 오른발 스윙
      - Right Support: 오른발 접지, 왼발 스윙
   
   c. Desired Contact Force 계산
      - 수직력: mg (SSP) 또는 mg/2 (DSP)
      - 수평력: -m*ω²*(com - zmp_target) — LIPM 기반 ZMP 제어
   
   d. Layer3: TaskSpaceWBC (QP)
      → feedforward 토크 생성 (아래 상세)
   
   e. 최종 토크
      tau = tau_ff(QP) + Kp*(q0 - q) - Kd*dq
      (PD는 초기 자세 q0 기준 — IK 없음)
```

---

## Layer3: TaskSpaceWBC (`Layer3.py`)

단일 QP로 가속도와 접촉력을 동시에 최적화하여 feedforward 토크를 생성한다.

### QP 변수
```
x = [ddq (nv), f_contact (n_c)]
```
- ddq: 일반화 가속도 (floating base 6 + actuated joints)
- f_contact: 접촉력 (접지 발당 3DoF)

### 목적 함수
```
min  Σ ||J_i * ddq - ddx_i||²_Wi  +  ||f - f_des||²_Wf  +  ε*||ddq||²
```

#### Task A: Torso 추종
```
ddx_torso = ddx_ref + Kd*(v_ref - v) + Kp*(pos_ref - pos)
            [위치 3D]   [자세 3D (axis-angle 오차)]
```

#### Task B: Swing Foot 추종 (SSP일 때만)
```
ddx_swing = ddx_ref + Kd*(v_ref - v) + Kp*(pos_ref - pos)
            [위치 3D]   [자세 3D]
```

### 등식 제약: Floating Base 동역학
```
S * (M*ddq + C - Jc^T*f) = 0
```
- S = [I₆ 0]: floating base 6DoF 선택 행렬
- 의미: floating base에는 액추에이터가 없으므로 순수 동역학만 만족

### 등식 제약: 접촉 가속도 = 0
```
Jc * ddq = -dJc*dq ≈ 0
```
- 접지 발이 지면에서 미끄러지지 않도록 구속

### 부등식 제약: 마찰 원뿔
```
fz ≥ 0
|fx| ≤ μ*fz
|fy| ≤ μ*fz
```
- 선형화된 마찰 원뿔 (5개 부등식/접촉점)

### 토크 복원
```
tau_full = M*ddq_opt + C - Jc^T*f_opt
tau_actuated = tau_full[actuator_dof_ids]
```

---

## 두 모드 비교

| 항목 | g1_kinematic | g1_wbc_dynamics_qp |
|------|-------------|-------------------|
| Layer3 | mink IK (위치 레벨) | QP WBC (가속도 레벨) |
| 출력 | 목표 관절각 q_ik | feedforward 토크 tau_ff |
| PD 기준 | q_ik (IK 결과) | q0 (초기 자세) |
| 접촉력 | 암묵적 (IK가 발 고정) | 명시적 QP 변수 |
| 마찰 제약 | 없음 | 마찰 원뿔 제약 |
| 동역학 일관성 | 없음 (키네마틱만) | 등식 제약으로 보장 |
| 계산 비용 | 낮음 | 높음 (QP 매 스텝) |

---

## config.py 주요 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| STEP_TIME | 0.7s | 한 스텝 시간 |
| DSP_TIME | 0.1s | Double Support Phase 시간 |
| STEP_LENGTH | 0.1m | 전진 보폭 |
| STEP_HEIGHT | 0.1m | 발 들어올리기 높이 |
| K_DCM | 2.0 | DCM 비례 게인 |
| COM_SHIFT_TIME | 0.2s | 보행 전 CoM 이동 시간 |
| LEG_FREQ | 2.0Hz | 다리 PD 게인 주파수 |
| TORSO_FREQ | 10.0Hz | 토르소 task-space PD 주파수 |
