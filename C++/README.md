# FrMoCo — Frlab Humanoid Motion Controller

Unitree G1 휴머노이드 로봇을 위한 계층적 전신 보행 제어 프레임워크.  
MuJoCo 물리 시뮬레이션 + Pinocchio 동역학 라이브러리 기반으로 동작한다.

---

## 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                     main.cpp (1kHz 시뮬레이션 루프)               │
│                                                                 │
│  ① mj_step()          — MuJoCo 물리 전진                        │
│  ② Pinocchio FK/CoM   — 상태 최신화 (q, dq → FK, CoM)           │
│  ③ MPC 100Hz          — mpcLoop() (10스텝마다)                   │
│  ④ WBC 1kHz           — wbcLoop() or standingLoop()             │
│  ⑤ τ → g_d->ctrl[]   — MuJoCo 액추에이터에 토크 인가             │
└─────────────────────────────────────────────────────────────────┘
```

제어 파이프라인은 크게 3개 계층으로 나뉜다:

| 계층 | 주기 | 역할 |
|------|------|------|
| Layer 1: 궤적 계획 | 오프라인 + 100Hz | 발자국 계획, ZMP/CoM/발 궤적 생성 |
| Layer 2: MPC | 100Hz | LIPM 기반 Preview Control, CoM jerk 최적화 |
| Layer 3: WBC | 1kHz | Differential IK + Force Optimization + Torque Generation |

---

## 제어 파이프라인 상세

### Layer 1 — 궤적 계획 (오프라인)

컨트롤러 초기화 시 한 번에 전체 보행 궤적을 생성한다.

```
planFootsteps()          N개 발자국 좌표 생성
        │
        ├──→ ZmpTrajectory.generateZmpRef()
        │       DSP: 코사인 보간 (이전 발 → 현재 발)
        │       SSP: 현재 발 위치에 고정
        │       출력: zmp_ref_x, zmp_ref_y (MPC_DT 해상도)
        │
        ├──→ FootTrajectory.computeFull()
        │       XY: Cycloid 보간 (부드러운 스윙)
        │       Z:  5차 Bezier 곡선 (step_height 제어)
        │       출력: pos, vel, acc (WBC_DT 해상도)
        │
        └──→ 오프라인 MPC Forward Simulation
                전체 CoM 참조 궤적 사전 계산
                출력: com_ref_traj_ (MPC_DT 해상도)
```

### Layer 2 — MPC (100Hz)

LIPM(Linear Inverted Pendulum Model) 기반 Preview Control.

```
상태: x = [position, velocity, acceleration]
입력: u = jerk
출력: ZMP = Cd · x = [1, 0, -z_c/g] · x

이산시간 상태방정식:
  x_{k+1} = Ad · x_k + Bd · u_k

MPC QP (x축, y축 통합):
  min  α·||U||² + γ·||ZMP_pred - ZMP_ref||²
  결정변수: U = [jerk_x(0..N-1), jerk_y(0..N-1)]  (2N × 1)
  Horizon: 160 스텝 (1.6초)
```

매 MPC 스텝에서:
1. Pinocchio 실측값으로 상태 보정 (CoM pos/vel)
2. 가속도 보정: `ddx = ω²·(x - p_zmp_ref)`
3. QP 풀어서 최적 jerk 획득 (ProxSuite)
4. LIPM 상태 전파

### Layer 3 — WBC (1kHz)

WBC는 두 경로를 병렬로 실행하여 최종 토크를 합산한다.

```
                    ┌─────────────────────────────────┐
                    │  MPC 상태 (com_des, x_state)     │
                    │  발 궤적 (pos, vel, acc)          │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                                 ▼
     ┌─── 경로 A: IK + PD ───┐      ┌─── 경로 B: Force Control ───┐
     │                        │      │                              │
     │  WholeBodyIK           │      │  BalanceTask                 │
     │    태스크 스택:          │      │    ddc = Kp·Δc + Kd·Δċ      │
     │    [J_com (3)]         │      │                              │
     │    [J_rf  (6)]         │      │  ForceOptimizer (QP)         │
     │    [J_lf  (6)]         │      │    min ||K·F - u||² + F'WF   │
     │    = 15 × nv           │      │    s.t. 마찰원뿔 + CoP + 스윙  │
     │                        │      │    → F_hat (12×1)            │
     │  Damped Pseudo-Inverse │      │                              │
     │    dq = J⁺ · dx_err    │      │  WholeBodyTorqueGenerator    │
     │    q_des = q + dq·dt   │      │    [M  -S'] [ddq]   [-N + Jc'F]│
     │                        │      │    [Jc  0 ] [τ  ] = [-J̇q̇     ]│
     │  PD 토크:               │      │    → τ_ff                    │
     │    τ_fb = Kp·Δq + Kd·Δv│      │                              │
     └────────────┬───────────┘      └──────────────┬───────────────┘
                  │                                  │
                  └──────────┬───────────────────────┘
                             ▼
                      τ = τ_ff + τ_fb
```

---

## 모듈 구성

### dynamics_model/

| 클래스 | 파일 | 설명 |
|--------|------|------|
| `LIPM` | LIPM.hpp/cpp | 이산시간 LIPM 상태방정식 (Ad, Bd, Cd) |
| `CenterOfMass` | com_dynamics.hpp/cpp | CoM 동역학 매핑: `[D1; D2]·f = u` (선운동량 + 각운동량) |

### controller/

| 클래스 | 파일 | 설명 |
|--------|------|------|
| `LIPM_MPC` | LIPM_MPC.hpp/cpp | Kajita Preview Control MPC. 예측 행렬 구성 후 ProxSuite QP로 풀이 |

### trajectory_planner/

| 클래스 | 파일 | 설명 |
|--------|------|------|
| `ZmpTrajectory` | zmp_trajectory.hpp/cpp | 발자국 계획 + ZMP 참조 배열 생성 (DSP 코사인 보간, SSP 고정) |
| `FootTrajectory` | foot_trajectory.hpp/cpp | 발 스윙 궤적 (XY: Cycloid, Z: 5차 Bezier). pos/vel/acc 출력 |

### whole_body_controller/

| 클래스 | 파일 | 설명 |
|--------|------|------|
| `WholeBodyIK` | whole_body_ik.hpp/cpp | Differential IK (Resolved Motion Rate Control). CoM + 양발 6DoF 태스크 스택 |
| `ForceOptimizer` | Force_Optimizer.hpp/cpp | 지면 반력 QP 최적화. `min ||K·F - u||² + F'WF` s.t. 부등식 제약 |
| `WholeBodyTorqueGenerator` | whole_body_torque.hpp/cpp | 최적 반력 → 관절 토크 변환. 전신 동역학 `G·z = f` 풀이 |
| `WBC` (DBFC_core) | DBFC_core.hpp/cpp | IK 모드 / DBFC 모드 통합 인터페이스 |
| `BalanceTask` | tasks/balance_task.hpp/cpp | CoM PD 제어 → 목표 가속도 생성: `ddc = Kp·(c_des - c) + Kd·(ċ_des - ċ)` |

### constraints/

| 클래스 | 파일 | 설명 |
|--------|------|------|
| `ConstraintCore` | constraint_core.hpp | 제약조건 추상 인터페이스 (`A`, `l`, `u`) |
| `FrictionCone` | friction_cone.hpp/cpp | 선형화 마찰원뿔 (5 부등식/발). `μ_eff = μ/√2` |
| `CoPLimits` | cop_limits.hpp/cpp | CoP 범위 제약 (4 부등식/발). `CoP_x = -My/Fz`, `CoP_y = Mx/Fz` |
| `ConvexHull` | convex_hull.hpp/cpp | 지지 다각형 볼록 껍질 (미구현) |
| `TaskSpace` | task_space.hpp/cpp | 태스크 공간 제약 (미구현) |

### main_controller/

| 클래스 | 파일 | 설명 |
|--------|------|------|
| `G1WalkingController` | g1_walking_controller.hpp/cpp | 전체 파이프라인 오케스트레이션. 초기화 + mpcLoop + wbcLoop + standingLoop |

### utils/

| 함수 | 파일 | 설명 |
|------|------|------|
| `skewSymmetric()` | math_utils.hpp | 벡터 → skew-symmetric 행렬 (외적 연산용) |

---

## 주요 파라미터 (config.hpp)

```
제어 주기
  MuJoCo / WBC: 1kHz,  MPC: 100Hz

보행 파라미터
  STEP_TIME   = 0.8s     한 스텝 시간
  DSP_TIME    = 0.12s    양발 지지 시간
  STEP_HEIGHT = 0.06m    발 들어올림 높이
  STEP_LENGTH = 0.1m     보폭
  STEP_WIDTH  = 0.1185m  좌우 발 간격
  N_STEPS     = 20       총 스텝 수

로봇
  COM_HEIGHT  = 0.69m    CoM 높이 (LIPM 기준)
  FRICTION_MU = 1.0      마찰 계수

MPC
  MPC_HORIZON = 160      Preview 길이 (1.6초)
  MPC_ALPHA   = 1e-6     Jerk 페널티
  MPC_GAMMA   = 1.0      ZMP 추종 페널티

WBC 게인
  IK_KP / IK_KD       = 300 / 35     IK PD 게인
  BAL_KP / BAL_KD     = 20 / 9       BalanceTask CoM PD 게인
  FORCE_OPT_REG       = 1e-4         ForceOptimizer 정규화
  WBT_W_DDQ / WBT_W_TAU = 1e-6 / 1e-4  TorqueGenerator 가중치
```

---

## 로봇 모델

- Unitree G1 (29 DoF)
  - Floating base: 6 DoF (3 위치 + 3 회전)
  - Actuated joints: 23 DoF
  - 총 질량: ~35 kg
- MuJoCo XML: `model/g1/scene_29dof.xml`
- Pinocchio URDF: `g1_29dof.urdf`
- 발 프레임: `right_ankle_roll_link`, `left_ankle_roll_link`

---

## ForceOptimizer 제약조건 구조

결정변수 (12차원):
```
F = [Fx_R, Fy_R, Fz_R, Mx_R, My_R, Mz_R,   ← 오른발
     Fx_L, Fy_L, Fz_L, Mx_L, My_L, Mz_L]   ← 왼발
```

제약 스택 (24행):
```
[마찰원뿔  10행]   |Fx| ≤ μ_eff·Fz,  |Fy| ≤ μ_eff·Fz,  Fz ≥ 0  (양발)
[CoP 제약   8행]   dX_min·Fz ≤ -My ≤ dX_max·Fz  (양발)
[스윙 강제  6행]   스윙 발의 6개 변수 = 0
```

---

## 시각화

MuJoCo + GLFW 기반 실시간 렌더링:
- 로봇 반투명 (alpha=0.4)
- 접촉력 화살표 (파란색)
- ZMP 참조 궤적 (초록색 선)
- CoM 참조 궤적 (흰색 선)
- 접촉점 / 접촉력 표시

---

## 빌드 & 실행

의존성: Pinocchio, ProxSuite, MuJoCo 3.5.0, GLFW, Eigen3

```bash
# 빌드
./build.sh

# 실행
./run.sh
```

---

## 디렉토리 구조

```
FrMoCo/
├── main.cpp                              # 시뮬레이션 루프 + 시각화
├── CMakeLists.txt
├── build.sh / run.sh
├── include/
│   ├── config.hpp                        # 전역 파라미터
│   ├── dynamics_model/
│   │   ├── LIPM.hpp                      # 이산시간 LIPM
│   │   └── com_dynamics.hpp              # CoM 동역학 매핑
│   ├── controller/
│   │   └── LIPM_MPC.hpp                  # Preview Control MPC
│   ├── trajectory_planner/
│   │   ├── zmp_trajectory.hpp            # ZMP 참조 궤적
│   │   └── foot_trajectory.hpp           # 발 스윙 궤적
│   ├── whole_body_controller/
│   │   ├── whole_body_ik.hpp             # Differential IK
│   │   ├── Force_Optimizer.hpp           # 지면 반력 QP
│   │   ├── whole_body_torque.hpp         # 반력 → 토크 변환
│   │   ├── DBFC_core.hpp                 # WBC 통합 인터페이스
│   │   └── tasks/
│   │       ├── Task_core.hpp             # 태스크 추상 클래스
│   │       └── balance_task.hpp          # CoM PD 태스크
│   ├── constraints/
│   │   ├── constraint_core.hpp           # 제약 추상 클래스
│   │   ├── friction_cone.hpp             # 마찰원뿔
│   │   ├── cop_limits.hpp                # CoP 범위
│   │   ├── convex_hull.hpp               # 지지 다각형 (미구현)
│   │   └── task_space.hpp                # 태스크 공간 제약 (미구현)
│   ├── main_controller/
│   │   └── g1_walking_controller.hpp     # 전체 오케스트레이션
│   └── utils/
│       └── math_utils.hpp                # skew-symmetric 등
└── src/
    └── (위 헤더에 대응하는 .cpp 구현 파일들)
```
