### Step 6.5: ZMP Preview Control — 순수 키네마틱 모드
#
# ★ 구조:
#   [섹션1~5] Preview Control + 발 궤적 생성 (오프라인)
#   [섹션6] mink IK → qpos 직접 대입 → mj_forward (kinematic only)
#
#   mj_step 없음, 토크 없음, PD 없음, 중력보상 없음
#   IK 결과를 qpos에 직접 넣고 mj_forward로 기구학만 계산

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import mink
import time

# MuJoCo Playground G1 XML (센서 포함, MJX 최적화)
import os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
xml_path = _os.path.normpath(_os.path.join(_script_dir, '../../model/g1/scene_23dof.xml'))

### ========================================================= ###
### ==========  1. Preview Control Gain 계산  ================ ###
### ========================================================= ###

# ── 물리 모델 로드 + 실측값 취득 (dynamics.py와 동일) ──
_model_tmp = mujoco.MjModel.from_xml_path(xml_path)
_data_tmp = mujoco.MjData(_model_tmp)
mujoco.mj_resetDataKeyframe(_model_tmp, _data_tmp, _model_tmp.key("knees_bent").id)
_data_tmp.ctrl[:] = _data_tmp.qpos[7:]
mujoco.mj_forward(_model_tmp, _data_tmp)

com_init = _data_tmp.subtree_com[1].copy()
lf_init = _data_tmp.site("left_foot").xpos.copy()
rf_init = _data_tmp.site("right_foot").xpos.copy()

print(f"[실측] CoM:  {com_init}")
print(f"[실측] LF:   {lf_init}")
print(f"[실측] RF:   {rf_init}")

del _model_tmp, _data_tmp  # 섹션6에서 새로 로드

# 파라미터
dt = 0.005          # 5ms
zc = com_init[2]    # ★ 실측 CoM 높이 사용 (하드코딩 제거)
g = 9.81            # 중력
N = 320             # Preview horizon (1.6s / 0.005)

# LIPM (책 144p)
# 상태 방정식 위치 속도 가속도
# LIPM 미분방정식: dd_x = (g/zc) * (x - p)  (p: ZMP)
# 이를 jerk 입력 모델로 변환:
#   상태: x_k = [x, dx, ddx]^T  (위치, 속도, 가속도)
#   입력: u_k = dddx (jerk, 가속도의 미분)
#   출력: p_k = x - (zc/g)*ddx (ZMP 위치)
#
# 연속 시간: dx/dt = [dx, ddx, u]^T
# 이산화 (ZOH, dt 간격):
#   x_{k+1} = A*x_k + B*u_k
#   p_k     = C*x_k
#
# ★ A(3x3), B(3x1), C(1x3)는 LIPM의 기본 시스템 행렬
#   → 바로 아래에서 A_aug, B_aug로 확장됨
#   → A, B는 [섹션3]에서 상태 업데이트에도 직접 사용: x_state = A @ x_state + B * u
#   → C는 [섹션3]에서 현재 ZMP 계산에 사용: p = C @ x_state
A = np.array([[1, dt, dt**2/2],
              [0, 1,  dt     ],
              [0, 0,  1      ]])

# 우리는 연속적인 가속도를 위해 Jerk를 입력으로 주고 적분해 사용
B = np.array([[dt**3/6],
              [dt**2/2],
              [dt     ]])

# LIPM의 출력 p = x - zc/g dd_x
C = np.array([[1, 0, -zc/g]])

# 144~145p [3] Improvement of Preview Controller
# LQR은 regulator임 따라서 ref를 주어도 상태를 0으로 가게하고 싶어하므로 error에 대한 상태를 추가해서 4x4로 확장
# 오차, delta 위치, delta 속도, delta 가속도로 (왜 delta를 쓰는지는 무한시간 LQR 비용함수와 관련)
#
# ★ Augmented System 구조 (4x4):
#   상태 벡터: x_aug = [e(k), Δx, Δdx, Δddx]^T
#     e(k) = Σ(p - p_ref): ZMP 추종 오차의 누적 (적분항)
#     Δx = x(k) - x(k-1): 원래 상태의 차분
#
#   A_aug = [[1,    C*A   ],    (4x4)
#            [0,     A    ]]
#   B_aug = [[C*B],            (4x1)
#            [ B ]]
#
#   → A_aug, B_aug는 바로 아래 DARE에 입력됨
#   → DARE 결과 P로부터 K를 구하고, K에서 Ks, Kx를 분리

nx = 3
A_aug = np.vstack([
    np.hstack([np.eye(1),  C @ A            ]),
    np.hstack([np.zeros((nx, 1)), A          ])
]) 

B_aug = np.vstack([C @ B, B])  

# C_aug (4x1): Preview gain Gd 계산에 사용
# Gd 재귀식에서 X = -Ac_bar^T @ P @ C_aug 의 초기값으로 들어감
C_aug = np.array([[1], [0], [0], [0]]) 

# Cost weights
# Q, R들은 양의 준정부호 행렬이어야함
Q_aug = np.zeros((4, 4))
Q_aug[0, 0] = 1.0       # ZMP 추종 오차 가중치
R = np.array([[1e-6]])  # Jerk 입력 가중치

# DARE 풀기 (나중에 Jax로 감싸서 병렬로 돌려야하므로 lqrax로 solver를 변경해야할 듯)
# https://github.com/MaxMSun/lqrax
# 식 4.76
# ★ DARE: A_aug^T P A_aug - P - A_aug^T P B_aug (R + B_aug^T P B_aug)^-1 B_aug^T P A_aug + Q_aug = 0
#   → P(4x4)를 구함. P는 바로 아래에서 K, Gd 계산에 모두 사용됨
P = la.solve_discrete_are(A_aug, B_aug, Q_aug, R)

# 144p의 식 4.75를 aug 행렬 버전으로
# ★ inv_term = (R + B^T P B)^-1  (스칼라)
#   → K 계산에 사용되고, 아래 Gd 재귀 계산에서도 재사용됨
inv_term = la.inv(R + B_aug.T @ P @ B_aug)
# ★ K (1x4) = inv_term @ B^T P A_aug
#   → K[0,0] = Ks (오차 적분 게인), K[0,1:] = Kx (상태 피드백 게인 1x3)
#   → Ks, Kx는 [섹션3]의 제어 법칙에서 사용
K = inv_term @ (B_aug.T @ P @ A_aug)

# f_i = (A^T@P@A + c^T@Q@c - A^T@P@b(R+B^T@P@B)^-1 @ b^T@P@A)

Ks = K[0, 0]       # 오차 적분 gain → [섹션3] u = -Ks * e_sum - Kx @ x - preview 에서 사용
Kx = K[0, 1:]      # 상태 피드백 gain (1x3) → [섹션3] 동일 제어 법칙에서 사용

# Preview gains (Gd)
# ★ Gd[j]: j스텝 미래의 ZMP ref가 현재 입력에 미치는 영향의 가중치
#   → [섹션3]에서 preview_x += Gd[j] * zmp_ref_x[k+j+1] 로 사용
#   → Gd는 보통 0에서 시작해서 음수로 갔다가 0으로 수렴하는 형태
Gd = np.zeros(N)
# ★ Ac_bar: 폐루프 시스템 행렬 (A_aug - B_aug @ K)
#   → Gd 재귀 계산에서 X를 업데이트하는 데 사용
Ac_bar = A_aug - B_aug @ K   # Closed-loop matrix
# ★ 재귀 초기값: X = -Ac_bar^T @ P @ C_aug  (4x1)
X = -Ac_bar.T @ P @ C_aug
# N스텝을 위한 재귀적 행렬 계산
# ★ 매 스텝: Gd[i] = inv_term @ B^T @ X  (스칼라)
#            X ← Ac_bar^T @ X  (다음 스텝으로 전파)
for i in range(N):
    Gd[i] = (inv_term @ (B_aug.T @ X)).item() # 우변이 책에서의 수식 f_i
    X = Ac_bar.T @ X

print(f"Ks = {Ks:.6f}")
print(f"Kx = {Kx}")
print(f"Gd[0:5] = {Gd[:5]}")

# 아 그러니까 결국 식 4.80을 python으로 작성하기 위해서 for i ~ 이거는 시그마 f_i를 구현한거구나

### ========================================================= ###
### ===========  2. ZMP Reference 궤적 생성  ================== ###
### ========================================================= ###

# 보행 파라미터
n_steps = 8            # 걸음 수
step_length = 0.15     # 한 걸음 길이 책에서는 sx
step_width = 0.1185    # 좌우 발 간격 책에서는 sy (g1의 xml에서 가져옴)
step_time = 0.8        # 한 걸음 시간 
dsp_time = 0.1         # Double Support Phase 시간 (s) -> 보통 전체 시간의 10~20% 정도라던데
ssp_time = 0.6         # Single Support Phase 시간 (s)

# ★ 실측 CoM x를 기준으로 footsteps 생성 (dynamics.py의 Layer1.plan_footsteps와 동일)
init_x = com_init[0]   # 실측 CoM x (≈0.035)

# 발자국 계획 [x, y]
# ★ footsteps: 각 걸음의 착지 위치 리스트
#   → 바로 아래 ZMP ref 생성에서 zmp_ref_x/y의 목표값으로 사용
#   → [섹션5] generate_foot_trajectory()에서 swing foot의 시작/끝 위치로 사용
#   → [섹션4] matplotlib top view에서 발자국 마커로 표시
footsteps = []
for i in range(n_steps):
    if i == 0:
        x = init_x             # ★ 첫 발: CoM x 위치 (Layer1과 동일)
        y = step_width         # 첫 발: 왼발 지지 (+step_width)
    else:
        x = init_x + i * step_length
        if i % 2 != 0:
            y = -step_width    # 홀수: 오른발
        else:
            y = step_width     # 짝수: 왼발
    footsteps.append([x, y])

# ZMP ref 생성 
# ★ total_samples: 전체 시뮬레이션 + preview 여유분
#   첫 스텝은 init_dsp_extra만큼 길어짐
init_dsp_extra_zmp = 0.0  # DSP 확장 없음
first_step_time = step_time + init_dsp_extra_zmp
total_walk_time = first_step_time + (n_steps - 1) * step_time
total_samples = int(total_walk_time / dt) + N
zmp_ref_x = np.zeros(total_samples)
zmp_ref_y = np.zeros(total_samples)

# 스텝별 시작/끝 인덱스 계산 (첫 스텝만 길게)
def _step_start_sample(i):
    if i == 0:
        return 0
    return int(first_step_time / dt) + (i - 1) * int(step_time / dt)

def _step_end_sample(i):
    if i == 0:
        return int(first_step_time / dt)
    return int(first_step_time / dt) + i * int(step_time / dt)

def _dsp_time_zmp(i):
    return dsp_time + init_dsp_extra_zmp if i == 0 else dsp_time

for i in range(n_steps):
    t_start = _step_start_sample(i)
    t_end = min(_step_end_sample(i), total_samples)
    dsp_samples_i = int(_dsp_time_zmp(i) / dt)
    ssp_start = t_start + dsp_samples_i
    
    if i == 0:
        # 첫 스텝: 전체 구간 첫 발 위치 (이전 발 없음)
        zmp_ref_x[t_start:t_end] = footsteps[i][0]
        zmp_ref_y[t_start:t_end] = footsteps[i][1]
    else:
        prev_foot = footsteps[i - 1]
        curr_foot = footsteps[i]
        
        # DSP: ramp 전환
        for k in range(dsp_samples_i):
            if t_start + k >= total_samples:
                break
            alpha = 0.5 * (1 - np.cos(np.pi * k / dsp_samples_i))  # 코사인 보간
            zmp_ref_x[t_start + k] = (1 - alpha) * prev_foot[0] + alpha * curr_foot[0]
            zmp_ref_y[t_start + k] = (1 - alpha) * prev_foot[1] + alpha * curr_foot[1]
        
        # SSP: 현재 발 위치에 고정
        if ssp_start < t_end:
            zmp_ref_x[ssp_start:t_end] = curr_foot[0]
            zmp_ref_y[ssp_start:t_end] = curr_foot[1]

# 마지막 발자국 이후 유지
last_filled = _step_end_sample(n_steps - 1)
if last_filled < total_samples:
    zmp_ref_x[last_filled:] = footsteps[-1][0]
    zmp_ref_y[last_filled:] = footsteps[-1][1]

print(f"\nZMP ref 생성 완료: {total_samples} samples, {total_samples*dt:.1f}s")
print(f"첫 스텝 시간: {first_step_time:.1f}s (DSP: {_dsp_time_zmp(0):.1f}s)")

pad_samples = 0  # padding 없음

### ========================================================= ###
### ===========  3. Preview Control 시뮬레이션  =============== ###
### ========================================================= ###
## AI
# ★ X, Y 방향을 완전히 독립적으로 제어 (LIPM은 X/Y 디커플링)
#   각 방향마다: 상태 x_state(3x1), 오차 누적 e_sum, 출력 리스트 com/zmp
#   [섹션1]에서 구한 A,B,C,Ks,Kx,Gd와 [섹션2]에서 구한 zmp_ref를 여기서 사용
#
# ★ 제어 법칙 (146p 식 4.80):
#   u(k) = -Ks * Σe(i) - Kx @ x(k) - Σ_{j=0}^{N-1} Gd(j) * p_ref(k+j+1)
#          ─────────    ──────────   ──────────────────────────────────────
#          적분 피드백    상태 피드백    preview feedforward (미래 ZMP ref 반영)
#
# ★ 출력: com_x, com_y, zmp_x, zmp_y (numpy 배열)
#   → com_x, com_y는 [섹션5]에서 com_traj로 합쳐져 [섹션6] IK 목표로 사용
#   → zmp_x, zmp_y는 [섹션4] matplotlib 검증에서 실제 ZMP로 표시

# X 방향 — ★ 실측 CoM 위치에서 시작 (0이 아님!)
x_state = np.array([[com_init[0]], [0.0], [0.0]])  # [pos, vel, acc]
com_x = []
zmp_x = []
e_sum_x = 0.0

# Y 방향 — ★ 실측 CoM y에서 시작 (≈0.0)
y_state = np.array([[com_init[1]], [0.0], [0.0]])
com_y = []
zmp_y = []
e_sum_y = 0.0

# ★ sim_length: preview에 필요한 미래 ref가 있는 범위까지만 시뮬레이션
#   total_samples - N 이후로는 미래 N스텝 ref를 확보할 수 없으므로
sim_length = total_samples - N

for k in range(sim_length):
    # --- X 방향 ---
    # ★ Step 1: 현재 ZMP 계산 (C @ x_state = x - zc/g * ddx)
    p_x = (C @ x_state)[0, 0]  # 현재 ZMP
    # ★ Step 2: 추종 오차 = 실제 ZMP - 목표 ZMP, 누적하여 적분항 구성
    e_x = p_x - zmp_ref_x[k]
    e_sum_x += e_x
    
    # ★ Step 3: Preview feedforward — 미래 N스텝의 ZMP ref를 Gd 가중합
    #   [섹션1]에서 계산한 Gd[j]와 [섹션2]에서 생성한 zmp_ref_x[k+j+1] 사용
    preview_x = 0.0
    for j in range(N):
        idx = k + j + 1          # 미래 j+1 스텝의 인덱스
        if idx < total_samples:
            preview_x += Gd[j] * zmp_ref_x[idx]
        else:
            preview_x += Gd[j] * zmp_ref_x[-1]  # 범위 초과 시 마지막 값 유지
    
    # 146p 식 4.80
    # ★ Step 4: 제어 입력 u (jerk) 계산
    #   Ks([섹션1]) * 오차누적 + Kx([섹션1]) @ 현재상태 + preview합
    u_x = -Ks * e_sum_x - Kx @ x_state.flatten() - preview_x
    
    # ★ Step 5: 상태 업데이트 — [섹션1]의 A,B 행렬로 다음 상태 계산
    #   x(k+1) = A @ x(k) + B * u(k)
    x_state = A @ x_state + B * u_x
    com_x.append(x_state[0, 0])   # CoM 위치 기록 → [섹션5] com_traj로
    zmp_x.append(p_x)             # ZMP 기록 → [섹션4] plot용
    
    # --- Y 방향 --- (X와 완전히 동일한 구조, zmp_ref_y 사용)
    p_y = (C @ y_state)[0, 0]
    e_y = p_y - zmp_ref_y[k]
    e_sum_y += e_y
    
    preview_y = 0.0
    for j in range(N):
        idx = k + j + 1
        if idx < total_samples:
            preview_y += Gd[j] * zmp_ref_y[idx]
        else:
            preview_y += Gd[j] * zmp_ref_y[-1]
    
    u_y = -Ks * e_sum_y - Kx @ y_state.flatten() - preview_y
    y_state = A @ y_state + B * u_y
    com_y.append(y_state[0, 0])
    zmp_y.append(p_y)

com_x = np.array(com_x)
com_y = np.array(com_y)
zmp_x = np.array(zmp_x)
zmp_y = np.array(zmp_y)
time_axis = np.arange(sim_length) * dt  # [섹션4] plot의 x축

print(f"\nPreview Control 완료: {sim_length} steps")
print(f"CoM X range: [{com_x.min():.3f}, {com_x.max():.3f}] m")
print(f"CoM Y range: [{com_y.min():.3f}, {com_y.max():.3f}] m")

### ========================================================= ###
### =========  4. Matplotlib 검증  ========================== ###
### ========================================================= ###
# ★ [섹션3]의 결과를 시각적으로 확인
#   - (0,0) X방향: com_x(초록) vs zmp_x(파랑) vs zmp_ref_x(빨강점선)
#   - (0,1) Y방향: com_y vs zmp_y vs zmp_ref_y → 좌우 흔들림 확인
#   - (1,0) Top View: CoM 궤적이 발자국 사이를 S자로 지나가는지 확인
#   - (1,1) Preview Gains: Gd가 0으로 수렴하는지 확인 (수렴 안하면 N 부족)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# X Position
axes[0, 0].plot(time_axis, com_x, 'g-', linewidth=2, label='CoM X')
axes[0, 0].plot(time_axis, zmp_x, 'b-', linewidth=1, label='ZMP X')
axes[0, 0].plot(time_axis, zmp_ref_x[:sim_length], 'r--', linewidth=1, label='ZMP ref X')
axes[0, 0].set_xlabel('Time [s]')
axes[0, 0].set_ylabel('X [m]')
axes[0, 0].set_title('X Direction')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Y Position
axes[0, 1].plot(time_axis, com_y, 'g-', linewidth=2, label='CoM Y')
axes[0, 1].plot(time_axis, zmp_y, 'b-', linewidth=1, label='ZMP Y')
axes[0, 1].plot(time_axis, zmp_ref_y[:sim_length], 'r--', linewidth=1, label='ZMP ref Y')
axes[0, 1].set_xlabel('Time [s]')
axes[0, 1].set_ylabel('Y [m]')
axes[0, 1].set_title('Y Direction')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Top View (X-Y)
axes[1, 0].plot(com_x, com_y, 'g-', linewidth=2, label='CoM')
axes[1, 0].plot(zmp_x, zmp_y, 'b.', markersize=1, alpha=0.3, label='ZMP')
axes[1, 0].plot(zmp_ref_x[:sim_length], zmp_ref_y[:sim_length], 'r.', markersize=1, alpha=0.3, label='ZMP ref')
for i, fs in enumerate(footsteps):
    color = 'blue' if i % 2 == 0 else 'red'
    axes[1, 0].plot(fs[0], fs[1], 's', color=color, markersize=10)
axes[1, 0].set_xlabel('X [m]')
axes[1, 0].set_ylabel('Y [m]')
axes[1, 0].set_title('Top View')
axes[1, 0].legend()
axes[1, 0].grid(True)
axes[1, 0].axis('equal')

# Preview Gains
axes[1, 1].plot(np.arange(N) * dt, Gd, 'k-', linewidth=1)
axes[1, 1].set_xlabel('Preview Time [s]')
axes[1, 1].set_ylabel('Gd')
axes[1, 1].set_title('Preview Gains')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('/home/frlab/mujoco_demo/tutorial/g1/step6.5_preview_control_result.png', dpi=150)
plt.show()

### ========================================================= ###
### ============  5. 발 궤적 생성  ============================= ###
### ========================================================= ###
# ★ [섹션2]의 footsteps를 기반으로 각 발의 3D 궤적 생성
#   - stance foot (지지발): 바닥에 고정 (z=0)
#   - swing foot (유각발): start→end로 이동하면서 z는 sin 곡선으로 들어올림
#   - DSP 구간에서는 양발 모두 바닥 (swing_phase=0)
#   → 출력: left_foot_traj, right_foot_traj (sim_length x 3)
#   → [섹션6]에서 feet_tasks[0], feet_tasks[1]의 IK 목표로 사용

def generate_foot_trajectory(footsteps, n_steps, step_time, dsp_time, step_height, dt, sim_length,
                              init_lf, init_rf, step_width_val, step_length_val,
                              pad_samples=0, init_dsp_extra=0.0):
    """발 궤적 생성 (swing foot: cycloid, stance foot: 고정)
    
    Args:
        footsteps: ZMP ref용 발자국 리스트 (CoM 기준 x)
        step_length_val: 한 걸음 길이 (m)
        init_dsp_extra: 첫 스텝 DSP 추가 시간 (s)
    """
    left_traj = np.zeros((sim_length, 3))
    right_traj = np.zeros((sim_length, 3))
    
    left_pos = init_lf.copy()
    right_pos = init_rf.copy()
    ground_z_lf = init_lf[2]
    ground_z_rf = init_rf[2]
    foot_x_start = init_lf[0]  # 양발 x 동일

    # ★ 발 기준 착지점 (footsteps x와 독립)
    foot_targets = []
    for i in range(n_steps):
        fx = foot_x_start if i == 0 else foot_x_start + i * step_length_val
        fy = footsteps[i][1]
        foot_targets.append((fx, fy))

    def step_time_for(i):
        return step_time + init_dsp_extra if i == 0 else step_time

    def dsp_time_for(i):
        return dsp_time + init_dsp_extra if i == 0 else dsp_time

    def step_start_time(step):
        return sum(step_time_for(j) for j in range(step))
    
    for k in range(sim_length):
        if k < pad_samples:
            left_traj[k] = left_pos
            right_traj[k] = right_pos
            continue
        
        t = (k - pad_samples) * dt
        
        # 가변 step_time에서 현재 step_idx 찾기
        step_idx = 0
        accum = 0.0
        for si in range(n_steps):
            if t < accum + step_time_for(si):
                step_idx = si
                break
            accum += step_time_for(si)
        else:
            step_idx = n_steps - 1
        
        t_in_step = t - step_start_time(step_idx)
        curr_dsp = dsp_time_for(step_idx)
        curr_step = step_time_for(step_idx)
        curr_ssp = curr_step - curr_dsp
        
        is_left_support = (step_idx % 2 == 0)
        
        if t_in_step < curr_dsp:
            swing_phase = 0.0
        else:
            swing_phase = (t_in_step - curr_dsp) / curr_ssp
        
        if is_left_support:
            if step_idx + 1 < n_steps:
                target = foot_targets[step_idx + 1]
                x = right_pos[0] + (target[0] - right_pos[0]) * swing_phase
                z = ground_z_rf + step_height * np.sin(np.pi * swing_phase)
                right_pos_now = np.array([x, init_rf[1], z])
            else:
                right_pos_now = right_pos
            left_traj[k] = left_pos
            right_traj[k] = right_pos_now
        else:
            if step_idx + 1 < n_steps:
                target = foot_targets[step_idx + 1]
                x = left_pos[0] + (target[0] - left_pos[0]) * swing_phase
                z = ground_z_lf + step_height * np.sin(np.pi * swing_phase)
                left_pos_now = np.array([x, init_lf[1], z])
            else:
                left_pos_now = left_pos
            left_traj[k] = left_pos_now
            right_traj[k] = right_pos
        
        # 착지 위치 업데이트
        if swing_phase >= 1.0 and step_idx + 1 < n_steps:
            target = foot_targets[step_idx + 1]
            if is_left_support:
                right_pos = np.array([target[0], init_rf[1], ground_z_rf])
            else:
                left_pos = np.array([target[0], init_lf[1], ground_z_lf])
    
    return left_traj, right_traj

step_height = 0.05  # swing foot 최대 높이 (5cm)
init_dsp_extra = 0.0  # DSP 확장 없음
left_foot_traj, right_foot_traj = generate_foot_trajectory(
    footsteps, n_steps, step_time, dsp_time, step_height, dt, sim_length,
    init_lf=lf_init, init_rf=rf_init, step_width_val=step_width,
    step_length_val=step_length, pad_samples=pad_samples,
    init_dsp_extra=init_dsp_extra
)

# ★ com_traj (sim_length x 3): [섹션3]의 com_x, com_y + 고정 높이 zc
#   → [섹션6]에서 com_task.set_target(target_com)의 입력으로 사용
com_traj = np.column_stack([com_x, com_y, np.full(sim_length, zc)])

print(f"\n궤적 생성 완료!")
print(f"  CoM: {com_traj.shape}")
print(f"  Left foot: {left_foot_traj.shape}")
print(f"  Right foot: {right_foot_traj.shape}")

### ========================================================= ###
### ==========  6. MuJoCo + Mink IK (Kinematic Mode)  ======= ###
### ========================================================= ###
# ★ 순수 키네마틱: mink IK → qpos 직접 대입 → mj_forward
#   mj_step 없음, 토크/PD/중력보상 없음

if __name__ == "__main__":

    # ── 물리 시뮬용 model/data ──
    model_sim = mujoco.MjModel.from_xml_path(xml_path)
    data_sim = mujoco.MjData(model_sim)

    # ── mink IK 계산용 configuration ──
    configuration = mink.Configuration(model_sim)

    tasks = [
        pelvis_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=0.01,
        ),
        torso_task := mink.FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=50.0,
            orientation_cost=5.0,
            lm_damping=0.01,
        ),
        com_task := mink.ComTask(cost=100.0),
        posture_task := mink.PostureTask(model_sim, cost=1.0),
    ]

    feet_tasks = []
    for foot in ["left_foot", "right_foot"]:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=100.0,
            lm_damping=0.01,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    limits = [mink.ConfigurationLimit(model_sim)]
    solver = "daqp"

    with mujoco.viewer.launch_passive(
        model=model_sim, data=data_sim, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model_sim, viewer.cam)

        # ── 초기 자세: knees_bent keyframe ──
        key_id = model_sim.key("knees_bent").id
        mujoco.mj_resetDataKeyframe(model_sim, data_sim, key_id)
        mujoco.mj_forward(model_sim, data_sim)

        # ── 관절 매핑 (팔 자세 고정용) ──
        arm_names = [
            "left_shoulder_pitch", "left_shoulder_roll",
            "left_shoulder_yaw", "left_elbow",
            "right_shoulder_pitch", "right_shoulder_roll",
            "right_shoulder_yaw", "right_elbow",
        ]
        _arm_jids = np.array([model_sim.joint(n + "_joint").id for n in arm_names])
        _arm_qpos = np.array([model_sim.jnt_qposadr[j] for j in _arm_jids])

        # ★ 상단에서 이미 실측한 com_init, lf_init, rf_init 재확인
        print(f"\n[섹션6] Left foot:  {data_sim.site('left_foot').xpos}")
        print(f"[섹션6] Right foot: {data_sim.site('right_foot').xpos}")
        print(f"[섹션6] CoM:        {data_sim.subtree_com[1]}")
        print(f"CoM z (실측): {com_init[2]:.6f}  (zc 설정값: {zc})")

        q0 = data_sim.qpos.copy()
        q0_arms = data_sim.qpos[_arm_qpos].copy()

        # mink 초기화
        configuration.update(q0)
        posture_task.set_target(q0)
        com_task.set_target(com_init)
        pelvis_task.set_target_from_configuration(configuration)
        torso_task.set_target_from_configuration(configuration)

        # ── 루프 설정 ──
        ik_dt = 0.005  # IK timestep
        max_iters = 10
        ik_damping = 1e-3

        traj_idx = 0

        print(f"\n순수 키네마틱 모드: IK → qpos 직접 대입 → mj_forward")

        print("\n" + "="*60)
        print("  Step 6.5: ZMP Preview Control + IK (Kinematic Mode)")
        print("  mj_forward only (no mj_step, no dynamics)")
        print("  Press ESC to exit")
        print("="*60)

        # 시각화 옵션
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True

        while viewer.is_running():
            step_start = time.time()

            traj_idx = min(int(data_sim.time / dt), sim_length - 1)
            if traj_idx >= sim_length - 1:
                break

            # ── 목표 설정 (궤적이 이미 월드좌표) ──
            target_com = np.array([
                com_traj[traj_idx, 0],
                com_traj[traj_idx, 1],
                com_init[2]
            ])

            left_target = left_foot_traj[traj_idx].copy()
            right_target = right_foot_traj[traj_idx].copy()

            com_task.set_target(target_com)
            feet_tasks[0].set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.identity(), left_target))
            feet_tasks[1].set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.identity(), right_target))
            posture_task.set_target(q0)

            # 상체 앞으로 살짝 숙이기 (pitch ≈ 3도)
            torso_pitch = np.radians(3.0)
            half = torso_pitch / 2.0
            torso_quat = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
            torso_target_pos = np.array([
                com_traj[traj_idx, 0],
                com_traj[traj_idx, 1],
                data_sim.body("torso_link").xpos[2]
            ])
            torso_task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(torso_quat), torso_target_pos))

            # ── IK 반복 수렴 ──
            for ik_iter in range(max_iters):
                vel = mink.solve_ik(configuration, tasks, ik_dt, solver,
                                    damping=ik_damping, limits=limits)
                configuration.integrate_inplace(vel, ik_dt)

            # ── 순수 kinematic: IK 결과를 qpos에 직접 대입 ──
            q_ik = configuration.q.copy()
            # 팔은 스폰 자세 유지
            q_ik[_arm_qpos] = q0_arms
            data_sim.qpos[:] = q_ik
            data_sim.time += dt
            mujoco.mj_forward(model_sim, data_sim)

            # ── 디버깅 ──
            actual_com = data_sim.subtree_com[1]
            actual_lf = data_sim.site("left_foot").xpos
            actual_rf = data_sim.site("right_foot").xpos
            com_err = np.linalg.norm(target_com[:2] - actual_com[:2])
            lf_pos_err = np.linalg.norm(left_target - actual_lf)
            rf_pos_err = np.linalg.norm(right_target - actual_rf)

            if traj_idx % 100 == 0:
                print(
                    f"t={data_sim.time:.2f}s | idx={traj_idx} | "
                    f"CoM_err={com_err*1000:.1f}mm | "
                    f"LF_err={lf_pos_err*1000:.1f}mm | RF_err={rf_pos_err*1000:.1f}mm"
                )

            # ── 시각화 ──
            viewer.user_scn.ngeom = 0

            # CoM 현재 위치 (바닥 투영, 큰 빨간 구체)
            g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0],
                                [actual_com[0], actual_com[1], 0.005], np.eye(3).flatten(),
                                [1, 0.2, 0.2, 0.9])
            viewer.user_scn.ngeom += 1

            # footsteps 표시 (구체)
            for fi, fs in enumerate(footsteps):
                if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                    break
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.015, 0, 0],
                                    [fs[0], fs[1], 0.005],
                                    np.eye(3).flatten(),
                                    [1, 0, 0, 0.8] if fi % 2 == 0 else [0, 0, 1, 0.8])
                viewer.user_scn.ngeom += 1

            # 전체 CoM 목표 궤적 (노란선, CoM 높이에 그리기)
            for i in range(0, sim_length - 1, 5):
                if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 10:
                    break
                p1 = np.array([com_traj[i, 0],
                               com_traj[i, 1], com_init[2]])
                p2 = np.array([com_traj[i+1, 0],
                               com_traj[i+1, 1], com_init[2]])
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, width=0.002,
                    from_=p1, to=p2)
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 1, 0, 0.6]
                viewer.user_scn.ngeom += 1

            # 전체 ZMP ref 궤적 (마젠타선, 바닥에 깔기)
            for i in range(0, sim_length - 1, 5):
                if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 10:
                    break
                p1 = np.array([zmp_ref_x[i], zmp_ref_y[i], 0.008])
                p2 = np.array([zmp_ref_x[i+1], zmp_ref_y[i+1], 0.008])
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, width=0.002,
                    from_=p1, to=p2)
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 0, 1, 0.6]
                viewer.user_scn.ngeom += 1

            # 발 궤적 표시 (빨간=RF, 시안=LF) — 현재 주변만
            vis_start = max(0, traj_idx - 100)
            vis_end = min(sim_length - 1, traj_idx + 200)
            for i in range(vis_start, vis_end - 1, 3):
                if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 2:
                    break
                # RF 궤적 (빨간)
                rf_p1 = right_foot_traj[i].copy()
                rf_p2 = right_foot_traj[i+1].copy()
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, width=0.002,
                    from_=rf_p1, to=rf_p2)
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 0.3, 0.3, 0.7]
                viewer.user_scn.ngeom += 1
                # LF 궤적 (시안)
                lf_p1 = left_foot_traj[i].copy()
                lf_p2 = left_foot_traj[i+1].copy()
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, width=0.002,
                    from_=lf_p1, to=lf_p2)
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [0, 1, 1, 0.7]
                viewer.user_scn.ngeom += 1

            # CoM 현재 주변 궤적 (초록선, 바닥)
            for i in range(vis_start, vis_end - 1):
                if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                    break
                p1 = np.array([com_traj[i, 0],
                               com_traj[i, 1], 0.01])
                p2 = np.array([com_traj[i+1, 0],
                               com_traj[i+1, 1], 0.01])
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    width=0.003, from_=p1, to=p2)
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [0, 1, 0, 0.5]
                viewer.user_scn.ngeom += 1

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        print("\n완료!")
