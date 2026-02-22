"""ZMP-WBC 기반 보행 제어 설정"""

# ============================================
# 시뮬레이션 설정
# ============================================
dt: float = 0.002

# ============================================
# Layer1: Gait Generator
# ============================================
# 논문 Case 1: Stepping Impact
stepping_frequency: float = 2.2  # 보행 주파수 (Hz) — 느리게 조정
t_cycle: float = 1.0 / stepping_frequency  # 보행 주기 (~0.667s)
t_swing: float = t_cycle / 2.0   # Swing phase 시간 (~0.333s)
t_stance: float = t_cycle / 2.0  # Stance phase 시간 (~0.333s)
foot_height: float = 0.12        # 발 들어올리는 높이 (m)

# DSP (Double Support Phase)
dsp_ratio: float = 0.15          # 스텝 주기 대비 DSP 비율 (앞뒤 합산)
init_dsp_extra: float = 0.0     # 첫 스텝 추가 DSP 시간 (s) — 초기 weight shift

# Raibert Heuristic
raibert_kp: float = 0.2 # 발자국 위치 조정 게인

# DCM 플래너
k_dcm: float = 10.0       # DCM 비례 게인 (> 1.0)
ki_dcm: float = 1.0      # DCM 적분 게인

# ============================================
# Layer2: ZMP Control (External Contact Control)
# ============================================
robot_mass: float = 35.0      # G1 로봇 전체 질량 (kg)
com_height: float = 0.69      # CoM 높이 (m) - LIPM 모델용 (Layer2 전용)
gravity: float = 9.81         # 중력 가속도 (m/s^2)

# ============================================
# Layer3: Floating Base 목표 높이
# ============================================
torso_height: float = 0.809    # Torso 목표 높이 (m) - WBC floating base task 전용

# ==========================================ㄴㄴㄴ==
# Layer3: Kinematic WBC
# ============================================
# Task별 P 게인
wbc_waist_kp: float = 5.0         # Waist constraint 게인
wbc_contact_kp: float = 5.0       # Static contact 게인
wbc_torso_kp_pos: float = 2.0     # Floating base 위치 게인
wbc_torso_kp_ori: float = 2.0     # Floating base 자세 게인
wbc_swing_kp_pos: float = 2.0     # Swing leg 위치 게인
wbc_swing_kp_ori: float = 2.0     # Swing leg 자세 게인

# Damped Pseudo Inverse
wbc_damping: float = 1e-4  # 댐핑 계수 (lambda)

# ============================================
# Layer3: Dynamics WBC (QP)
# ============================================
# QP 목적 함수 가중치
qp_w_ddq: float = 0.01     # 가속도 추종 가중치
qp_w_f: float = 10.0       # 힘 추종 가중치

# 제약 조건
qp_friction_coef: float = 1.2  # 마찰 계수 (mu)

# 가속도 및 힘 제한
qp_ddq_max: float = 300.0   # 최대 가속도 (rad/s^2 or m/s^2)
qp_force_max: float = 700.0 # 최대 힘 (N)

