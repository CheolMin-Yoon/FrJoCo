"""ZMP-WBC 기반 보행 제어 설정"""

# ============================================
# 시뮬레이션 설정
# ============================================
dt: float = 0.001

# ============================================
# PD 게인 (주파수 기반 자동 계산)
# ωn = 2π*f, Kp = ωn², Kd = 2*ωn  (ζ=1)
# ============================================
import math as _math

def _pd_gains(freq_hz: float):
    wn = 2 * _math.pi * freq_hz
    return wn ** 2, 2 * wn

LEG_FREQ:    float = 5.0
ANKLE_FREQ:  float = 5.0
ARM_FREQ:    float = 2.0
WRIST_FREQ:  float = 2.0

LEG_KP,   LEG_KD   = _pd_gains(LEG_FREQ)
ANKLE_KP, ANKLE_KD = _pd_gains(ANKLE_FREQ)
ARM_KP,   ARM_KD   = _pd_gains(ARM_FREQ)
WRIST_KP, WRIST_KD = _pd_gains(WRIST_FREQ)

# ============================================
# Layer1: Gait Generator
# ============================================
# 논문 Case 1: Stepping Impact
stepping_frequency: float = 2.2  # 보행 주파수 (Hz)
t_cycle: float = 1.0 / stepping_frequency  # 보행 주기 (~0.4545s)
t_swing: float = t_cycle / 2.0   # Swing phase 시간 (~0.227s)
t_stance: float = t_cycle / 2.0  # Stance phase 시간 (~0.227s)
foot_height: float = 0.1        # 발 들어올리는 높이 (m)

# Raibert Heuristic
raibert_kp: float = 0.2  # 발자국 위치 조정 게인

# ============================================
# Layer2: ZMP Control (External Contact Control)
# ============================================
robot_mass: float = 35.0      # G1 로봇 전체 질량 (kg)
com_height: float = 0.686      # CoM 높이 (m) - 실측값 (knees_bent keyframe 기준)
gravity: float = 9.81         # 중력 가속도 (m/s^2)

# ZMP 제어 게인
zmp_kd: float = 5.0   # ZMP 속도 제어 게인

# ============================================
# Layer3: Kinematic WBC
# ============================================
# Task별 P 게인 (1차 오차 동역학: dx = kp * error)
# kp가 높을수록 빠르게 수렴, 너무 높으면 진동
wbc_waist_kp: float = 50.0        # Waist constraint 게인
wbc_contact_kp: float = 40.0      # Static contact 게인 (높아야 발이 안 뜸)
wbc_torso_kp_pos: float = 30.0    # Floating base 위치 게인
wbc_torso_kp_ori: float = 20.0    # Floating base 자세 게인
wbc_swing_kp_pos: float = 50.0    # Swing leg 위치 게인
wbc_swing_kp_ori: float = 30.0     # Swing leg 자세 게인

# Damped Pseudo Inverse
wbc_damping: float = 1e-4  # 댐핑 계수 (lambda)

# ============================================
# Layer3: Dynamics WBC (QP)
# ============================================
# QP 목적 함수 가중치
qp_w_ddq: float =1.0     # 가속도 추종 가중치
qp_w_f: float = 100.0       # 힘 추종 가중치

# 제약 조건
qp_friction_coef: float = 1.0  # 마찰 계수 (mu)

# 가속도 및 힘 제한
qp_ddq_max: float = 500.0   # 최대 가속도 (rad/s^2 or m/s^2)
qp_force_max: float = 1000.0 # 최대 힘 (N)

