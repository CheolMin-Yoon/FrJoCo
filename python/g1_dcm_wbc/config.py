"""DCM+WBC 통합 보행 제어기 파라미터 설정

DCM/config.py와 new_ZMP/config.py의 파라미터를 병합한 통합 설정 파일.
"""

import numpy as np

# ============================================
# 시뮬레이션
# ============================================
DT: float = 0.002
GRAVITY: float = 9.81

# ============================================
# 로봇 물리
# ============================================
ROBOT_MASS: float = 35.0
COM_HEIGHT: float = 0.69       # LIPM 모델용 (Layer2)
TORSO_HEIGHT: float = 0.809    # WBC floating base 목표 (Layer3)

# ============================================
# DCM 플래너 (Layer1)
# ============================================
STEP_TIME: float = 0.7
DSP_TIME: float = 0.08
INIT_DSP_EXTRA: float = 0.12
N_STEPS: int = 20
STEP_LENGTH: float = 0.1
STEP_WIDTH: float = 0.1185
STEP_HEIGHT: float = 0.08
ARM_SWING_AMP: float = 0.15

# ============================================
# DCM 트래커 (Layer2)
# ============================================
K_DCM: float = 2.0
KI_DCM: float = 0.0
DCM_INTEGRAL_LIMIT: float = 0.05
K_ZMP: float = 1.0
K_COM: float = 1.0

# ============================================
# Raibert Heuristic (Layer1)
# ============================================
RAIBERT_KP: float = 0.5

# ============================================
# Kinematic WBC (Layer3)
# ============================================
WBC_WAIST_KP: float = 5.0
WBC_CONTACT_KP: float = 5.0
WBC_TORSO_KP_POS: float = 2.0
WBC_TORSO_KP_ORI: float = 2.0
WBC_SWING_KP_POS: float = 2.0
WBC_SWING_KP_ORI: float = 2.0
WBC_DAMPING: float = 1e-4

# ============================================
# Dynamics WBC QP (Layer3)
# ============================================
QP_W_DDQ: float = 0.01
QP_W_F: float = 10.0
QP_FRICTION_COEF: float = 1.2
QP_DDQ_MAX: float = 300.0
QP_FORCE_MAX: float = 700.0

# ============================================
# PD 게인
# ============================================
# 다리 (hip, knee): Kp=200, Kd=10
# 발목 (ankle): Kp=50, Kd=5
# 더미 (waist_roll, waist_pitch): Kp=0, Kd=0
# 팔: Kp=100, Kd=5

LEG_KP: float = 200.0
LEG_KD: float = 10.0
ANKLE_KP: float = 50.0
ANKLE_KD: float = 5.0
WAIST_KP: float = 0.0
WAIST_KD: float = 0.0
ARM_KP: float = 100.0
ARM_KD: float = 5.0
