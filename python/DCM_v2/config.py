"""DCM 기반 보행 제어 설정"""

GRAVITY: float = 9.81
DT: float = 0.001
STEP_TIME: float = 0.8
N_STEPS: int = 20
STEP_LENGTH: float = 0.1
STEP_WIDTH: float = 0.1185
STEP_HEIGHT: float = 0.12
DSP_TIME: float = 0.12
INIT_DSP_EXTRA: float = 0.08
COM_SHIFT_TIME: float = 0.0   # 보행 전 CoM을 지지발로 옮기는 시간 (s)
K_DCM: float = 1.0
KI_DCM: float = 1.0
K_ZMP: float = 10.0
K_COM: float = 1.0

# ============================================
# 로봇 물리
# ============================================
ROBOT_MASS: float = 35.0
COM_HEIGHT: float = 0.6846
TORSO_HEIGHT: float = 0.8027

# ============================================
# QP WBC (Layer3)
# ============================================
QP_W_DDQ: float = 0.001
QP_W_F: float = 1.0
QP_FRICTION_COEF: float = 1.0
QP_DDQ_MAX: float = 300.0
QP_FORCE_MAX: float = 700.0

import math as _math

def _pd_gains(freq_hz: float):
    wn = 2 * _math.pi * freq_hz
    return wn ** 2, 2 * wn

LEG_FREQ:   float = 2.1 
ANKLE_FREQ: float = 2.1 
ARM_FREQ:   float = 2.0  

LEG_KP,   LEG_KD   = _pd_gains(LEG_FREQ)
ANKLE_KP, ANKLE_KD = _pd_gains(ANKLE_FREQ)
ARM_KP,   ARM_KD   = _pd_gains(ARM_FREQ)

# Task-Space WBC 게인
TORSO_FREQ:     float = 50.0   
SWING_FREQ_POS: float = 10.0   
SWING_FREQ_ORI: float = 1.0   

TORSO_KP_POS, TORSO_KD_POS = _pd_gains(TORSO_FREQ)
TORSO_KP_ORI, TORSO_KD_ORI = _pd_gains(TORSO_FREQ)
SWING_KP_POS,  SWING_KD_POS  = _pd_gains(SWING_FREQ_POS)
SWING_KP_ORI,  SWING_KD_ORI  = _pd_gains(SWING_FREQ_ORI)

# QP task 가중치
W_TORSO_POS: float = 50.0
W_TORSO_ORI: float = 5.0
W_SWING_POS: float = 50.0
W_SWING_ORI: float = 1.0
