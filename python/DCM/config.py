"""DCM 기반 보행 제어 설정"""

GRAVITY: float = 9.81
DT: float = 0.001
STEP_TIME: float = 1.0
N_STEPS: int = 20
STEP_LENGTH: float = 0.15
STEP_WIDTH: float = 0.1185
STEP_HEIGHT: float = 0.15
DSP_TIME: float = 0.1
INIT_DSP_EXTRA: float = 0.12
K_DCM: float = 2.0
KI_DCM: float = 1.0
K_ZMP: float = 1.0
K_COM: float = 1.0
ARM_SWING_AMP: float = 0.15
