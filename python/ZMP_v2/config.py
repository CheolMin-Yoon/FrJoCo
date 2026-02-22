"""ZMP Preview Control + Kinematic WBC 통합 설정

g1_29dof_pos 모델 기준 (ctrl의 g1.xml과 동일 구조).
trajectory_planner.py, kinematic_wbc.py, play.py, play_sim.py에서 공유.

ctrl(lib_ZMPctrl.py) 파이프라인 (성공한 방식):
  1. mpc2humn: 오프라인 Cartesian 궤적 생성 (CoM, LF, RF)
     → CubicSpline 보간 (oCMx, oLx, oRx 등)
  2. cart2joint: 매 dt마다 numik 호출 → qtraj 배열
     → CubicSpline(ttraj, qtraj) → qspl[i](t) 보간
  3. controller: 실시간에서 qspl[i](data.time) → qdes
     → posCTRL=True: data.ctrl = qdes[6:] (position actuator)
  4. sim: data.ctrl = controller() → mj_step

ctrl 데이터 접근 방식:
  - CoM: data.subtree_com[0] (body 0 기준)
  - CoM 속도: data.subtree_linvel[0]
  - 발 위치: data.site('left_foot_site').xpos
  - 발 방향: data.site('left_foot_site').xmat
  - numik: mj_jacSubtreeCom(model, data, Jcm, 0)
"""

import numpy as np

# ============================================
# 시뮬레이션
# ============================================
DT: float = 0.001
GRAVITY: float = 9.81
XML_PATH: str = "../g1/scene_29dof_pos.xml"  # play.py 기준 상대 경로

# ============================================
# 로봇 물리
# ============================================
ROBOT_MASS: float = 35.0
COM_HEIGHT: float = 0.69       # LIPM 모델용 (ZMP preview)

# ============================================
# MuJoCo 이름 매핑
# ============================================
# actuator 이름 = joint 이름 (ctrl과 동일)
ACTUATOR_NAMES: list = [
    # Left leg (6)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # Right leg (6)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# 다리 관절만 (IK 대상, 12 DoF)
LEG_ACTUATOR_NAMES: list = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

# 상체 관절 (WBC 2순위 잠금 대상)
UPPER_BODY_ACTUATOR_NAMES: list = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Site / Body 이름 (ctrl과 동일)
LEFT_FOOT_SITE: str = "left_foot_site"
RIGHT_FOOT_SITE: str = "right_foot_site"
TORSO_BODY: str = "torso_link"

# CoM body 인덱스 — ctrl과 동일하게 body 0
COM_BODY_ID: int = 0

# ============================================
# 궤적 생성 (TrajectoryPlanner)
# ============================================
N_STEPS: int = 10
STEP_LENGTH: float = 0.1
STEP_WIDTH: float = 0.1185     # 좌우 발 간격 (y)
STEP_HEIGHT: float = 0.08
STEP_TIME: float = 0.8
DSP_RATIO: float = 0.15         # DSP 비율 (0~1)
PREVIEW_HORIZON: int = 1000    # preview 샘플 수

# ============================================
# Kinematic WBC (numik 스타일)
# ============================================
WBC_MAX_ITERS: int = 5
WBC_TOL: float = 1e-6
WBC_DAMPING: float = 1e-6
