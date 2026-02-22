'''
Integrated ZMP-WBC Framework for Dynamic Stability in
Humanoid Robot Locomotion의 구현체

Layer1은 두번째 이미지에서의 Gait Generation과 B. Foot Placement and Swing Leg Control의 구현체임

Section B. Foot Placement and Swing Leg Control는 
1. Raibert_Heuristic
2. Cycloid_Trajectory_Planning
3. Bezier_Curve_Interpolation 으로 구현되므로 class와 def를 선정

이를 위해 필요한 것은 

step phase를 반환할 상태머신 함수
'''

import numpy as np
from config import foot_height, raibert_kp, dsp_ratio, init_dsp_extra, gravity, com_height, k_dcm, ki_dcm, dt as CONFIG_DT


class GaitGenerator():
    """
    DSP/SSP 상태 머신 + Raibert Heuristic + DCM 피드백.
    
    보행 사이클 (한 스텝):
      [DSP] → [SSP]
      DSP: 양발 지지, CoM weight shift (DCM 기반)
      SSP: 한발 지지, 스윙 발 이동 (Raibert + Cycloid/Bezier)
    """
    
    def __init__(self, T_s, T_st=None, T_shift=None):
        self.T_swing = T_s
        self.T_stance = T_st if T_st is not None else T_s
        
        # DSP 시간 계산
        step_time = self.T_swing + self.T_stance
        self.T_dsp = dsp_ratio * step_time       # 스텝 내 DSP 시간
        self.T_ssp = step_time - self.T_dsp       # 스텝 내 SSP 시간
        self.T_shift = T_shift if T_shift is not None else init_dsp_extra
        
        self.H = foot_height
        self.len = 0.0
        self.p = 0.0          # swing phase (0~1), DSP 중에는 0
        
        self.phase = 0.0
        self.elapsed_time = 0.0
        self.contact_state = [1, 1]
        self.gait_started = False
        
        # 현재 보행 상태: "init_dsp", "dsp", "ssp"
        self.gait_phase_name = "init_dsp"
        
        # 스텝 카운터 (0부터, 짝수=오른발 스윙, 홀수=왼발 스윙)
        self.step_count = 0
        # 현재 스텝 내 경과 시간
        self.step_elapsed = 0.0
        
        self.current_swing_foot = np.zeros(3)
        self.desired_swing_foot = np.zeros(3)
        
        # Swing 시작 시점의 발 위치 저장
        self.swing_start_left_pos = None
        self.swing_start_right_pos = None
        
        # Stance 발 고정 위치 저장
        self.stance_left_pos = None
        self.stance_right_pos = None
        
        # Raibert 목표 착지점 (swing 시작 시 1회 고정)
        self.fixed_next_footstep_left = None
        self.fixed_next_footstep_right = None
        
        # DSP→SSP 전환 플래그 (swing 시작 초기화용)
        self._ssp_initialized = False
        
        # DCM 피드백 상태
        self.omega = np.sqrt(gravity / com_height)
        self.kp_dcm = k_dcm
        self.ki_dcm = ki_dcm
        self.dcm_error_sum = np.zeros(2)
        self.dcm_integral_limit = 0.05
    
    
    def state_machine(self, dt, current_time, left_foot_pos=None, right_foot_pos=None):
        """
        DSP/SSP 상태 머신.
        
        Returns:
            p: swing phase (0~1). DSP 중에는 0.
            contact_state: [left, right] 1=contact, 0=swing
            swing_leg_index: -1=DSP(양발), 0=left swing, 1=right swing
        """
        prev_phase_name = self.gait_phase_name
        self.elapsed_time += dt
        
        # 초기 stance 위치 저장 (최초 1회)
        if left_foot_pos is not None and self.stance_left_pos is None:
            self.stance_left_pos = left_foot_pos.copy()
        if right_foot_pos is not None and self.stance_right_pos is None:
            self.stance_right_pos = right_foot_pos.copy()
        
        # ── Phase 1: 초기 DSP (weight shift) ──
        if not self.gait_started:
            if self.elapsed_time < self.T_shift:
                self.p = 0.0
                self.contact_state = [1, 1]
                self.gait_phase_name = "init_dsp"
                return self.p, self.contact_state, -1
            else:
                # 초기 DSP 완료 → 보행 시작
                self.gait_started = True
                self.step_elapsed = 0.0
                self.step_count = 0
                self._ssp_initialized = False
                self.gait_phase_name = "dsp"
        
        # ── Phase 2+: 정상 보행 사이클 ──
        self.step_elapsed += dt
        step_time = self.T_dsp + self.T_ssp
        
        # 스텝 완료 → 다음 스텝으로
        if self.step_elapsed >= step_time:
            self.step_elapsed -= step_time
            self.step_count += 1
            self._ssp_initialized = False
            self.gait_phase_name = "dsp"
        
        # 현재 스텝 내 위치 판별
        if self.step_elapsed < self.T_dsp:
            # ── DSP 구간 ──
            self.gait_phase_name = "dsp"
            self.p = 0.0
            self.contact_state = [1, 1]
            swing_leg_index = -1
            
        else:
            # ── SSP 구간 ──
            self.gait_phase_name = "ssp"
            ssp_elapsed = self.step_elapsed - self.T_dsp
            self.p = np.clip(ssp_elapsed / self.T_ssp, 0.0, 1.0)
            
            # 짝수 스텝: 오른발 스윙 (왼발 stance)
            # 홀수 스텝: 왼발 스윙 (오른발 stance)
            if self.step_count % 2 == 0:
                swing_leg_index = 1  # right swing
                self.contact_state = [1, 0]
            else:
                swing_leg_index = 0  # left swing
                self.contact_state = [0, 1]
            
            # SSP 시작 시 1회 초기화
            if not self._ssp_initialized:
                self._ssp_initialized = True
                
                if swing_leg_index == 1:
                    # 오른발 스윙 시작
                    self.swing_start_right_pos = None  # 다음 호출에서 저장
                    self.fixed_next_footstep_right = None
                    if left_foot_pos is not None:
                        self.stance_left_pos = left_foot_pos.copy()
                else:
                    # 왼발 스윙 시작
                    self.swing_start_left_pos = None
                    self.fixed_next_footstep_left = None
                    if right_foot_pos is not None:
                        self.stance_right_pos = right_foot_pos.copy()
        
        return self.p, self.contact_state, swing_leg_index
    
    # ================================================================
    # DCM 피드백 (Layer2 보조 — CoM 레퍼런스 보정)
    # ================================================================
    def compute_dcm_feedback(self, com_pos_xy, com_vel_xy, stance_foot_xy):
        """
        실시간 DCM 피드백으로 desired ZMP 오프셋을 계산.
        
        DCM = x + dx/ω
        desired_zmp = stance_foot + Kp*(dcm - stance_foot)
        
        Returns:
            zmp_offset: (2,) ZMP 보정 벡터 (stance foot 기준)
        """
        dcm = com_pos_xy + com_vel_xy / self.omega
        
        # DCM 오차 (stance foot 기준)
        dcm_error = dcm - stance_foot_xy
        
        # 적분
        self.dcm_error_sum += dcm_error * CONFIG_DT
        self.dcm_error_sum = np.clip(
            self.dcm_error_sum, -self.dcm_integral_limit, self.dcm_integral_limit
        )
        
        zmp_offset = self.kp_dcm * dcm_error + self.ki_dcm * self.dcm_error_sum
        return zmp_offset
    
    def get_dsp_progress(self):
        """DSP 구간 내 진행률 (0~1). SSP이면 1.0 반환."""
        if self.gait_phase_name == "init_dsp":
            return np.clip(self.elapsed_time / self.T_shift, 0.0, 1.0)
        elif self.gait_phase_name == "dsp":
            return np.clip(self.step_elapsed / self.T_dsp, 0.0, 1.0) if self.T_dsp > 0 else 1.0
        else:
            return 1.0
    
    # 속도 명령을 받아서 실시간으로 리아버트 휴리스틱 기반의 새로운 발자국 위치 생성
    # 원래 나는 Pelvis (골반)의 CoM을 사용했는데 이 논문에서는 Torso (몸통)으로 하네
    def Raibert_Heuristic_foot_step_planner(self, current_foot_pos, torso_pos, torso_vel, desired_torso_vel, swing_leg_index, data=None):
        
        # Swing 시작 시점의 발 위치 저장
        if swing_leg_index == 0 and self.swing_start_left_pos is None:
            self.swing_start_left_pos = current_foot_pos.copy()
        elif swing_leg_index == 1 and self.swing_start_right_pos is None:
            self.swing_start_right_pos = current_foot_pos.copy()
        
        # 목표 착지점이 이미 고정되어 있으면 재계산 없이 반환 (진동 방지)
        if swing_leg_index == 0 and self.fixed_next_footstep_left is not None:
            return self.fixed_next_footstep_left
        if swing_leg_index == 1 and self.fixed_next_footstep_right is not None:
            return self.fixed_next_footstep_right
        
        # 초기 위치 사용
        if swing_leg_index == 0:
            init_pos = self.swing_start_left_pos if self.swing_start_left_pos is not None else current_foot_pos
        else:
            init_pos = self.swing_start_right_pos if self.swing_start_right_pos is not None else current_foot_pos
        
        Kp = raibert_kp
        
        if data is not None:
            if swing_leg_index == 0:  # left
                hip_pos = data.body("left_hip_yaw_link").xpos
            else:                    # right
                hip_pos = data.body("right_hip_yaw_link").xpos
            
            hip_offset = hip_pos[:2] - torso_pos[:2]
            R = np.linalg.norm(hip_offset)
            theta = np.arctan2(hip_offset[1], hip_offset[0])
        else:
            R = 0.1185  # hip width / 2
            theta = np.pi / 2 if swing_leg_index == 0 else -np.pi / 2
        
        # swing 시작 시점(p≈0)에서 계산하므로 remaining_swing_time = T_swing 전체
        x = (torso_pos[0] + R * np.cos(theta) +
             torso_vel[0] * self.T_swing +
             0.5 * torso_vel[0] * self.T_stance +
             Kp * (desired_torso_vel[0] - torso_vel[0]))
        
        y = (torso_pos[1] + R * np.sin(theta) +
             torso_vel[1] * self.T_swing +
             0.5 * torso_vel[1] * self.T_stance +
             Kp * (desired_torso_vel[1] - torso_vel[1]))
        
        z = 0.0
        
        result = np.array([x, y, z])
        
        # 고정 저장 (이후 스텝에서 재계산 안 함)
        if swing_leg_index == 0:
            self.fixed_next_footstep_left = result.copy()
        else:
            self.fixed_next_footstep_right = result.copy()
        
        return result
        
    # 여기서부터는 리아버트 휴리스틱으로부터 받은 새 발자국 위치로의 궤적을 생성 X-Y, Z
    
    # 수평 방향 (X-Y) — 위치, 속도, 가속도
    def Cycloid_Trajectory(self, p, init_pos, next_xy):
        dp = next_xy - init_pos
        theta = 2 * np.pi * p
        
        # 위치: (theta - sin(theta)) / (2*pi)
        mj = (theta - np.sin(theta)) / (2 * np.pi)
        pos = init_pos + dp * mj
        
        # dp/ds: d(mj)/dp = (1 - cos(theta)) * 2*pi / (2*pi) = 1 - cos(theta)
        dmj = 1.0 - np.cos(theta)
        vel = dp * dmj  # dp/ds (phase 미분, 시간 미분은 caller에서 /T_ssp)
        
        # d2p/ds2: d(dmj)/dp = sin(theta) * 2*pi
        d2mj = 2.0 * np.pi * np.sin(theta)
        acc = dp * d2mj
        
        return pos, vel, acc
        
    # 수직 방향 (Z) — 위치, 속도, 가속도
    def Bezier_Curve_interpolation(self, s, init_z=0.0, target_z=0.0):
        # 단일 5차 베지어: init_z → H → target_z
        P = np.array([init_z, init_z, self.H, self.H, target_z, target_z])

        t = s
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        u = 1.0 - t
        u2 = u * u
        u3 = u2 * u
        u4 = u3 * u
        u5 = u4 * u

        # 위치
        c = np.array([u5, 5*u4*t, 10*u3*t2, 10*u2*t3, 5*u*t4, t5])
        pos = c @ P

        # 1차 미분 (ds 기준)
        dc = np.array([
            -5*u4,
            5*u3*(1 - 5*t),
            10*u2*t*(2 - 5*t),
            10*u*t2*(3 - 5*t),
            5*t3*(4 - 5*t),
            5*t4
        ])
        vel = dc @ P

        # 2차 미분 (ds^2 기준)
        d2c = np.array([
            20*u3,
            20*u2*(5*t - 2),
            10*u*(10*t2 - 8*t + 1),
            10*t*(10*t2 - 12*t + 3),
            20*t2*(5*t - 3),
            20*t3
        ])
        acc = d2c @ P

        return pos, vel, acc
    
    
    
    def get_swing_start_pos(self, swing_leg_idx):
        """
        Swing 시작 시점의 발 위치 반환
        
        Args:
            swing_leg_idx: int, 0=left, 1=right
        
        Returns:
            np.ndarray (3,) or None
        """
        if swing_leg_idx == 0:
            return self.swing_start_left_pos
        else:
            return self.swing_start_right_pos
    
    def get_stance_foot_pos(self, swing_leg_idx):
        """
        현재 stance 발의 고정 위치 반환
        
        Args:
            swing_leg_idx: int, -1=초기, 0=left swing(right stance), 1=right swing(left stance)
        
        Returns:
            tuple: (stance_left_pos, stance_right_pos)
        """
        return self.stance_left_pos, self.stance_right_pos
    
    def generate_swing_trajectory(self, p, init_pos, next_footstep):
        """
        Returns:
            pos: (3,) 위치
            vel: (3,) 속도 (phase 미분, 시간 변환은 caller에서 /T_ssp)
            acc: (3,) 가속도 (phase 미분^2, 시간 변환은 /T_ssp^2)
        """
        xy_pos, xy_vel, xy_acc = self.Cycloid_Trajectory(p, init_pos[:2], next_footstep[:2])
        z_pos, z_vel, z_acc = self.Bezier_Curve_interpolation(p, init_pos[2], next_footstep[2])
        
        pos = np.array([xy_pos[0], xy_pos[1], z_pos])
        vel = np.array([xy_vel[0], xy_vel[1], z_vel])
        acc = np.array([xy_acc[0], xy_acc[1], z_acc])
        
        return pos, vel, acc