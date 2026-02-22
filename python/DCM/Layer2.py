import numpy as np
from typing import Tuple

from config import GRAVITY, K_DCM, KI_DCM, K_ZMP, K_COM, DT

class SimplifiedModelControl:

    def __init__(
        self,
        z_c: float,         # CoM 높이
        g: float = GRAVITY,       # 중력 가속도
        k_dcm: float = K_DCM,     # Eq (7) DCM 비례 게인 (> 1.0)
        ki_dcm: float = KI_DCM,   # Eq (7) DCM 적분 게인 (>= 0.0)
        k_zmp: float = K_ZMP,     # Eq (13) ZMP 오차 Gain (0 < K_zmp < w)
        k_com: float = K_COM,     # Eq (13) CoM 오차 Gain (K_com > w)
        dt: float = DT,
        dcm_integral_limit: float = 0.05,  # 적분 anti-windup 한계 (m)
    ):
        self.omega = np.sqrt(g / z_c)  

        self.kp_dcm = k_dcm 
        self.ki_dcm = ki_dcm 
        self.k_zmp = k_zmp
        self.k_com = k_com
        
        self.dt = dt
        self.dcm_integral_limit = dcm_integral_limit
        
        self.dcm_error_sum = np.zeros(2) # 적분항 누적

    # ========================================================================= #
    # 1. 현재 DCM 계산 (Measurement)
    # ========================================================================= #
    def calculate_current_dcm(self, com_pos: np.ndarray, com_vel: np.ndarray) -> np.ndarray:
        # XY 평면 추출
        pos_xy = com_pos[:2] if len(com_pos) > 2 else com_pos
        vel_xy = com_vel[:2] if len(com_vel) > 2 else com_vel
        
        # DCM 정의: ξ = x + dx/w
        current_dcm = pos_xy + vel_xy / self.omega
        return current_dcm
    
    # ========================================================================= #
    # 2. DCM Instantaneous Control (Eq. 7) -> 목표 ZMP 생성
    # ========================================================================= #
    def compute_desired_zmp(
        self,
        current_dcm: np.ndarray,   # 현재 DCM (측정값)
        ref_dcm: np.ndarray,       # 목표 DCM (플래너로부터)
        ref_dcm_vel: np.ndarray    # 목표 DCM 속도 (플래너로부터)
    ) -> np.ndarray:

        # 1) DCM 오차 계산
        e_dcm = current_dcm - ref_dcm
        
        # 2) 오차 적분 (anti-windup 클램핑)
        self.dcm_error_sum += e_dcm * self.dt
        self.dcm_error_sum = np.clip(
            self.dcm_error_sum,
            -self.dcm_integral_limit,
            self.dcm_integral_limit
        )
        
        # 3) 목표 ZMP 식 (7)
        # r_ref = ξ_ref - (1/ω) * dξ_ref + Kp * (ξ - ξ_ref) + Ki * ∫(ξ - ξ_ref)dt
        term_feedforward = ref_dcm - ref_dcm_vel / self.omega
        term_feedback = self.kp_dcm * e_dcm + self.ki_dcm * self.dcm_error_sum 
        
        desired_zmp = term_feedforward + term_feedback
        
        return desired_zmp 
    
    # ========================================================================= #
    # 3. ZMP-CoM Controller (Eq. 13) -> 목표 CoM 속도 생성
    # ========================================================================= #
    def compute_desired_com_vel(
        self,
        com_pos_meas: np.ndarray,    # 현재 CoM 위치 (측정값 x)
        com_pos_ref: np.ndarray,     # 목표 CoM 위치 (Ref x_ref)
        com_vel_ref: np.ndarray,     # 목표 CoM 속도 (Ref dx_ref)
        zmp_meas: np.ndarray,        # 현재 ZMP (측정값 r)
        desired_zmp: np.ndarray      # compute_desired_zmp()로부터 계산된 desired ZMP
    ) -> np.ndarray:
        
        # XY 평면 추출
        pos_xy = com_pos_meas[:2] if len(com_pos_meas) > 2 else com_pos_meas
        ref_pos_xy = com_pos_ref[:2] if len(com_pos_ref) > 2 else com_pos_ref
        
        # 식 (13): dx* = dx_ref - K_zmp*(r_ref - r) + K_com*(x_ref - x)
        # 1. Feedforward 속도
        term_ff_vel = com_vel_ref[:2] 
        
        # 2. ZMP 오차 피드백 
        term_zmp_fb = self.k_zmp * (desired_zmp - zmp_meas)
        
        # 3. CoM 위치 오차 피드백
        term_pos_fb = self.k_com * (ref_pos_xy - pos_xy)
        
        # 최종 목표 속도
        desired_com_vel = term_ff_vel - term_zmp_fb + term_pos_fb
        
        return desired_com_vel
    
    # ========================================================================= #
    # Main Loop Function
    # ========================================================================= #
    def control_step(
        self,
        # 측정값 (Sensors)
        meas_com_pos: np.ndarray,
        meas_com_vel: np.ndarray,
        meas_zmp: np.ndarray,
        # 목표값 (Trajectory Planner)
        ref_dcm: np.ndarray,
        ref_dcm_vel: np.ndarray,
        ref_com_pos: np.ndarray,
        ref_com_vel: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # 1. 현재 DCM 계산
        curr_dcm = self.calculate_current_dcm(meas_com_pos, meas_com_vel)
        
        # 2. Desired ZMP 계산 (Eq 7)
        desired_zmp = self.compute_desired_zmp(curr_dcm, ref_dcm, ref_dcm_vel)
        
        # 3. Desired CoM 속도 계산 (Eq 13)
        desired_com_vel = self.compute_desired_com_vel(
            meas_com_pos, 
            ref_com_pos, 
            ref_com_vel, 
            meas_zmp, 
            desired_zmp
        )
        
        # 반환: [최종 명령 속도, desired ZMP(디버깅용), 현재 DCM(디버깅용)]
        return desired_com_vel, desired_zmp, curr_dcm