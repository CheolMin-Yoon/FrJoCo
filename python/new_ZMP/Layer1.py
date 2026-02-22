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
from config import foot_height, raibert_kp


class GaitGenerator():
    
    def __init__(self, T_s, T_st=None):
        self.T_swing = T_s
        self.T_stance = T_st if T_st is not None else T_s
        self.H = foot_height  
        self.len = 0.0  
        self.p = 0.0  
        
        self.phase = 0.0
        self.elapsed_time = 0.0
        self.contact_state = [1, 1]
        
        self.current_swing_foot = np.zeros(3) # 현재 (x, y, z)
        self.desired_swing_foot = np.zeros(3) # 목표 (x, y, z)
        
        self.swing_start_left_pos = None
        self.swing_start_right_pos = None
        
        # 목표 발자국 위치 고정 (스윙 시작 시 한 번만 계산)
        self.target_footstep_left = None
        self.target_footstep_right = None
    
    
    # 여기서 이제 step phase를 상태 머신으로 계산 후 0, 1 상태 반환
    def state_machine(self, dt, current_time):
        prev_p = self.p
        self.elapsed_time += dt
        
        cycle_time = self.T_swing + self.T_stance
        total_phase = (self.elapsed_time % cycle_time) / cycle_time
            
        if total_phase < 0.5:
            self.p = total_phase * 2
            self.contact_state = [1, 0]
            swing_leg_index = 1 # right
            
            # Swing 시작 시점 감지 (phase가 0.9 -> 0.1로 넘어갈 때)
            if prev_p > 0.9 and self.p < 0.1:
                print(f"[RESET] RIGHT swing cycle complete")
                self.swing_start_right_pos = None
                self.target_footstep_right = None  # 목표 위치도 리셋
            
        else:
            self.p = (total_phase - 0.5) * 2
            self.contact_state = [0, 1]
            swing_leg_index = 0 # left
            
            # Swing 시작 시점 감지
            if prev_p > 0.9 and self.p < 0.1:
                print(f"[RESET] LEFT swing cycle complete")
                self.swing_start_left_pos = None
                self.target_footstep_left = None  # 목표 위치도 리셋
        
        return self.p, self.contact_state, swing_leg_index
    
    # 속도 명령을 받아서 실시간으로 새로운 발자국 위치 생성
    def Raibert_Heuristic_foot_step_planner(self, current_foot_pos, torso_pos, torso_vel, desired_torso_vel, swing_leg_index, data=None):
        
        # Swing 시작 시점의 발 위치 저장 (한 번만)
        if swing_leg_index == 0 and self.swing_start_left_pos is None:
            self.swing_start_left_pos = current_foot_pos.copy()
            print(f"[START] LEFT: [{current_foot_pos[0]:.4f}, {current_foot_pos[1]:.4f}]")
        elif swing_leg_index == 1 and self.swing_start_right_pos is None:
            self.swing_start_right_pos = current_foot_pos.copy()
            print(f"[START] RIGHT: [{current_foot_pos[0]:.4f}, {current_foot_pos[1]:.4f}]")
        
        # 목표 발자국 위치 계산 (한 번만)
        if swing_leg_index == 0 and self.target_footstep_left is None:
            self.target_footstep_left = self._calculate_footstep(
                torso_pos, torso_vel, desired_torso_vel, swing_leg_index, data
            )
        elif swing_leg_index == 1 and self.target_footstep_right is None:
            self.target_footstep_right = self._calculate_footstep(
                torso_pos, torso_vel, desired_torso_vel, swing_leg_index, data
            )
        
        # 저장된 목표 위치 반환
        if swing_leg_index == 0:
            return self.target_footstep_left
        else:
            return self.target_footstep_right
    
    def _calculate_footstep(self, torso_pos, torso_vel, desired_torso_vel, swing_leg_index, data=None):
        """실제 발자국 위치 계산 (내부 함수)"""
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
        
        remaining_swing_time = (1 - self.p) * self.T_swing
        
        x = (torso_pos[0] + R * np.cos(theta) + 
             torso_vel[0] * remaining_swing_time + 
             0.5 * torso_vel[0] * self.T_stance + 
             Kp * (torso_vel[0] - desired_torso_vel[0]))
        
        y = (torso_pos[1] + R * np.sin(theta) + 
             torso_vel[1] * remaining_swing_time + 
             0.5 * torso_vel[1] * self.T_stance + 
             Kp * (torso_vel[1] - desired_torso_vel[1]))
        
        # 최소 step width 보장 (발이 엇갈리지 않도록)
        min_width = R  # hip width
        if swing_leg_index == 0:   # left foot → y > 0
            y = max(y, torso_pos[1] + min_width * 0.5)
        else:                      # right foot → y < 0
            y = min(y, torso_pos[1] - min_width * 0.5)
        
        z = 0.0
        
        footstep = np.array([x, y, z])
        
        # 디버그: 목표 발자국 위치 출력 (한 번만)
        leg_name = "LEFT" if swing_leg_index == 0 else "RIGHT"
        print(f"[TARGET] {leg_name}: [{x:.4f}, {y:.4f}] (World coordinate)")

        return footstep 
        
    # 여기서부터는 리아버트 휴리스틱으로부터 받은 새 발자국 위치로의 궤적을 생성 X-Y, Z
    
    # 수평 방향 (X-Y)
    def Cycloid_Trajectory(self, p, init_pos, next_xy):
        
        theta = 2 * np.pi * p
        cycloid_mod = (theta - np.sin(theta)) / (2 * np.pi)
        
        x_current = init_pos[0] + (next_xy[0] - init_pos[0]) * cycloid_mod
        y_current = init_pos[1] + (next_xy[1] - init_pos[1]) * cycloid_mod
        
        xy_current = [x_current, y_current]
        
        return np.array(xy_current)
        
    # 수직 방향 (Z)
    def Bezier_Curve_interpolation(self, s, init_z=0.0, target_z=0.0):
        # 단일 5차 베지어: init_z → H → target_z
        P0 = init_z
        P1 = init_z
        P2 = self.H
        P3 = self.H
        P4 = target_z
        P5 = target_z

        t = s
        c0 = (1 - t)**5
        c1 = 5 * (1 - t)**4 * t
        c2 = 10 * (1 - t)**3 * t**2
        c3 = 10 * (1 - t)**2 * t**3
        c4 = 5 * (1 - t) * t**4
        c5 = t**5

        return c0*P0 + c1*P1 + c2*P2 + c3*P3 + c4*P4 + c5*P5
    
    
    
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
    
    def generate_swing_trajectory(self, p, init_pos, next_footstep):
        
        xy_current = self.Cycloid_Trajectory(p, init_pos[:2], next_footstep[:2])
        z_current = self.Bezier_Curve_interpolation(p, init_pos[2], next_footstep[2])
        
        return np.array([xy_current[0], xy_current[1], z_current])