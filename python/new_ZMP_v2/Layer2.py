import numpy as np
from config import gravity


class ExternalContactControl:
    def __init__(self, mass, com_height):
        self.mass = mass
        self.g = gravity 
        self.z_c = com_height 
        
        self.omega = np.sqrt(self.g / self.z_c)
        
    def compute_desired_force(self, com_pos, com_vel, com_acc_ref, contact_state, 
                              left_foot_pos, right_foot_pos, swing_leg_idx=-1,
                              dsp_progress=0.0, next_stance_is_left=True):
        """
        Args:
            swing_leg_idx: -1 (DSP), 0 (left swing), 1 (right swing)
            dsp_progress: 0~1, DSP 구간 내 진행률 (weight shift용)
            next_stance_is_left: DSP 후 어느 발이 stance가 되는지
        """
        
        # ZMP 타겟 설정
        if swing_leg_idx == -1:
            # DSP: 다음 stance 발 쪽으로 점진적 weight shift
            if next_stance_is_left:
                # 다음에 오른발 스윙 → 왼발이 stance → ZMP를 왼발로 이동
                alpha = 0.5 * (1 - np.cos(np.pi * dsp_progress))  # smooth 0→1
                target_zmp = (1 - alpha) * (left_foot_pos[:2] + right_foot_pos[:2]) / 2.0 \
                             + alpha * left_foot_pos[:2]
            else:
                # 다음에 왼발 스윙 → 오른발이 stance → ZMP를 오른발로 이동
                alpha = 0.5 * (1 - np.cos(np.pi * dsp_progress))
                target_zmp = (1 - alpha) * (left_foot_pos[:2] + right_foot_pos[:2]) / 2.0 \
                             + alpha * right_foot_pos[:2]
            
        elif contact_state[0] == 1 and contact_state[1] == 0:  # Left stance
            target_zmp = left_foot_pos[:2]
            
        elif contact_state[0] == 0 and contact_state[1] == 1:  # Right stance
            target_zmp = right_foot_pos[:2]
            
        else:  # fallback double support
            target_zmp = (left_foot_pos[:2] + right_foot_pos[:2]) / 2.0
        
        # ZMP 기반 힘 계산
        f_x = -self.mass * (self.g / self.z_c) * (com_pos[0] - target_zmp[0])
        f_y = -self.mass * (self.g / self.z_c) * (com_pos[1] - target_zmp[1])
        f_z = self.mass * self.g
        
        total_force = np.array([f_x, f_y, f_z])
        
        fr_left = np.zeros(3)
        fr_right = np.zeros(3)
        
        if contact_state[0] == 1 and contact_state[1] == 0: # Left Stance
            fr_left = total_force
            fr_right = np.zeros(3)
            
        elif contact_state[0] == 0 and contact_state[1] == 1: # Right Stance
            fr_left = np.zeros(3)
            fr_right = total_force
            
        else:  # Double support (DSP weight shift 반영)
            if swing_leg_idx == -1:
                # DSP: dsp_progress에 따라 다음 stance 발에 하중 이동
                alpha = 0.5 * (1 - np.cos(np.pi * dsp_progress))  # 0→1 smooth
                if next_stance_is_left:
                    w_left = 0.5 + 0.5 * alpha
                    w_right = 1.0 - w_left
                else:
                    w_right = 0.5 + 0.5 * alpha
                    w_left = 1.0 - w_right
                fr_left = w_left * total_force
                fr_right = w_right * total_force
            else:
                fr_left = total_force / 2
                fr_right = total_force / 2
            
        return fr_left, fr_right