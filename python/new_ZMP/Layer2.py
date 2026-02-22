import numpy as np
from config import gravity, zmp_kd


class ExternalContactControl:
    def __init__(self, mass, com_height):
        self.mass = mass
        self.g = gravity 
        self.z_c = com_height 
        
        self.omega = np.sqrt(self.g / self.z_c)
        self.kd = zmp_kd   

    def compute_desired_force(self, com_pos, com_vel, com_acc_ref, stance_foot_pos, contact_state):
        
        target_zmp = np.array(stance_foot_pos)
        
        f_x = -self.mass * (self.g / self.z_c) * (com_pos[0] - target_zmp[0])
        f_y = -self.mass * (self.g / self.z_c) * (com_pos[1] - target_zmp[1])
        f_z = self.mass * self.g
        
        total_force = np.array([f_x, f_y, f_z])
        
        fr_left = np.zeros(3)
        fr_right = np.zeros(3)
        
        if contact_state[0] == 1 and contact_state[1] == 0: # Left Stance
            fr_left = total_force
            fr_right = np.zeros(3)
            contact_mode = "LEFT_STANCE"
            
        elif contact_state[0] == 0 and contact_state[1] == 1: # Right Stance
            fr_left = np.zeros(3)
            fr_right = total_force
            contact_mode = "RIGHT_STANCE"
            
        else: 
            fr_left = total_force / 2
            fr_right = total_force / 2
            contact_mode = "DOUBLE_SUPPORT"
        
        # 디버그 로그 (매 초마다)
        import time
        if not hasattr(self, '_last_log_time'):
            self._last_log_time = 0
        
        current_time = time.time()
        if current_time - self._last_log_time > 1.0:
            com_error = np.linalg.norm(com_pos[:2] - target_zmp[:2])
            print(f"[ZMP] mode={contact_mode}, CoM=[{com_pos[0]:.3f},{com_pos[1]:.3f}], "
                  f"ZMP=[{target_zmp[0]:.3f},{target_zmp[1]:.3f}], error={com_error:.4f}m, "
                  f"F_total=[{f_x:.1f},{f_y:.1f},{f_z:.1f}]N")
            self._last_log_time = current_time
            
        return fr_left, fr_right