"""
Layer1 발 궤적 시각화 (MuJoCo 없이, config + Layer1만 사용)

3D 공간에서:
- 왼발/오른발 swing 궤적 (Cycloid XY + Bezier Z)
- CoM (virtual) 궤적
- 발자국 착지점
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Layer1 import GaitGenerator
from config import dt, t_swing, t_stance, com_height, foot_height


def main():
    layer1 = GaitGenerator(T_s=t_swing, T_st=t_stance)

    # 초기 조건
    hip_width = 0.1185
    left_foot = np.array([0.0, hip_width, 0.0])
    right_foot = np.array([0.0, -hip_width, 0.0])
    com_pos = np.array([0.0, 0.0, com_height])
    com_vel = np.array([0.0, 0.0, 0.0])

    # 시뮬레이션 파라미터
    SIM_TIME = 6.0  # 초
    desired_vel_final = np.array([0.3, 0.0])
    ramp_time = 3.0

    # 로깅
    log_time = []
    log_com = []
    log_lf = []
    log_rf = []
    log_ref_lf = []
    log_ref_rf = []
    log_footsteps = []  # (x, y, z, leg_idx)

    sim_time = 0.0
    prev_swing_leg = -1

    n_steps = int(SIM_TIME / dt)
    for _ in range(n_steps):
        sim_time += dt

        # 속도 ramp
        progress = min(1.0, sim_time / ramp_time)
        desired_vel = desired_vel_final * progress

        # CoM 적분 (가상)
        com_pos[0] += desired_vel[0] * dt
        com_pos[1] += desired_vel[1] * dt
        com_vel[:2] = desired_vel

        # Gait state machine
        phase, contact_state, swing_leg_idx = layer1.state_machine(dt, sim_time)

        # 발자국 착지 기록 (swing leg 전환 시)
        if swing_leg_idx != prev_swing_leg and prev_swing_leg >= 0:
            if prev_swing_leg == 0:
                log_footsteps.append((*left_foot.copy(), 0))
            else:
                log_footsteps.append((*right_foot.copy(), 1))
        prev_swing_leg = swing_leg_idx

        # Raibert Heuristic (data=None → 기본 hip offset 사용)
        if swing_leg_idx == 0:  # left swing
            next_fs = layer1.Raibert_Heuristic_foot_step_planner(
                left_foot, com_pos, com_vel, desired_vel, swing_leg_idx
            )
            swing_start = layer1.get_swing_start_pos(swing_leg_idx)
            init_L = swing_start if swing_start is not None else left_foot
            ref_lf = layer1.generate_swing_trajectory(phase, init_L, next_fs)
            ref_rf = right_foot.copy()
            # swing foot 업데이트 (키네마틱 시뮬)
            left_foot = ref_lf.copy()
        else:  # right swing
            next_fs = layer1.Raibert_Heuristic_foot_step_planner(
                right_foot, com_pos, com_vel, desired_vel, swing_leg_idx
            )
            swing_start = layer1.get_swing_start_pos(swing_leg_idx)
            init_R = swing_start if swing_start is not None else right_foot
            ref_rf = layer1.generate_swing_trajectory(phase, init_R, next_fs)
            ref_lf = left_foot.copy()
            right_foot = ref_rf.copy()

        # 로깅 (매 20 step = 0.04초)
        if int(sim_time / dt) % 20 == 0:
            log_time.append(sim_time)
            log_com.append(com_pos.copy())
            log_lf.append(left_foot.copy())
            log_rf.append(right_foot.copy())
            log_ref_lf.append(ref_lf.copy())
            log_ref_rf.append(ref_rf.copy())

    # numpy 변환
    t = np.array(log_time)
    com = np.array(log_com)
    lf = np.array(log_lf)
    rf = np.array(log_rf)

    # ============================================
    # Plot 1: 3D 궤적
    # ============================================
    fig = plt.figure(figsize=(16, 10))

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot(lf[:, 0], lf[:, 1], lf[:, 2], 'b-', linewidth=1.5, label='Left foot')
    ax.plot(rf[:, 0], rf[:, 1], rf[:, 2], 'r-', linewidth=1.5, label='Right foot')
    ax.plot(com[:, 0], com[:, 1], com[:, 2], 'k--', linewidth=1, alpha=0.7, label='CoM')
    # 발자국 마커
    for (fx, fy, fz, leg) in log_footsteps:
        c = 'blue' if leg == 0 else 'red'
        ax.scatter(fx, fy, fz, color=c, s=30, zorder=5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Foot Trajectories')
    ax.legend(fontsize=8)

    # ============================================
    # Plot 2: XY 평면 (위에서 본 뷰)
    # ============================================
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(lf[:, 0], lf[:, 1], 'b-', linewidth=1, label='Left foot')
    ax2.plot(rf[:, 0], rf[:, 1], 'r-', linewidth=1, label='Right foot')
    ax2.plot(com[:, 0], com[:, 1], 'k--', linewidth=1, alpha=0.7, label='CoM')
    for (fx, fy, fz, leg) in log_footsteps:
        c = 'blue' if leg == 0 else 'red'
        ax2.plot(fx, fy, 'o', color=c, markersize=5)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (XY)')
    ax2.set_aspect('equal')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ============================================
    # Plot 3: X-Z 평면 (옆에서 본 뷰)
    # ============================================
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(lf[:, 0], lf[:, 2], 'b-', linewidth=1.5, label='Left foot Z')
    ax3.plot(rf[:, 0], rf[:, 2], 'r-', linewidth=1.5, label='Right foot Z')
    ax3.axhline(y=foot_height, color='gray', linestyle=':', alpha=0.5, label=f'max H={foot_height}m')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (XZ) - Bezier Profile')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ============================================
    # Plot 4: 시간 vs Z (Bezier 프로파일)
    # ============================================
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(t, lf[:, 2], 'b-', linewidth=1.5, label='Left foot Z')
    ax4.plot(t, rf[:, 2], 'r-', linewidth=1.5, label='Right foot Z')
    ax4.axhline(y=foot_height, color='gray', linestyle=':', alpha=0.5, label=f'max H={foot_height}m')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Foot Height over Time')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Layer1 Foot Trajectory (T_swing={t_swing:.3f}s, T_stance={t_stance:.3f}s, H={foot_height}m)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('plot_foot_traj.png', dpi=150)
    print("[INFO] Saved plot_foot_traj.png")
    plt.show()


if __name__ == "__main__":
    main()
