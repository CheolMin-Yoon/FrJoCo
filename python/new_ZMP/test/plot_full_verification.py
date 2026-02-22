"""
전체 로직 검증용 시각화 (LIPM + Layer1 + Layer2)

MuJoCo 없이 순수 Python으로 LIPM 동역학 + Layer1 Gait + Layer2 ZMP 힘 계산을 시뮬레이션.
FuncAnimation으로 애니메이션 재생.

참고: reference/BipedalWalkingRobots/LIPM/demo_LIPM_3D.py
      reference/ModelBasedFootstepPlanning-IROS2024/LIPM/demo_LIPM_3D_vt.py

서브플롯 구성 (2x3):
  [0,0] 3D LIPM 애니메이션 (CoM + legs + feet + trajectory)
  [0,1] CoM velocity (actual vs desired)
  [0,2] Foot Z height profile
  [1,0] 2D XY top view (CoM + feet + footsteps)
  [1,1] Step length / width
  [1,2] ZMP desired force (fx, fy, fz)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

from Layer1 import GaitGenerator
from Layer2 import ExternalContactControl
from config import (
    dt, t_swing, t_stance, com_height, foot_height,
    robot_mass, gravity
)

# ============================================
# LIPM 동역학 (단순 선형 역진자)
# ============================================
class SimpleLIPM:
    """x_ddot = (g/z_c) * (x - p_x), LIPM 동역학을 Euler 적분"""
    def __init__(self, mass, z_c, g=9.81, dt=0.002):
        self.mass = mass
        self.z_c = z_c
        self.g = g
        self.dt = dt
        self.omega2 = g / z_c  # (g/z_c)

    def step(self, com_pos, com_vel, stance_foot_pos):
        """1 step Euler 적분, XY만"""
        acc_x = self.omega2 * (com_pos[0] - stance_foot_pos[0])
        acc_y = self.omega2 * (com_pos[1] - stance_foot_pos[1])
        com_vel[0] += acc_x * self.dt
        com_vel[1] += acc_y * self.dt
        com_pos[0] += com_vel[0] * self.dt
        com_pos[1] += com_vel[1] * self.dt
        return com_pos, com_vel


# ============================================
# 시뮬레이션
# ============================================
def run_simulation():
    SIM_TIME = 8.0
    RAMP_TIME = 3.0
    desired_vel_final = np.array([0.3, 0.0])

    hip_width = 0.1185
    layer1 = GaitGenerator(T_s=t_swing, T_st=t_stance)
    layer2 = ExternalContactControl(mass=robot_mass, com_height=com_height)
    lipm = SimpleLIPM(mass=robot_mass, z_c=com_height, g=gravity, dt=dt)

    # 초기 조건
    com_pos = np.array([0.0, 0.0, com_height])
    com_vel = np.array([0.0, 0.0, 0.0])
    left_foot = np.array([0.0, hip_width, 0.0])
    right_foot = np.array([0.0, -hip_width, 0.0])

    # 로깅
    log = {
        'time': [], 'com_pos': [], 'com_vel': [],
        'lf': [], 'rf': [],
        'desired_vel': [],
        'fr_left': [], 'fr_right': [],
        'phase': [], 'swing_leg': [], 'contact': [],
    }
    footsteps = []  # (x, y, leg_idx)
    step_lengths = []
    step_widths = []
    step_times = []
    prev_stance_pos = np.array([0.0, 0.0, 0.0])

    sim_time = 0.0
    prev_swing_leg = -1
    n_steps = int(SIM_TIME / dt)
    log_interval = 10  # 매 10 step (0.02s)

    for step_i in range(n_steps):
        sim_time += dt

        # 속도 ramp
        progress = min(1.0, max(0.0, sim_time / RAMP_TIME))
        desired_vel = desired_vel_final * progress

        # Gait state machine
        phase, contact_state, swing_leg_idx = layer1.state_machine(dt, sim_time)

        # Stance foot 결정
        if contact_state[0] == 1 and contact_state[1] == 0:
            stance_foot_pos = left_foot.copy()
        elif contact_state[0] == 0 and contact_state[1] == 1:
            stance_foot_pos = right_foot.copy()
        else:
            stance_foot_pos = (left_foot + right_foot) / 2.0

        # LIPM 동역학 적분
        com_pos, com_vel = lipm.step(com_pos, com_vel, stance_foot_pos)

        # 발자국 전환 감지
        if swing_leg_idx != prev_swing_leg and prev_swing_leg >= 0:
            landed_pos = left_foot.copy() if prev_swing_leg == 0 else right_foot.copy()
            footsteps.append((landed_pos[0], landed_pos[1], prev_swing_leg))
            # step length/width 계산
            sl = np.abs(landed_pos[0] - prev_stance_pos[0])
            sw = np.abs(landed_pos[1] - prev_stance_pos[1])
            step_lengths.append(sl)
            step_widths.append(sw)
            step_times.append(sim_time)
            prev_stance_pos = landed_pos.copy()
        prev_swing_leg = swing_leg_idx

        # Raibert Heuristic + Swing trajectory
        if swing_leg_idx == 0:  # left swing
            next_fs = layer1.Raibert_Heuristic_foot_step_planner(
                left_foot, com_pos, com_vel, desired_vel, swing_leg_idx)
            swing_start = layer1.get_swing_start_pos(swing_leg_idx)
            init_L = swing_start if swing_start is not None else left_foot
            ref_lf = layer1.generate_swing_trajectory(phase, init_L, next_fs)
            ref_rf = right_foot.copy()
            left_foot = ref_lf.copy()
        else:  # right swing
            next_fs = layer1.Raibert_Heuristic_foot_step_planner(
                right_foot, com_pos, com_vel, desired_vel, swing_leg_idx)
            swing_start = layer1.get_swing_start_pos(swing_leg_idx)
            init_R = swing_start if swing_start is not None else right_foot
            ref_rf = layer1.generate_swing_trajectory(phase, init_R, next_fs)
            ref_lf = left_foot.copy()
            right_foot = ref_rf.copy()

        # Layer2: ZMP force
        fr_left, fr_right = layer2.compute_desired_force(
            com_pos, com_vel, np.zeros(3), stance_foot_pos, contact_state)

        # 로깅
        if step_i % log_interval == 0:
            log['time'].append(sim_time)
            log['com_pos'].append(com_pos.copy())
            log['com_vel'].append(com_vel.copy())
            log['lf'].append(left_foot.copy())
            log['rf'].append(right_foot.copy())
            log['desired_vel'].append(desired_vel.copy())
            log['fr_left'].append(fr_left.copy())
            log['fr_right'].append(fr_right.copy())
            log['phase'].append(phase)
            log['swing_leg'].append(swing_leg_idx)
            log['contact'].append(contact_state.copy())

    # numpy 변환
    for k in log:
        log[k] = np.array(log[k])

    return log, footsteps, step_lengths, step_widths, step_times


# ============================================
# 애니메이션 클래스
# ============================================
def create_animation(log, footsteps, step_lengths, step_widths, step_times):
    t = log['time']
    com = log['com_pos']
    lf = log['lf']
    rf = log['rf']
    com_vel = log['com_vel']
    des_vel = log['desired_vel']
    fr_left = log['fr_left']
    fr_right = log['fr_right']
    data_len = len(t)

    fig = plt.figure(figsize=(18, 10))
    spec = gridspec.GridSpec(nrows=2, ncols=3, height_ratios=[1.8, 1],
                             hspace=0.35, wspace=0.3)

    # ---- [0,0] 3D LIPM ----
    ax3d = fig.add_subplot(spec[0, 0], projection='3d')
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_zlim(-0.01, com_height + 0.2)
    ax3d.set_title('3D LIPM Animation')
    ax3d.view_init(25, -140)

    com_ball, = ax3d.plot([], [], [], 'ro', markersize=12)
    com_trail, = ax3d.plot([], [], [], 'g-', linewidth=1, alpha=0.7)
    left_leg_line, = ax3d.plot([], [], [], 'b-', linewidth=2.5)
    right_leg_line, = ax3d.plot([], [], [], 'r-', linewidth=2.5)
    left_foot_ball, = ax3d.plot([], [], [], 'bo', markersize=7)
    right_foot_ball, = ax3d.plot([], [], [], 'ro', markersize=7)
    # 발자국 마커 (정적)
    for (fx, fy, leg) in footsteps:
        c = 'blue' if leg == 0 else 'red'
        ax3d.plot([fx], [fy], [0], 's', color=c, markersize=4, alpha=0.5)

    # ---- [0,1] CoM velocity ----
    ax_vel = fig.add_subplot(spec[0, 1])
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('Velocity (m/s)')
    ax_vel.set_title('CoM Velocity vs Desired')
    ax_vel.grid(True, alpha=0.3)
    vel_x_line, = ax_vel.plot([], [], 'k-', label='vx actual')
    vel_y_line, = ax_vel.plot([], [], 'purple', label='vy actual')
    dvel_x_line, = ax_vel.plot([], [], 'k--', label='vx desired')
    dvel_y_line, = ax_vel.plot([], [], 'purple', linestyle='--', label='vy desired')
    ax_vel.set_xlim(0, t[-1])
    vel_min = min(com_vel[:, 0].min(), com_vel[:, 1].min(), 0) - 0.1
    vel_max = max(com_vel[:, 0].max(), com_vel[:, 1].max(), des_vel[:, 0].max()) + 0.1
    ax_vel.set_ylim(vel_min, vel_max)
    ax_vel.legend(fontsize=7, loc='upper left')

    # ---- [0,2] Foot Z ----
    ax_fz = fig.add_subplot(spec[0, 2])
    ax_fz.set_xlabel('Time (s)')
    ax_fz.set_ylabel('Z (m)')
    ax_fz.set_title('Foot Height (Z)')
    ax_fz.grid(True, alpha=0.3)
    fz_left_line, = ax_fz.plot([], [], 'b-', linewidth=1.5, label='Left foot')
    fz_right_line, = ax_fz.plot([], [], 'r-', linewidth=1.5, label='Right foot')
    ax_fz.axhline(y=foot_height, color='gray', linestyle=':', alpha=0.5, label=f'H={foot_height}m')
    ax_fz.set_xlim(0, t[-1])
    ax_fz.set_ylim(-0.02, foot_height + 0.05)
    ax_fz.legend(fontsize=7)

    # ---- [1,0] 2D XY top view ----
    ax_xy = fig.add_subplot(spec[1, 0])
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('Top View (XY)')
    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.3)
    com_xy_trail, = ax_xy.plot([], [], 'g-', linewidth=1, alpha=0.7)
    com_xy_ball, = ax_xy.plot([], [], 'ro', markersize=8)
    lf_xy, = ax_xy.plot([], [], 'bo', markersize=6)
    rf_xy, = ax_xy.plot([], [], 'ro', markersize=6)
    for (fx, fy, leg) in footsteps:
        c = 'blue' if leg == 0 else 'red'
        ax_xy.plot(fx, fy, 's', color=c, markersize=4, alpha=0.5)
    xy_text = ax_xy.text(0.02, 0.95, '', transform=ax_xy.transAxes, fontsize=8)

    # ---- [1,1] Step length / width ----
    ax_step = fig.add_subplot(spec[1, 1])
    ax_step.set_xlabel('Step #')
    ax_step.set_ylabel('Distance (m)')
    ax_step.set_title('Step Length / Width')
    ax_step.grid(True, alpha=0.3)
    if len(step_lengths) > 0:
        step_idx = np.arange(1, len(step_lengths) + 1)
        ax_step.bar(step_idx - 0.15, step_lengths, 0.3, color='gray', alpha=0.7, label='Step length')
        ax_step.bar(step_idx + 0.15, step_widths, 0.3, color='cyan', alpha=0.7, label='Step width')
        ax_step.legend(fontsize=7)

    # ---- [1,2] ZMP desired force ----
    ax_force = fig.add_subplot(spec[1, 2])
    ax_force.set_xlabel('Time (s)')
    ax_force.set_ylabel('Force (N)')
    ax_force.set_title('ZMP Desired Force (total)')
    ax_force.grid(True, alpha=0.3)
    total_f = fr_left + fr_right
    force_fx_line, = ax_force.plot([], [], 'r-', linewidth=1, label='Fx')
    force_fy_line, = ax_force.plot([], [], 'b-', linewidth=1, label='Fy')
    force_fz_line, = ax_force.plot([], [], 'k-', linewidth=1, alpha=0.5, label='Fz')
    ax_force.set_xlim(0, t[-1])
    f_min = min(total_f[:, 0].min(), total_f[:, 1].min()) - 20
    f_max = max(total_f[:, 2].max(), 50) + 20
    ax_force.set_ylim(f_min, f_max)
    ax_force.legend(fontsize=7)

    fig.suptitle(
        f'Full Logic Verification  (T_sw={t_swing:.3f}s, T_st={t_stance:.3f}s, '
        f'H={foot_height}m, z_c={com_height}m, mass={robot_mass}kg)',
        fontsize=12
    )

    # ============================================
    # Animation update
    # ============================================
    def init():
        return []

    def update(i):
        # 3D LIPM
        cx, cy, cz = com[i, 0], com[i, 1], com_height
        lfx, lfy, lfz = lf[i]
        rfx, rfy, rfz = rf[i]

        com_ball.set_data_3d([cx], [cy], [cz])
        com_trail.set_data_3d(com[:i+1, 0], com[:i+1, 1], np.zeros(i+1))
        left_leg_line.set_data_3d([cx, lfx], [cy, lfy], [cz, lfz])
        right_leg_line.set_data_3d([cx, rfx], [cy, rfy], [cz, rfz])
        left_foot_ball.set_data_3d([lfx], [lfy], [lfz])
        right_foot_ball.set_data_3d([rfx], [rfy], [rfz])

        # 3D view auto-scroll
        margin = 1.5
        ax3d.set_xlim(cx - margin, cx + margin)
        ax3d.set_ylim(cy - 0.8, cy + 0.8)

        # CoM velocity
        vel_x_line.set_data(t[:i+1], com_vel[:i+1, 0])
        vel_y_line.set_data(t[:i+1], com_vel[:i+1, 1])
        dvel_x_line.set_data(t[:i+1], des_vel[:i+1, 0])
        dvel_y_line.set_data(t[:i+1], des_vel[:i+1, 1])

        # Foot Z
        fz_left_line.set_data(t[:i+1], lf[:i+1, 2])
        fz_right_line.set_data(t[:i+1], rf[:i+1, 2])

        # 2D XY
        com_xy_trail.set_data(com[:i+1, 0], com[:i+1, 1])
        com_xy_ball.set_data([cx], [cy])
        lf_xy.set_data([lfx], [lfy])
        rf_xy.set_data([rfx], [rfy])
        ax_xy.set_xlim(cx - 1.0, cx + 1.0)
        ax_xy.set_ylim(cy - 0.5, cy + 0.5)
        xy_text.set_text(f'CoM=({cx:.2f}, {cy:.2f})')

        # ZMP force
        force_fx_line.set_data(t[:i+1], total_f[:i+1, 0])
        force_fy_line.set_data(t[:i+1], total_f[:i+1, 1])
        force_fz_line.set_data(t[:i+1], total_f[:i+1, 2])

        return []

    # interval: log_interval=10, dt=0.002 → 0.02s per frame → 20ms
    anim = FuncAnimation(fig, update, frames=range(1, data_len),
                         init_func=init, interval=20, blit=False, repeat=False)
    return fig, anim


# ============================================
# Main
# ============================================
if __name__ == "__main__":
    print("[INFO] Running full logic verification simulation...")
    log, footsteps, step_lengths, step_widths, step_times = run_simulation()
    print(f"[INFO] Simulation done. {len(log['time'])} frames logged, {len(footsteps)} footsteps.")

    fig, anim = create_animation(log, footsteps, step_lengths, step_widths, step_times)

    # 저장 옵션 (커맨드라인 인자로 --save 전달 시)
    if '--save' in sys.argv:
        filepath = os.path.join(os.path.dirname(__file__), 'plot_full_verification.mp4')
        print(f"[INFO] Saving animation to {filepath} ...")
        anim.save(filepath, fps=50, extra_args=['-vcodec', 'libx264'])
        print("[INFO] Saved.")
    else:
        plt.show()

    print("[INFO] Done.")
