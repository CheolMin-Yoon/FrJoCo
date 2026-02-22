"""
Layer1 Gait Phase 시각화

- Phase (0~1) 시계열
- Swing leg index (0=left, 1=right)
- Contact state (left/right)
- Cycloid X 진행률
- Bezier Z 프로파일 (단일 swing 주기)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from Layer1 import GaitGenerator
from config import dt, t_swing, t_stance, foot_height


def main():
    layer1 = GaitGenerator(T_s=t_swing, T_st=t_stance)

    SIM_TIME = 4.0
    n_steps = int(SIM_TIME / dt)

    log_time = []
    log_phase = []
    log_swing_leg = []
    log_contact_L = []
    log_contact_R = []

    sim_time = 0.0
    for _ in range(n_steps):
        sim_time += dt
        phase, contact_state, swing_leg_idx = layer1.state_machine(dt, sim_time)

        log_time.append(sim_time)
        log_phase.append(phase)
        log_swing_leg.append(swing_leg_idx)
        log_contact_L.append(contact_state[0])
        log_contact_R.append(contact_state[1])

    t = np.array(log_time)
    phase = np.array(log_phase)
    swing = np.array(log_swing_leg)
    cL = np.array(log_contact_L)
    cR = np.array(log_contact_R)

    # Bezier Z 프로파일 (단일 주기, phase 0→1)
    p_range = np.linspace(0, 1, 200)
    z_bezier = np.array([layer1.Bezier_Curve_interpolation(p) for p in p_range])

    # Cycloid XY 진행률 (단일 주기)
    theta_range = 2 * np.pi * p_range
    cycloid = (theta_range - np.sin(theta_range)) / (2 * np.pi)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # --- 1. Phase ---
    ax = axes[0, 0]
    ax.plot(t, phase, 'g-', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Phase (0~1)')
    ax.set_title('Gait Phase')
    ax.grid(True, alpha=0.3)

    # --- 2. Swing leg ---
    ax = axes[0, 1]
    ax.step(t, swing, 'm-', linewidth=1, where='post')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Swing leg')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Left (0)', 'Right (1)'])
    ax.set_title('Swing Leg Index')
    ax.grid(True, alpha=0.3)

    # --- 3. Contact state ---
    ax = axes[1, 0]
    ax.fill_between(t, 0, cL, alpha=0.4, color='blue', step='post', label='Left contact')
    ax.fill_between(t, 0, cR, alpha=0.4, color='red', step='post', label='Right contact')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Contact (0/1)')
    ax.set_title('Contact State')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 4. Bezier Z 프로파일 ---
    ax = axes[1, 1]
    ax.plot(p_range, z_bezier, 'k-', linewidth=2)
    ax.axhline(y=foot_height, color='gray', linestyle=':', alpha=0.5, label=f'H={foot_height}m')
    ax.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='rise/fall split (0.2)')
    ax.set_xlabel('Phase (0~1)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Bezier Z Profile (single swing)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 5. Cycloid XY 진행률 ---
    ax = axes[2, 0]
    ax.plot(p_range, cycloid, 'b-', linewidth=2)
    ax.plot(p_range, p_range, 'k--', linewidth=1, alpha=0.5, label='linear')
    ax.set_xlabel('Phase (0~1)')
    ax.set_ylabel('XY progress (0~1)')
    ax.set_title('Cycloid XY Progress')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 6. Phase vs Swing leg (combined) ---
    ax = axes[2, 1]
    # 왼발 swing 구간 하이라이트
    left_swing = swing == 0
    right_swing = swing == 1
    ax.fill_between(t, 0, phase, where=left_swing, alpha=0.3, color='blue', label='Left swing')
    ax.fill_between(t, 0, phase, where=right_swing, alpha=0.3, color='red', label='Right swing')
    ax.plot(t, phase, 'k-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Phase')
    ax.set_title('Phase colored by Swing Leg')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    cycle = t_swing + t_stance
    plt.suptitle(
        f'Gait Phase Analysis (T_swing={t_swing:.3f}s, T_stance={t_stance:.3f}s, cycle={cycle:.3f}s, freq={1/cycle:.1f}Hz)',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig('plot_leg_phase.png', dpi=150)
    print("[INFO] Saved plot_leg_phase.png")
    plt.show()


if __name__ == "__main__":
    main()
