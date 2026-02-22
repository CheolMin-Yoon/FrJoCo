"""G1 Dynamics-based Whole-Body Controller (Torque Control)

Layer1(궤적) → Layer2(피드백) → Layer3(Dynamics WBC) → Torque Actuator → mj_step()

Unitree G1 29DOF 모델 사용 (motor actuator, torque control).

계층 구조:
  Layer 1: DCM 기반 궤적 생성 (CoM, Foot)
  Layer 2: DCM/ZMP 피드백 제어
  Layer 3: Dynamics WBC (Task Space Impedance → Torque)
  MuJoCo: motor actuator + mj_step()

Usage:
  conda activate mujoco_env
  python tutorial/g1_new/SRBD/test/g1_wbc_dynamics.py
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Layer1_DCM import TrajectoryOptimization
from Layer2_DCM import SimplifiedModelControl
from Layer3_dynamics_wbc import DynamicsWBC

# Unitree G1 29DOF Scene XML (motor actuator + floor)
# Get workspace root (4 levels up from test folder: test -> SRBD -> g1_new -> tutorial -> workspace)
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
xml_path = os.path.join(
    workspace_root,
    'reference', 'unitree_mujoco', 'unitree_robots', 'g1', 'scene_29dof.xml'
)


# ========================================================================= #
# 파라미터
# ========================================================================= #
dt = 0.002  # Control timestep
zc = 0.75   # CoM height
n_steps = 4
step_length = 0.2
step_width = 0.1
step_time = 0.8
dsp_time = 0.2
step_height = 0.05

# DCM gains
K_DCM = 1.0
KI_DCM = 0.1
K_ZMP = 0.5
K_COM = 0.3

# Arm swing
ARM_SWING_AMP = 0.3


# ========================================================================= #
# Layer 1: 궤적 생성
# ========================================================================= #
print("=" * 60)
print("  Layer 1: 궤적 생성 (Dynamics)")
print("=" * 60)

planner = TrajectoryOptimization(
    z_c=zc, step_time=step_time, dsp_time=dsp_time,
    step_height=step_height, dt=dt
)

# Temporary initial values for trajectory generation
# Will be updated after model initialization
init_com_temp = np.array([0.0, 0.0, zc])
init_lf_temp = np.array([0.0, step_width, 0.0])
init_rf_temp = np.array([0.0, -step_width, 0.0])

footsteps, dcm_ref, dcm_vel_ref, com_ref_temp, com_vel_ref, lf_traj_temp, rf_traj_temp = \
    planner.compute_all_trajectories(
        n_steps=n_steps, step_length=step_length,
        step_width=step_width, init_com=init_com_temp,
        init_lf=init_lf_temp, init_rf=init_rf_temp
    )

sim_length = len(com_ref_temp)
samples_per_step = planner.samples_per_step
print(f"  궤적 길이: {sim_length} steps ({sim_length * dt:.1f}s)")


# ========================================================================= #
# Support Phase 판별 함수
# ========================================================================= #
def get_support_phase(traj_idx: int) -> str:
    """현재 궤적 인덱스에서 support phase를 판별."""
    step_idx = traj_idx // samples_per_step
    step_idx = min(step_idx, n_steps - 1)
    local_k = traj_idx - step_idx * samples_per_step
    local_t = local_k * dt
    if local_t < dsp_time:
        return 'dsp'
    elif step_idx % 2 == 0:
        return 'left_support'
    else:
        return 'right_support'


# ========================================================================= #
# Layer 2: 피드백 제어기
# ========================================================================= #
controller = SimplifiedModelControl(
    z_c=zc, k_dcm=K_DCM, ki_dcm=KI_DCM,
    k_zmp=K_ZMP, k_com=K_COM, dt=dt
)


# ========================================================================= #
# 팔 스윙
# ========================================================================= #
def generate_arm_swing_angles(sim_length, step_time, dt, amp=ARM_SWING_AMP):
    left_traj = np.zeros(sim_length)
    right_traj = np.zeros(sim_length)
    gait_period = 2 * step_time
    for k in range(sim_length):
        t = k * dt
        phase = 2 * np.pi * t / gait_period
        swing = amp * np.cos(phase)
        target_left = swing
        target_right = -swing
        if t < gait_period:
            envelope = 0.5 * (1 - np.cos(np.pi * t / gait_period))
            target_left *= envelope
            target_right *= envelope
        left_traj[k] = target_left
        right_traj[k] = target_right
    return left_traj, right_traj

left_shoulder_swing, right_shoulder_swing = generate_arm_swing_angles(
    sim_length, step_time, dt
)


# ========================================================================= #
# MuJoCo Dynamics + Layer3 Dynamics WBC + Torque Actuator
# ========================================================================= #
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Dynamics WBC (Layer3 Torque Control)")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. MuJoCo 모델/데이터
    # ------------------------------------------------------------------ #
    # Load model directly from file
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Check if foot sites exist, if not we need to add them
    try:
        left_site = model.site("left_foot")
        right_site = model.site("right_foot")
        print(f"  Foot sites found: left_foot (id={left_site.id}), right_foot (id={right_site.id})")
    except:
        print("  Warning: Foot sites not found in model!")
        print("  Using ankle bodies as foot reference instead...")
        # We'll use body positions instead of sites
        # This is a fallback - ideally the model should have sites

    # Reset to standing pose
    data.qpos[2] = 0.75  # Base height
    data.qpos[3:7] = [1, 0, 0, 0]  # Quaternion
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------------ #
    # 2. Layer 3: Dynamics WBC 초기화
    # ------------------------------------------------------------------ #
    wbc = DynamicsWBC(model, data)
    
    # Set desired posture to current pose
    wbc.q_des = data.qpos.copy()
    
    com_init = data.subtree_com[0].copy()
    
    # Try to get foot positions from sites, fallback to body positions
    try:
        left_foot_init = data.site("left_foot").xpos.copy()
        right_foot_init = data.site("right_foot").xpos.copy()
        print("  Using foot sites for tracking")
    except:
        # Use ankle body positions as fallback
        left_foot_init = data.body("left_ankle_roll_link").xpos.copy()
        right_foot_init = data.body("right_ankle_roll_link").xpos.copy()
        print("  Using ankle bodies for tracking")
        # Update WBC to use bodies instead of sites
        wbc.use_bodies = True
        wbc.left_foot_body = model.body("left_ankle_roll_link").id
        wbc.right_foot_body = model.body("right_ankle_roll_link").id

    offset_x = com_init[0]
    offset_y = com_init[1]

    # Regenerate trajectories with actual initial positions
    footsteps, dcm_ref, dcm_vel_ref, com_ref, com_vel_ref, lf_traj, rf_traj = \
        planner.compute_all_trajectories(
            n_steps=n_steps, step_length=step_length,
            step_width=step_width, init_com=com_init,
            init_lf=left_foot_init, init_rf=right_foot_init
        )
    
    com_traj = com_ref  # Already 3D from planner

    n_physics_steps = max(1, int(round(dt / model.opt.timestep)))

    print(f"\n초기 상태:")
    print(f"  CoM:        {com_init}")
    print(f"  Left foot:  {left_foot_init}")
    print(f"  Right foot: {right_foot_init}")
    print(f"  Control dt: {dt}s, Physics dt: {model.opt.timestep}s")
    print(f"  Physics steps per control: {n_physics_steps}")
    print(f"  Model: {xml_path}")

    # ------------------------------------------------------------------ #
    # 3. 사전 안정화
    # ------------------------------------------------------------------ #
    print("  사전 안정화 중...")
    for _ in range(1000):
        tau = wbc.solve_qp(com_init, left_foot_init, right_foot_init)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        
        # Check for instability
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            print("  Warning: Simulation became unstable during stabilization!")
            print("  Resetting to initial pose...")
            data.qpos[2] = 0.75
            data.qpos[3:7] = [1, 0, 0, 0]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            break

    com_init = data.subtree_com[0].copy()
    
    # Get foot positions using the same method as WBC
    if hasattr(wbc, 'use_bodies') and wbc.use_bodies:
        left_foot_init = data.body(wbc.left_foot_body).xpos.copy()
        right_foot_init = data.body(wbc.right_foot_body).xpos.copy()
    else:
        left_foot_init = data.site("left_foot").xpos.copy()
        right_foot_init = data.site("right_foot").xpos.copy()
    
    offset_x = com_init[0]
    offset_y = com_init[1]

    print(f"  안정화 후 CoM: {com_init}")
    print(f"  안정화 후 CoM Z: {com_init[2]:.4f}m")

    # ------------------------------------------------------------------ #
    # 4. 시뮬레이션 루프
    # ------------------------------------------------------------------ #
    t_init = 2.0
    sim_time = 0.0
    traj_idx = 0

    com_offset = np.zeros(2)
    controller.dcm_error_sum = np.zeros(2)

    print("\n" + "=" * 60)
    print("  G1 WBC Dynamics (Layer3 Torque Control)")
    print("  Press ESC to exit")
    print("=" * 60)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        while viewer.is_running() and traj_idx < sim_length:
            step_start = time.time()
            sim_time += dt

            if sim_time < t_init:
                # Stand still
                tau = wbc.solve_qp(com_init, left_foot_init, right_foot_init)
            else:
                t_walk = sim_time - t_init
                traj_idx = min(int(t_walk / dt), sim_length - 1)

                # --- Layer 2: 피드백 제어 ---
                meas_com_pos = data.subtree_com[0].copy()
                meas_com_vel = data.qvel[:3].copy()

                # Get current foot positions for ZMP calculation
                if hasattr(wbc, 'use_bodies') and wbc.use_bodies:
                    lf_pos = data.body(wbc.left_foot_body).xpos
                    rf_pos = data.body(wbc.right_foot_body).xpos
                else:
                    lf_pos = data.site("left_foot").xpos
                    rf_pos = data.site("right_foot").xpos
                
                phase = get_support_phase(traj_idx)
                if phase == 'left_support':
                    meas_zmp = np.array([lf_pos[0], lf_pos[1]])
                elif phase == 'right_support':
                    meas_zmp = np.array([rf_pos[0], rf_pos[1]])
                else:
                    meas_zmp = np.array([
                        0.5 * (lf_pos[0] + rf_pos[0]),
                        0.5 * (lf_pos[1] + rf_pos[1]),
                    ])

                desired_com_vel, calc_zmp, curr_dcm = controller.control_step(
                    meas_com_pos=meas_com_pos,
                    meas_com_vel=meas_com_vel,
                    meas_zmp=meas_zmp,
                    ref_dcm=dcm_ref[traj_idx],
                    ref_dcm_vel=dcm_vel_ref[traj_idx],
                    ref_com_pos=com_ref[traj_idx],
                    ref_com_vel=com_vel_ref[traj_idx],
                )

                # --- Layer 3: Dynamics WBC ---
                com_offset += desired_com_vel[:2] * dt

                target_com = np.array([
                    com_traj[traj_idx, 0] + com_offset[0],
                    com_traj[traj_idx, 1] + com_offset[1],
                    com_traj[traj_idx, 2]
                ])

                left_target = lf_traj[traj_idx].copy()
                right_target = rf_traj[traj_idx].copy()

                # Compute torque
                tau = wbc.solve_qp(target_com, left_target, right_target)

                if int(t_walk * 10) % 20 == 0:
                    actual_com = data.subtree_com[0]
                    com_err = np.linalg.norm(target_com - actual_com)
                    print(
                        f"t={t_walk:.2f}s | idx={traj_idx} | "
                        f"CoM err={com_err:.4f}m | "
                        f"CoM=({actual_com[0]:.3f}, {actual_com[1]:.3f}, {actual_com[2]:.3f})"
                    )

            # Apply torque
            data.ctrl[:] = tau

            for _ in range(n_physics_steps):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            # 시각화
            viewer.user_scn.ngeom = 0
            if sim_time >= t_init and traj_idx > 0:
                vis_start = max(0, traj_idx - 100)
                vis_end = min(sim_length - 1, traj_idx + 200)
                for i in range(vis_start, vis_end - 1):
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                        break
                    p1 = np.array([
                        com_traj[i, 0],
                        com_traj[i, 1], 0.01
                    ])
                    p2 = np.array([
                        com_traj[i + 1, 0],
                        com_traj[i + 1, 1], 0.01
                    ])
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                        width=0.003, from_=p1, to=p2
                    )
                    viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [0, 1, 0, 0.5]
                    viewer.user_scn.ngeom += 1

            viewer.sync()
            
            # Rate limiting
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        print("\n완료!")
