"""
g1_wbc_dynamics_qp.py — DCM 플래너 + DCM PI 피드백 + QP WBC 토크 제어

Layer1 (TrajectoryOptimization) → 오프라인 궤적 (DCM, CoM, 발)
Layer2 (SimplifiedModelControl)  → DCM PI 피드백 → desired CoM velocity
Layer3 (TaskSpaceWBC)            → QP → feedforward 토크

최종: tau = tau_ff + Kp*(q0 - q) + Kd*(0 - dq)
"""

import numpy as np
import mujoco
import mujoco.viewer
import time

from Layer1 import TrajectoryOptimization
from Layer2 import SimplifiedModelControl
from Layer3 import TaskSpaceWBC

from config import (
    N_STEPS, STEP_LENGTH, STEP_WIDTH, STEP_HEIGHT, DSP_TIME,
    K_DCM, KI_DCM, K_ZMP, K_COM,
    DT, STEP_TIME, INIT_DSP_EXTRA, COM_SHIFT_TIME,
    GRAVITY, ROBOT_MASS, COM_HEIGHT,
    LEG_KP, LEG_KD, ANKLE_KP, ANKLE_KD, ARM_KP, ARM_KD,
)
from zmp_sensor import compute_zmp_from_ft_sensors

xml_path = '../../model/g1/scene_23dof.xml'
dt = DT

N_SHIFT = int(COM_SHIFT_TIME / DT) if COM_SHIFT_TIME > 0 else 0


# ========================================================================= #
# Support Phase 판별
# ========================================================================= #
def get_support_phase(traj_idx, samples_per_step):
    if traj_idx < N_SHIFT:
        return 'dsp', -1

    walk_idx = traj_idx - N_SHIFT
    first_step_samples = samples_per_step + int(INIT_DSP_EXTRA / dt)
    if walk_idx < first_step_samples:
        step_idx = 0
        local_t = walk_idx * dt
        first_dsp = DSP_TIME + INIT_DSP_EXTRA
    else:
        remaining = walk_idx - first_step_samples
        step_idx = 1 + min(remaining // samples_per_step, N_STEPS - 2)
        local_t = (remaining - (step_idx - 1) * samples_per_step) * dt
        first_dsp = DSP_TIME

    if local_t < first_dsp:
        return 'dsp', step_idx
    elif step_idx % 2 == 0:
        return 'left_support', step_idx
    else:
        return 'right_support', step_idx


def compute_desired_contact_force(phase, com_pos, lf_pos, rf_pos):
    """support phase에 따라 desired contact force 계산."""
    mg = ROBOT_MASS * GRAVITY
    omega2 = GRAVITY / COM_HEIGHT

    if phase == 'dsp':
        # 양발 지지: 50/50 분배 + ZMP 기반 수평력
        target_zmp = (lf_pos[:2] + rf_pos[:2]) / 2.0
        fx = -ROBOT_MASS * omega2 * (com_pos[0] - target_zmp[0])
        fy = -ROBOT_MASS * omega2 * (com_pos[1] - target_zmp[1])
        fr_left = np.array([fx / 2, fy / 2, mg / 2])
        fr_right = np.array([fx / 2, fy / 2, mg / 2])
    elif phase == 'left_support':
        target_zmp = lf_pos[:2]
        fx = -ROBOT_MASS * omega2 * (com_pos[0] - target_zmp[0])
        fy = -ROBOT_MASS * omega2 * (com_pos[1] - target_zmp[1])
        fr_left = np.array([fx, fy, mg])
        fr_right = np.zeros(3)
    else:  # right_support
        target_zmp = rf_pos[:2]
        fx = -ROBOT_MASS * omega2 * (com_pos[0] - target_zmp[0])
        fy = -ROBOT_MASS * omega2 * (com_pos[1] - target_zmp[1])
        fr_left = np.zeros(3)
        fr_right = np.array([fx, fy, mg])

    return fr_left, fr_right




# ========================================================================= #
# 메인
# ========================================================================= #
if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_full = os.path.normpath(os.path.join(script_dir, xml_path))

    model = mujoco.MjModel.from_xml_path(xml_full)
    data = mujoco.MjData(model)
    model.opt.timestep = dt

    # ── 관절 매핑 ──
    leg_names = [
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
        "left_knee", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
        "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "waist_yaw", "waist_roll", "waist_pitch",
    ]
    arm_names = [
        "left_shoulder_pitch", "left_shoulder_roll",
        "left_shoulder_yaw", "left_elbow",
        "right_shoulder_pitch", "right_shoulder_roll",
        "right_shoulder_yaw", "right_elbow",
    ]

    joint_ids = np.array([model.joint(n + "_joint").id for n in leg_names])
    actuator_ids = np.array([model.actuator(n).id for n in leg_names])
    arm_joint_ids = np.array([model.joint(n + "_joint").id for n in arm_names])
    arm_actuator_ids = np.array([model.actuator(n).id for n in arm_names])

    qpos_ids = np.array([model.jnt_qposadr[j] for j in joint_ids])
    dof_ids = np.array([model.jnt_dofadr[j] for j in joint_ids])
    arm_qpos_ids = np.array([model.jnt_qposadr[j] for j in arm_joint_ids])
    arm_dof_ids = np.array([model.jnt_dofadr[j] for j in arm_joint_ids])

    nu = len(actuator_ids)

    # ── 초기 자세 ──
    key_id = model.key("knees_bent").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    q0_legs = data.qpos[qpos_ids].copy()
    q0_arms = data.qpos[arm_qpos_ids].copy()

    # ── Site/Body ID ──
    lf_site = model.site("left_foot").id
    rf_site = model.site("right_foot").id
    torso_body = model.body("torso_link").id

    # ── PD 게인 ──
    Kp = np.full(nu, LEG_KP)
    Kd = np.full(nu, LEG_KD)
    for i, n in enumerate(leg_names):
        if "ankle" in n:
            Kp[i], Kd[i] = ANKLE_KP, ANKLE_KD
        if n in ("waist_roll", "waist_pitch"):
            Kp[i], Kd[i] = 0.0, 0.0

    Kp_arm = np.full(len(arm_names), ARM_KP)
    Kd_arm = np.full(len(arm_names), ARM_KD)

    # ── 초기 위치 캡처 ──
    com_init = data.subtree_com[0].copy()
    lf_init = data.site(lf_site).xpos.copy()
    rf_init = data.site(rf_site).xpos.copy()
    z_c = com_init[2]

    print(f"CoM: {com_init}, LF: {lf_init}, RF: {rf_init}")
    torso_com_z_offset = 0.0  # 보행 시작 시 업데이트됨

    # ── Layer1/Layer2/Layer3: 보행 시작 후 생성 (post-stabilization 값 사용) ──
    planner = None
    controller = None
    footsteps = None
    dcm_ref = dcm_vel_ref = com_traj = com_vel_ref = lf_traj = rf_traj = None
    sim_length = 0
    samples_per_step = 0

    # ── Layer3: QP WBC ──
    wbc = TaskSpaceWBC(model.nv, nu, model, data, actuator_dof_ids=dof_ids)

    # ── 궤적 히스토리 (시각화) ──
    TRAIL_LEN = 500
    com_trail = np.zeros((TRAIL_LEN, 3))
    trail_idx_vis = 0
    trail_filled = False

    # ── 메인 루프 ──
    sim_time = 0.0
    step_count = 0
    STABILIZE_TIME = 2.0  # 초기 안정화 시간 (s)
    walking_started = False

    print(f"\n[대기] {STABILIZE_TIME}s 안정화 후 보행 시작")
    print(f"컨트롤러: DCM PI + QP WBC (토크 제어)")

    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE

        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        while viewer.is_running():
            step_start = time.time()

            q_curr = data.qpos.copy()
            dq_curr = data.qvel.copy()

            lf_pos = data.site(lf_site).xpos.copy()
            rf_pos = data.site(rf_site).xpos.copy()

            # ============================================
            # 초기 안정화 (2초) — gravity comp + PD
            # ============================================
            if data.time < STABILIZE_TIME:
                tau_grav = data.qfrc_bias[dof_ids].copy()
                # 더미 관절 bias 제거
                for i, n in enumerate(leg_names):
                    if n in ("waist_roll", "waist_pitch"):
                        tau_grav[i] = 0.0

                tau_fb = Kp * (q0_legs - q_curr[qpos_ids]) - Kd * dq_curr[dof_ids]
                tau_cmd = tau_grav + tau_fb

                tau_arms = (Kp_arm * (q0_arms - q_curr[arm_qpos_ids])
                            - Kd_arm * dq_curr[arm_dof_ids]
                            + data.qfrc_bias[arm_dof_ids])

                np.clip(tau_cmd, model.actuator_ctrlrange[actuator_ids, 0],
                        model.actuator_ctrlrange[actuator_ids, 1], out=tau_cmd)
                np.clip(tau_arms, model.actuator_ctrlrange[arm_actuator_ids, 0],
                        model.actuator_ctrlrange[arm_actuator_ids, 1], out=tau_arms)

                data.ctrl[actuator_ids] = tau_cmd
                data.ctrl[arm_actuator_ids] = tau_arms

                step_count += 1
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                viewer.sync()

                elapsed = time.time() - step_start
                if dt - elapsed > 0:
                    time.sleep(dt - elapsed)
                continue

            # ── 보행 시작 시점: post-stabilization 값으로 궤적 생성 ──
            if not walking_started:
                walking_started = True
                walk_start_time = data.time

                # post-stabilization 실제 값 캡처
                walk_com = data.subtree_com[0].copy()
                walk_lf = data.site(lf_site).xpos.copy()
                walk_rf = data.site(rf_site).xpos.copy()
                walk_torso = data.body(torso_body).xpos.copy()
                torso_com_z_offset = walk_torso[2] - walk_com[2]
                z_c = walk_com[2]

                # Layer1: 궤적 생성
                planner = TrajectoryOptimization(
                    z_c=z_c, step_time=STEP_TIME, dsp_time=DSP_TIME,
                    step_height=STEP_HEIGHT, dt=dt,
                )
                footsteps, dcm_ref, dcm_vel_ref, com_traj, com_vel_ref, lf_traj, rf_traj = \
                    planner.compute_all_trajectories(
                        n_steps=N_STEPS, step_length=STEP_LENGTH,
                        step_width=STEP_WIDTH,
                        init_com=walk_com, init_lf=walk_lf, init_rf=walk_rf,
                    )
                sim_length = len(com_traj)
                samples_per_step = planner.samples_per_step

                # Layer2: 컨트롤러
                controller = SimplifiedModelControl(
                    z_c=z_c, k_dcm=K_DCM, ki_dcm=KI_DCM,
                    k_zmp=K_ZMP, k_com=K_COM, dt=dt,
                )

                print(f"[보행 시작] t={data.time:.2f}s")
                print(f"  CoM:   ({walk_com[0]:.4f}, {walk_com[1]:.4f}, {walk_com[2]:.4f})")
                print(f"  Torso: ({walk_torso[0]:.4f}, {walk_torso[1]:.4f}, {walk_torso[2]:.4f})")
                print(f"  LF:    ({walk_lf[0]:.4f}, {walk_lf[1]:.4f}, {walk_lf[2]:.4f})")
                print(f"  RF:    ({walk_rf[0]:.4f}, {walk_rf[1]:.4f}, {walk_rf[2]:.4f})")
                print(f"  z_c={z_c:.4f}, torso_com_z_offset={torso_com_z_offset:.4f}")
                print(f"  궤적: {sim_length} samples ({sim_length * dt:.1f}s)")

            sim_time = data.time - walk_start_time
            traj_idx = int(sim_time / dt)
            if traj_idx >= sim_length:
                break

            # ── Layer2: DCM PI 피드백 ──
            meas_com_pos = data.subtree_com[0].copy()
            mujoco.mj_subtreeVel(model, data)
            meas_com_vel = data.subtree_linvel[0].copy()
            meas_zmp = compute_zmp_from_ft_sensors(model, data)

            desired_com_vel, calc_zmp, curr_dcm = controller.control_step(
                meas_com_pos=meas_com_pos,
                meas_com_vel=meas_com_vel,
                meas_zmp=meas_zmp,
                ref_dcm=dcm_ref[traj_idx],
                ref_dcm_vel=dcm_vel_ref[traj_idx],
                ref_com_pos=com_traj[traj_idx, :2],
                ref_com_vel=com_vel_ref[traj_idx],
            )

            # ── Torso 목표 (Layer2 보정 반영, 실제 CoM 높이 기반) ──
            target_com = com_traj[traj_idx].copy()
            target_com[0] += desired_com_vel[0] * dt
            target_com[1] += desired_com_vel[1] * dt
            ref_torso = np.array([target_com[0], target_com[1], target_com[2] + torso_com_z_offset])

            target_lf = lf_traj[traj_idx].copy()
            target_rf = rf_traj[traj_idx].copy()

            # ── Support phase → contact force ──
            phase, step_idx = get_support_phase(traj_idx, samples_per_step)
            fr_left, fr_right = compute_desired_contact_force(
                phase, meas_com_pos, lf_pos, rf_pos)

            # ── Swing foot 판별 ──
            if phase == 'dsp':
                swing_site_id = -1
                ref_swing_pos = None
            elif phase == 'left_support':
                # 오른발 스윙
                swing_site_id = rf_site
                ref_swing_pos = target_rf
            else:
                # 왼발 스윙
                swing_site_id = lf_site
                ref_swing_pos = target_lf

            # ── Layer3: QP WBC → feedforward 토크 ──
            tau_ff = wbc.compute_torque(
                fr_left=fr_left,
                fr_right=fr_right,
                left_foot_site_id=lf_site,
                right_foot_site_id=rf_site,
                torso_body_id=torso_body,
                ref_torso_pos=ref_torso,
                swing_foot_site_id=swing_site_id,
                ref_swing_pos=ref_swing_pos,
                dt=dt,
            )

            # ── PD 피드백 + feedforward + gravity comp ──
           #tau_grav = data.qfrc_bias[dof_ids].copy()
           # for i, n in enumerate(leg_names):
           #     if n in ("waist_roll", "waist_pitch"):
           #         tau_grav[i] = 0.0

            tau_fb = Kp * (q0_legs - q_curr[qpos_ids]) - Kd * dq_curr[dof_ids]
            tau_cmd = tau_ff + tau_fb

            # 팔: PD
            tau_arms = (Kp_arm * (q0_arms - q_curr[arm_qpos_ids])
                        - Kd_arm * dq_curr[arm_dof_ids]
                        + data.qfrc_bias[arm_dof_ids])

            # 토크 클리핑
            np.clip(tau_cmd, model.actuator_ctrlrange[actuator_ids, 0],
                    model.actuator_ctrlrange[actuator_ids, 1], out=tau_cmd)
            np.clip(tau_arms, model.actuator_ctrlrange[arm_actuator_ids, 0],
                    model.actuator_ctrlrange[arm_actuator_ids, 1], out=tau_arms)

            data.ctrl[actuator_ids] = tau_cmd
            data.ctrl[arm_actuator_ids] = tau_arms

            # ── 시뮬레이션 스텝 ──
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            # 넘어짐 감지
            if data.qpos[2] < 0.3:
                print(f"[CRASH] t={sim_time:.3f}s pelvis_z={data.qpos[2]:.3f}")
                break
            if np.any(np.isnan(data.qpos)):
                print(f"[FATAL] NaN at t={sim_time:.3f}s")
                break

            # ── 디버그 출력 (0.5초마다) ──
            step_count += 1
            if step_count % 250 == 0:
                actual_com = data.subtree_com[0]
                com_err = np.linalg.norm(com_traj[traj_idx, :2] - actual_com[:2])
                dcm_err = np.linalg.norm(dcm_ref[traj_idx] - curr_dcm)
                lf_err = np.linalg.norm(target_lf - lf_pos)
                rf_err = np.linalg.norm(target_rf - rf_pos)

                print(
                    f"[t={data.time:.2f}s walk={sim_time:.2f}s] {phase} step={step_idx} "
                    f"CoM_err={com_err*1000:.1f}mm DCM_err={dcm_err*1000:.1f}mm "
                    f"LF_err={lf_err*1000:.1f}mm RF_err={rf_err*1000:.1f}mm "
                    f"|tau_ff|={np.max(np.abs(tau_ff)):.1f} |tau_fb|={np.max(np.abs(tau_fb)):.1f}"
                )

            # ── 궤적 히스토리 ──
            com_trail[trail_idx_vis] = meas_com_pos
            trail_idx_vis = (trail_idx_vis + 1) % TRAIL_LEN
            if trail_idx_vis == 0:
                trail_filled = True
            n_trail = TRAIL_LEN if trail_filled else trail_idx_vis

            # ── 시각화 ──
            viewer.user_scn.ngeom = 0
            max_geom = viewer.user_scn.maxgeom

            # CoM 현재 (빨간 구)
            if viewer.user_scn.ngeom < max_geom:
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0],
                                    [meas_com_pos[0], meas_com_pos[1], 0.005],
                                    np.eye(3).flatten(), [1, 0.2, 0.2, 0.9])
                viewer.user_scn.ngeom += 1

            # ZMP desired (파란 구)
            if viewer.user_scn.ngeom < max_geom:
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0, 0],
                                    [calc_zmp[0], calc_zmp[1], 0.005],
                                    np.eye(3).flatten(), [0, 0.3, 1, 0.9])
                viewer.user_scn.ngeom += 1

            # Swing 목표 (초록 구)
            if ref_swing_pos is not None and viewer.user_scn.ngeom < max_geom:
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0, 0],
                                    ref_swing_pos, np.eye(3).flatten(), [0, 1, 0, 0.8])
                viewer.user_scn.ngeom += 1

            # Footsteps (빨간/파란 구)
            for fi, fs in enumerate(footsteps):
                if viewer.user_scn.ngeom >= max_geom - 50:
                    break
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                rgba = [1, 0, 0, 0.7] if fi % 2 == 0 else [0, 0, 1, 0.7]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0],
                                    [fs[0], fs[1], 0.003], np.eye(3).flatten(), rgba)
                viewer.user_scn.ngeom += 1

            # CoM 궤적 (빨간 선)
            trail_step = 3
            for i in range(0, n_trail - trail_step, trail_step):
                if viewer.user_scn.ngeom >= max_geom - 30:
                    break
                i0 = (trail_idx_vis - n_trail + i) % TRAIL_LEN
                i1 = (i0 + trail_step) % TRAIL_LEN
                p0 = np.array([com_trail[i0, 0], com_trail[i0, 1], 0.003])
                p1 = np.array([com_trail[i1, 0], com_trail[i1, 1], 0.003])
                if np.linalg.norm(p1 - p0) > 0.5:
                    continue
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002, p0, p1)
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 0.3, 0.3, 0.6]
                viewer.user_scn.ngeom += 1

            # 오프라인 CoM 궤적 (노란 선, 현재 주변)
            vis_start = max(0, traj_idx - 100)
            vis_end = min(sim_length - 1, traj_idx + 300)
            for i in range(vis_start, vis_end - 1, 3):
                if viewer.user_scn.ngeom >= max_geom - 20:
                    break
                p1 = com_traj[i]
                p2 = com_traj[i + 1]
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002, p1, p2)
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 1, 0, 0.5]
                viewer.user_scn.ngeom += 1

            # 발 궤적 (시안=LF, 주황=RF, 현재 주변)
            for i in range(vis_start, vis_end - 1, 3):
                if viewer.user_scn.ngeom >= max_geom - 5:
                    break
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002,
                    lf_traj[i], lf_traj[i + 1])
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [0, 1, 1, 0.6]
                viewer.user_scn.ngeom += 1

                if viewer.user_scn.ngeom >= max_geom - 2:
                    break
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002,
                    rf_traj[i], rf_traj[i + 1])
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 0.5, 0, 0.6]
                viewer.user_scn.ngeom += 1

            viewer.sync()

            elapsed = time.time() - step_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    print("\n[완료]")
