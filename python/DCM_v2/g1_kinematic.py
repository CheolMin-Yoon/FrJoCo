"""
g1_kinematic.py — DCM 플래너 + DCM PI 피드백 + mink IK (키네마틱 WBC)

Layer1 (TrajectoryOptimization) → 오프라인 궤적 (DCM, CoM, 발)
Layer2 (SimplifiedModelControl)  → DCM PI 피드백 → desired CoM velocity
Layer3 (mink IK)                 → CoM/발/토르소 목표 → 관절각

최종: tau = tau_grav + Kp*(q_ik - q) + Kd*(0 - dq)
"""

import numpy as np
import mujoco
import mujoco.viewer
import mink
import time
import os

from Layer1 import TrajectoryOptimization
from Layer2 import SimplifiedModelControl

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

# IK 파라미터
IK_DT = 0.001
IK_DAMPING = 1e-3
IK_MAX_ITERS = 1
IK_POS_THRESHOLD = 1e-4

# CoM shift 구간 샘플 수
N_SHIFT = int(COM_SHIFT_TIME / DT) if COM_SHIFT_TIME > 0 else 0


# ========================================================================= #
# Support Phase 판별 (shift 구간 오프셋 포함)
# ========================================================================= #
def get_support_phase(traj_idx, samples_per_step):
    # shift 구간은 DSP 취급
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


# ========================================================================= #
# 메인
# ========================================================================= #
if __name__ == "__main__":
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
    torso_init = data.body(torso_body).xpos.copy()
    z_c = com_init[2]
    torso_com_z_offset = torso_init[2] - com_init[2]  # torso와 CoM의 높이 차이

    print(f"[INIT] CoM:   ({com_init[0]:.4f}, {com_init[1]:.4f}, {com_init[2]:.4f})")
    print(f"[INIT] Torso: ({torso_init[0]:.4f}, {torso_init[1]:.4f}, {torso_init[2]:.4f})")
    print(f"[INIT] LF:    ({lf_init[0]:.4f}, {lf_init[1]:.4f}, {lf_init[2]:.4f})")
    print(f"[INIT] RF:    ({rf_init[0]:.4f}, {rf_init[1]:.4f}, {rf_init[2]:.4f})")
    print(f"[INIT] z_c={z_c:.4f}, torso_com_z_offset={torso_com_z_offset:.4f}")
    print(f"[INIT] 궤적은 안정화 후 생성됩니다.")

    # ── mink IK 설정 ──
    configuration = mink.Configuration(model)
    configuration.update(data.qpos)
    mujoco.mj_forward(configuration.model, configuration.data)

    com_task = mink.ComTask(cost=200.0)
    left_foot_task = mink.FrameTask(
        frame_name="left_foot", frame_type="site",
        position_cost=100.0, orientation_cost=10.0, lm_damping=0.01)
    right_foot_task = mink.FrameTask(
        frame_name="right_foot", frame_type="site",
        position_cost=100.0, orientation_cost=10.0, lm_damping=0.01)
    torso_task = mink.FrameTask(
        frame_name="torso_link", frame_type="body",
        position_cost=200.0, orientation_cost=100.0, lm_damping=0.01)
    pelvis_task = mink.FrameTask(
        frame_name="pelvis", frame_type="body",
        position_cost=0.0, orientation_cost=200.0, lm_damping=0.01)
    posture_task = mink.PostureTask(model, cost=1.0)
    arm_task = mink.PostureTask(model, cost=0.0)

    tasks = [com_task, left_foot_task, right_foot_task, torso_task, pelvis_task, posture_task, arm_task]
    limits = [mink.ConfigurationLimit(model)]
    ik_solver = "daqp"

    # 초기 타겟 설정
    com_task.set_target(com_init)
    left_foot_task.set_target(
        mink.SE3.from_rotation_and_translation(mink.SO3.identity(), lf_init))
    right_foot_task.set_target(
        mink.SE3.from_rotation_and_translation(mink.SO3.identity(), rf_init))
    torso_task.set_target_from_configuration(configuration)
    pelvis_task.set_target_from_configuration(configuration)
    posture_task.set_target(data.qpos.copy())
    arm_task.set_target(data.qpos.copy())

    # ── Layer1/Layer2: 보행 시작 후 생성 (post-stabilization 값 사용) ──
    planner = None
    controller = None
    footsteps = None
    dcm_ref = dcm_vel_ref = com_traj = com_vel_ref = lf_traj = rf_traj = None
    sim_length = 0
    samples_per_step = 0

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

            # ============================================
            # 초기 안정화 — gravity comp + PD
            # ============================================
            if data.time < STABILIZE_TIME:
                tau_grav = data.qfrc_bias[dof_ids].copy()
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
                configuration.update(data.qpos)
                mujoco.mj_forward(configuration.model, configuration.data)
                posture_task.set_target(data.qpos.copy())
                arm_task.set_target(data.qpos.copy())

                # post-stabilization 실제 값 캡처
                walk_com = data.subtree_com[0].copy()
                walk_torso = data.body(torso_body).xpos.copy()
                walk_lf = data.site(lf_site).xpos.copy()
                walk_rf = data.site(rf_site).xpos.copy()
                walk_zmp = compute_zmp_from_ft_sensors(model, data)

                # torso-com offset 업데이트
                torso_com_z_offset = walk_torso[2] - walk_com[2]
                z_c = walk_com[2]

                # Layer1: 궤적 생성 (post-stabilization 값 사용)
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

                # Layer2: 컨트롤러 (post-stabilization z_c)
                controller = SimplifiedModelControl(
                    z_c=z_c, k_dcm=K_DCM, ki_dcm=KI_DCM,
                    k_zmp=K_ZMP, k_com=K_COM, dt=dt,
                )

                print(f"[보행 시작] t={data.time:.2f}s")
                print(f"  CoM:   ({walk_com[0]:.4f}, {walk_com[1]:.4f}, {walk_com[2]:.4f})")
                print(f"  Torso: ({walk_torso[0]:.4f}, {walk_torso[1]:.4f}, {walk_torso[2]:.4f})")
                print(f"  LF:    ({walk_lf[0]:.4f}, {walk_lf[1]:.4f}, {walk_lf[2]:.4f})")
                print(f"  RF:    ({walk_rf[0]:.4f}, {walk_rf[1]:.4f}, {walk_rf[2]:.4f})")
                print(f"  ZMP:   ({walk_zmp[0]:.4f}, {walk_zmp[1]:.4f})")
                print(f"  z_c={z_c:.4f}, torso_com_z_offset={torso_com_z_offset:.4f}")
                print(f"  궤적: {sim_length} samples ({sim_length * dt:.1f}s)")
                print(f"  dcm_ref[0]: ({dcm_ref[0][0]:.4f}, {dcm_ref[0][1]:.4f})")
                print(f"  com_traj[0]: ({com_traj[0][0]:.4f}, {com_traj[0][1]:.4f}, {com_traj[0][2]:.4f})")

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

            # ── 목표 설정 ──
            target_com = com_traj[traj_idx].copy()
            target_com[0] += desired_com_vel[0] * dt
            target_com[1] += desired_com_vel[1] * dt

            target_lf = lf_traj[traj_idx].copy()
            target_rf = rf_traj[traj_idx].copy()

            phase, step_idx = get_support_phase(traj_idx, samples_per_step)

            # ── Torso 목표 (CoM과 함께 전진, 실제 CoM 높이 기반) ──
            ref_torso = np.array([target_com[0], target_com[1], target_com[2] + torso_com_z_offset])

            # ── mink IK 풀기 ──
            com_task.set_target(target_com)
            left_foot_task.set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3.identity(), target_lf))
            right_foot_task.set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3.identity(), target_rf))
            torso_task.set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3.identity(), ref_torso))
            pelvis_task.set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3.identity(),
                    data.body("pelvis").xpos.copy()))

            for ik_iter in range(IK_MAX_ITERS):
                vel = mink.solve_ik(
                    configuration, tasks, IK_DT, ik_solver,
                    damping=IK_DAMPING, limits=limits,
                )
                configuration.integrate_inplace(vel, IK_DT)

                com_err_ik = np.linalg.norm(com_task.compute_error(configuration)[:3])
                lf_err_ik = np.linalg.norm(left_foot_task.compute_error(configuration)[:3])
                rf_err_ik = np.linalg.norm(right_foot_task.compute_error(configuration)[:3])
                if (com_err_ik <= IK_POS_THRESHOLD and
                    lf_err_ik <= IK_POS_THRESHOLD and
                    rf_err_ik <= IK_POS_THRESHOLD):
                    break

            # ── IK 결과 → 목표 관절각 (팔은 스폰 자세 유지) ──
            q_ik = configuration.q.copy()
            q_ik_legs = q_ik[qpos_ids]

            # ── gravity comp + PD (IK 목표 추종) ──
            tau_grav = data.qfrc_bias[dof_ids].copy()
            for i, n in enumerate(leg_names):
                if n in ("waist_roll", "waist_pitch"):
                    tau_grav[i] = 0.0

            tau_fb = Kp * (q_ik_legs - q_curr[qpos_ids]) - Kd * dq_curr[dof_ids]
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

            # ── 시뮬레이션 스텝 ──
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            # Closed-loop: IK config를 실제 자세로 동기화
            configuration.update(data.qpos)

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
                lf_pos = data.site(lf_site).xpos
                rf_pos = data.site(rf_site).xpos
                com_err = np.linalg.norm(com_traj[traj_idx, :2] - actual_com[:2])
                dcm_err = np.linalg.norm(dcm_ref[traj_idx] - curr_dcm)
                lf_err = np.linalg.norm(target_lf - lf_pos)
                rf_err = np.linalg.norm(target_rf - rf_pos)

                print(
                    f"[t={data.time:.2f}s walk={sim_time:.2f}s] {phase} step={step_idx} "
                    f"CoM_err={com_err*1000:.1f}mm DCM_err={dcm_err*1000:.1f}mm "
                    f"LF_err={lf_err*1000:.1f}mm RF_err={rf_err*1000:.1f}mm "
                    f"|tau_fb|={np.max(np.abs(tau_fb)):.1f} IK_iter={ik_iter+1}"
                )

            # ── 궤적 히스토리 ──
            com_trail[trail_idx_vis] = meas_com_pos
            trail_idx_vis = (trail_idx_vis + 1) % TRAIL_LEN
            if trail_idx_vis == 0:
                trail_filled = True
            n_trail = TRAIL_LEN if trail_filled else trail_idx_vis

            # ── 시각화 (전체 궤적 미리 표시) ──
            viewer.user_scn.ngeom = 0
            max_geom = viewer.user_scn.maxgeom

            # (1) CoM 현재 위치 (빨간 큰 구)
            if viewer.user_scn.ngeom < max_geom:
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0, 0],
                                    [meas_com_pos[0], meas_com_pos[1], 0.005],
                                    np.eye(3).flatten(), [1, 0, 0, 1.0])
                viewer.user_scn.ngeom += 1

            # (2) ZMP desired (파란 큰 구)
            if viewer.user_scn.ngeom < max_geom:
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0],
                                    [calc_zmp[0], calc_zmp[1], 0.005],
                                    np.eye(3).flatten(), [0, 0, 1, 1.0])
                viewer.user_scn.ngeom += 1

            # (3) Footsteps (빨간=왼발, 파란=오른발)
            for fi, fs in enumerate(footsteps):
                if viewer.user_scn.ngeom >= max_geom - 10:
                    break
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                rgba = [1, 0, 0, 1.0] if fi % 2 == 0 else [0, 0, 1, 1.0]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.015, 0, 0],
                                    [fs[0], fs[1], 0.003], np.eye(3).flatten(), rgba)
                viewer.user_scn.ngeom += 1

            # (4) 전체 CoM 오프라인 궤적 — DSP 구간은 초록으로 표시
            traj_step = max(1, sim_length // 800)
            for i in range(0, sim_length - traj_step, traj_step):
                if viewer.user_scn.ngeom >= max_geom - 100:
                    break
                p1 = com_traj[i]
                p2 = com_traj[min(i + traj_step, sim_length - 1)]
                ph, _ = get_support_phase(i, samples_per_step)
                if ph == 'dsp':
                    color = [0, 0.8, 0, 1.0]
                else:
                    color = [1, 1, 0, 1.0]
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, p1, p2)
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = color
                viewer.user_scn.ngeom += 1

            # (5) 전체 DCM 궤적 (마젠타 선, 바닥)
            for i in range(0, len(dcm_ref) - traj_step, traj_step):
                if viewer.user_scn.ngeom >= max_geom - 80:
                    break
                p1 = np.array([dcm_ref[i, 0], dcm_ref[i, 1], 0.008])
                p2 = np.array([dcm_ref[min(i + traj_step, len(dcm_ref)-1), 0],
                               dcm_ref[min(i + traj_step, len(dcm_ref)-1), 1], 0.008])
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, p1, p2)
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 0, 1, 1.0]
                viewer.user_scn.ngeom += 1

            # (6) 전체 발 궤적 (시안=LF, 주황=RF)
            foot_step = max(1, sim_length // 400)
            for i in range(0, sim_length - foot_step, foot_step):
                if viewer.user_scn.ngeom >= max_geom - 20:
                    break
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003,
                    lf_traj[i], lf_traj[min(i + foot_step, sim_length - 1)])
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [0, 1, 1, 1.0]
                viewer.user_scn.ngeom += 1

                if viewer.user_scn.ngeom >= max_geom - 10:
                    break
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003,
                    rf_traj[i], rf_traj[min(i + foot_step, sim_length - 1)])
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 0.5, 0, 1.0]
                viewer.user_scn.ngeom += 1

            # (7) 실제 CoM 궤적 (빨간 선)
            trail_step = 3
            for i in range(0, n_trail - trail_step, trail_step):
                if viewer.user_scn.ngeom >= max_geom - 5:
                    break
                i0 = (trail_idx_vis - n_trail + i) % TRAIL_LEN
                i1 = (i0 + trail_step) % TRAIL_LEN
                p0 = np.array([com_trail[i0, 0], com_trail[i0, 1], 0.003])
                p1 = np.array([com_trail[i1, 0], com_trail[i1, 1], 0.003])
                if np.linalg.norm(p1 - p0) > 0.5:
                    continue
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, p0, p1)
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 0, 0, 1.0]
                viewer.user_scn.ngeom += 1

            viewer.sync()

            elapsed = time.time() - step_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    print("\n[완료]")
