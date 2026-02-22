"""
g1_traj_viewer.py — DCM 궤적 확인용 (mj_step 없음, 순수 kinematic)

Layer1 오프라인 궤적 + mink IK → qpos 직접 대입 + mj_forward만 호출
물리 시뮬레이션 없이 궤적만 시각화
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
    DT, STEP_TIME, INIT_DSP_EXTRA, COM_SHIFT_TIME,
    GRAVITY, COM_HEIGHT, TORSO_HEIGHT,
)

xml_path = '../../model/g1/scene_23dof.xml'
dt = DT

IK_DT = 0.1
IK_DAMPING = 1e-3
IK_MAX_ITERS = 5

N_SHIFT = int(COM_SHIFT_TIME / DT) if COM_SHIFT_TIME > 0 else 0


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


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_full = os.path.normpath(os.path.join(script_dir, xml_path))

    model = mujoco.MjModel.from_xml_path(xml_full)
    data = mujoco.MjData(model)
    model.opt.timestep = dt

    # ── 관절 매핑 ──
    arm_names = [
        "left_shoulder_pitch", "left_shoulder_roll",
        "left_shoulder_yaw", "left_elbow",
        "right_shoulder_pitch", "right_shoulder_roll",
        "right_shoulder_yaw", "right_elbow",
    ]
    arm_joint_ids = np.array([model.joint(n + "_joint").id for n in arm_names])
    arm_qpos_ids = np.array([model.jnt_qposadr[j] for j in arm_joint_ids])

    # ── 초기 자세 ──
    key_id = model.key("knees_bent").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    q0_arms = data.qpos[arm_qpos_ids].copy()

    # ── Site/Body ID ──
    lf_site = model.site("left_foot").id
    rf_site = model.site("right_foot").id
    torso_body = model.body("torso_link").id

    # ── 초기 위치 캡처 ──
    com_init = data.subtree_com[0].copy()
    lf_init = data.site(lf_site).xpos.copy()
    rf_init = data.site(rf_site).xpos.copy()
    z_c = com_init[2]

    print(f"CoM: {com_init}, LF: {lf_init}, RF: {rf_init}")

    # ── mink IK 설정 ──
    configuration = mink.Configuration(model)
    configuration.update(data.qpos)
    mujoco.mj_forward(configuration.model, configuration.data)

    com_task = mink.ComTask(cost=100.0)
    left_foot_task = mink.FrameTask(
        frame_name="left_foot", frame_type="site",
        position_cost=500.0, orientation_cost=100.0, lm_damping=0.01)
    right_foot_task = mink.FrameTask(
        frame_name="right_foot", frame_type="site",
        position_cost=500.0, orientation_cost=100.0, lm_damping=0.01)
    torso_task = mink.FrameTask(
        frame_name="torso_link", frame_type="body",
        position_cost=50.0, orientation_cost=5.0, lm_damping=0.01)
    posture_task = mink.PostureTask(model, cost=1.0)

    tasks = [com_task, left_foot_task, right_foot_task, torso_task, posture_task]
    limits = [mink.ConfigurationLimit(model)]
    ik_solver = "daqp"

    com_task.set_target(com_init)
    left_foot_task.set_target(
        mink.SE3.from_rotation_and_translation(mink.SO3.identity(), lf_init))
    right_foot_task.set_target(
        mink.SE3.from_rotation_and_translation(mink.SO3.identity(), rf_init))
    torso_task.set_target_from_configuration(configuration)
    posture_task.set_target(data.qpos.copy())

    # ── Layer1: 오프라인 궤적 생성 ──
    planner = TrajectoryOptimization(
        z_c=z_c, step_time=STEP_TIME, dsp_time=DSP_TIME,
        step_height=STEP_HEIGHT, dt=dt,
    )
    footsteps, dcm_ref, dcm_vel_ref, com_traj, com_vel_ref, lf_traj, rf_traj = \
        planner.compute_all_trajectories(
            n_steps=N_STEPS, step_length=STEP_LENGTH,
            step_width=STEP_WIDTH,
            init_com=com_init, init_lf=lf_init, init_rf=rf_init,
        )

    sim_length = len(com_traj)
    samples_per_step = planner.samples_per_step

    print(f"\n[궤적 뷰어] {sim_length} samples ({sim_length * dt:.1f}s), mj_step 없음")

    # ── 메인 루프 ──
    step_count = 0

    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE

        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        traj_idx = 0

        while viewer.is_running() and traj_idx < sim_length:
            step_start = time.time()

            # ── 궤적에서 목표 읽기 ──
            target_com = com_traj[traj_idx].copy()
            target_lf = lf_traj[traj_idx].copy()
            target_rf = rf_traj[traj_idx].copy()
            ref_torso = np.array([target_com[0], target_com[1], TORSO_HEIGHT])

            phase, step_idx = get_support_phase(traj_idx, samples_per_step)

            # ── mink IK ──
            com_task.set_target(target_com)
            left_foot_task.set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3.identity(), target_lf))
            right_foot_task.set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3.identity(), target_rf))
            torso_task.set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3.identity(), ref_torso))

            for _ in range(IK_MAX_ITERS):
                vel = mink.solve_ik(
                    configuration, tasks, IK_DT, ik_solver,
                    damping=IK_DAMPING, limits=limits)
                configuration.integrate_inplace(vel, IK_DT)

            # ── qpos 직접 대입 (물리 없음) ──
            q_ik = configuration.q.copy()
            q_ik[arm_qpos_ids] = q0_arms
            data.qpos[:] = q_ik
            data.time = traj_idx * dt
            mujoco.mj_forward(model, data)

            # ── 디버그 출력 ──
            step_count += 1
            if step_count % 500 == 0:
                actual_com = data.subtree_com[0]
                com_err = np.linalg.norm(target_com[:2] - actual_com[:2])
                lf_err = np.linalg.norm(target_lf - data.site(lf_site).xpos)
                rf_err = np.linalg.norm(target_rf - data.site(rf_site).xpos)
                print(f"[t={data.time:.2f}s] {phase} step={step_idx} "
                      f"CoM_err={com_err*1000:.1f}mm "
                      f"LF_err={lf_err*1000:.1f}mm RF_err={rf_err*1000:.1f}mm")

            # ── 시각화 ──
            viewer.user_scn.ngeom = 0
            max_geom = viewer.user_scn.maxgeom

            # (1) CoM 현재 위치 (빨간 구)
            if viewer.user_scn.ngeom < max_geom:
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0],
                                    [target_com[0], target_com[1], 0.005],
                                    np.eye(3).flatten(), [1, 0.2, 0.2, 0.9])
                viewer.user_scn.ngeom += 1

            # (2) Footsteps
            for fi, fs in enumerate(footsteps):
                if viewer.user_scn.ngeom >= max_geom - 10:
                    break
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                rgba = [1, 0, 0, 0.7] if fi % 2 == 0 else [0, 0, 1, 0.7]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0],
                                    [fs[0], fs[1], 0.003], np.eye(3).flatten(), rgba)
                viewer.user_scn.ngeom += 1

            # (3) CoM 궤적 (노란/초록)
            traj_step = max(1, sim_length // 800)
            for i in range(0, sim_length - traj_step, traj_step):
                if viewer.user_scn.ngeom >= max_geom - 100:
                    break
                p1 = com_traj[i]
                p2 = com_traj[min(i + traj_step, sim_length - 1)]
                ph, _ = get_support_phase(i, samples_per_step)
                color = [0, 1, 0, 0.8] if ph == 'dsp' else [1, 1, 0, 0.5]
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002, p1, p2)
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = color
                viewer.user_scn.ngeom += 1

            # (4) DCM 궤적 (마젠타)
            for i in range(0, len(dcm_ref) - traj_step, traj_step):
                if viewer.user_scn.ngeom >= max_geom - 80:
                    break
                p1 = np.array([dcm_ref[i, 0], dcm_ref[i, 1], 0.008])
                j = min(i + traj_step, len(dcm_ref) - 1)
                p2 = np.array([dcm_ref[j, 0], dcm_ref[j, 1], 0.008])
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002, p1, p2)
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 0, 1, 0.5]
                viewer.user_scn.ngeom += 1

            # (5) 발 궤적 (시안=LF, 주황=RF)
            foot_step = max(1, sim_length // 400)
            for i in range(0, sim_length - foot_step, foot_step):
                if viewer.user_scn.ngeom >= max_geom - 20:
                    break
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002,
                    lf_traj[i], lf_traj[min(i + foot_step, sim_length - 1)])
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [0, 1, 1, 0.6]
                viewer.user_scn.ngeom += 1
                if viewer.user_scn.ngeom >= max_geom - 10:
                    break
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002,
                    rf_traj[i], rf_traj[min(i + foot_step, sim_length - 1)])
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 0.5, 0, 0.6]
                viewer.user_scn.ngeom += 1

            viewer.sync()
            traj_idx += 1

            elapsed = time.time() - step_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    print("\n[완료]")
