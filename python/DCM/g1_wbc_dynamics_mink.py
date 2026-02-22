import numpy as np
import mujoco
import mujoco.viewer
import mink
from loop_rate_limiters import RateLimiter

from Layer1 import TrajectoryOptimization
from Layer2 import SimplifiedModelControl
from Layer3 import WholeBodyController

from config import (
    N_STEPS, STEP_LENGTH, STEP_WIDTH, STEP_HEIGHT, DSP_TIME,
    K_DCM, KI_DCM, K_ZMP, K_COM,
    ARM_SWING_AMP, DT, STEP_TIME, INIT_DSP_EXTRA,
)
from zmp_sensor import compute_zmp_from_ft_sensors

xml_path = '../../model/g1/scene_23dof.xml'

# 파라미터
dt = DT
step_time = STEP_TIME

# IK 수렴 파라미터
pos_threshold = 1e-4
max_iters = 5
ik_dt = 0.005
ik_damping = 1e-3


# ========================================================================= #
# Support Phase 판별
# ========================================================================= #
def get_support_phase(traj_idx: int, samples_per_step: int) -> str:
    # 첫 스텝은 DSP가 확장됨
    first_step_samples = samples_per_step + int(INIT_DSP_EXTRA / dt)
    if traj_idx < first_step_samples:
        step_idx = 0
        local_t = traj_idx * dt
        first_dsp = DSP_TIME + INIT_DSP_EXTRA
    else:
        remaining = traj_idx - first_step_samples
        step_idx = 1 + min(remaining // samples_per_step, N_STEPS - 2)
        local_t = (remaining - (step_idx - 1) * samples_per_step) * dt
        first_dsp = DSP_TIME

    if local_t < first_dsp:
        return 'dsp'
    elif step_idx % 2 == 0:
        return 'left_support'
    else:
        return 'right_support'


# ========================================================================= #
# 팔 스윙
# ========================================================================= #
def generate_arm_swing_angles(length, step_t, dt_val, amp=ARM_SWING_AMP):
    left = np.zeros(length)
    right = np.zeros(length)
    period = 2 * step_t
    for k in range(length):
        t = k * dt_val
        phase = 2 * np.pi * t / period
        swing = amp * np.cos(phase)
        envelope = min(1.0, 0.5 * (1 - np.cos(np.pi * t / period))) if t < period else 1.0
        left[k] = swing * envelope
        right[k] = -swing * envelope
    return left, right


# ========================================================================= #
# 메인 시뮬레이션
# ========================================================================= #
if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # 1. MuJoCo 모델 로드 + keyframe으로 서있기
    # ------------------------------------------------------------------ #
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, model.key("knees_bent").id)
    mujoco.mj_forward(model, data)

    # ── 관절 매핑 (중력보상 + PD용) ──
    all_joint_names = [
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
        "left_knee", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
        "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "waist_yaw", "waist_roll", "waist_pitch",
        "left_shoulder_pitch", "left_shoulder_roll",
        "left_shoulder_yaw", "left_elbow",
        "right_shoulder_pitch", "right_shoulder_roll",
        "right_shoulder_yaw", "right_elbow",
    ]
    all_joint_ids = np.array([model.joint(n + "_joint").id for n in all_joint_names])
    all_actuator_ids = np.array([model.actuator(n).id for n in all_joint_names])
    all_qpos_ids = np.array([model.jnt_qposadr[j] for j in all_joint_ids])
    all_dof_ids = np.array([model.jnt_dofadr[j] for j in all_joint_ids])
    n_act = len(all_actuator_ids)

    q0_all = data.qpos[all_qpos_ids].copy()

    # PD 게인 (간단한 고정값)
    Kp_all = np.full(n_act, 200.0)
    Kd_all = np.full(n_act, 20.0)
    for i, n in enumerate(all_joint_names):
        if "ankle" in n:
            Kp_all[i], Kd_all[i] = 100.0, 10.0
        if "shoulder" in n or "elbow" in n:
            Kp_all[i], Kd_all[i] = 100.0, 10.0
        if n in ("waist_roll", "waist_pitch"):
            Kp_all[i], Kd_all[i] = 0.0, 0.0

    # 초기 위치 캡처
    com_init = data.subtree_com[1].copy()
    lf_init = data.site("left_foot").xpos.copy()
    rf_init = data.site("right_foot").xpos.copy()

    print(f"CoM:  {com_init}")
    print(f"LF:   {lf_init}")
    print(f"RF:   {rf_init}")
    print(f"Control dt: {dt}s, Physics dt: {model.opt.timestep}s")

    n_physics_steps = max(1, int(round(dt / model.opt.timestep)))

    # ------------------------------------------------------------------ #
    # 2. Layer 3 (WBC) 초기화
    # ------------------------------------------------------------------ #
    wbc = WholeBodyController(model, data)
    wbc.configuration.update(data.qpos)
    mujoco.mj_forward(wbc.configuration.model, wbc.configuration.data)

    wbc.com_task.set_target(com_init)
    wbc.left_foot_task.set_target(
        mink.SE3.from_rotation_and_translation(mink.SO3.identity(), lf_init))
    wbc.right_foot_task.set_target(
        mink.SE3.from_rotation_and_translation(mink.SO3.identity(), rf_init))
    wbc.pelvis_task.set_target_from_configuration(wbc.configuration)
    wbc.torso_task.set_target_from_configuration(wbc.configuration)
    wbc.posture_task.set_target(data.qpos.copy())
    wbc.arm_task.set_target(data.qpos.copy())

    # shoulder 관절 인덱스
    left_sh_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pitch_joint")
    right_sh_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pitch_joint")
    left_sh_qadr = model.jnt_qposadr[left_sh_jid]
    right_sh_qadr = model.jnt_qposadr[right_sh_jid]
    q0 = data.qpos.copy()
    left_sh_init = q0[left_sh_qadr]
    right_sh_init = q0[right_sh_qadr]

    # ------------------------------------------------------------------ #
    # 3. 시뮬레이션 루프
    # ------------------------------------------------------------------ #
    rate = RateLimiter(frequency=1.0 / dt, warn=False)
    traj_idx = 0
    sim_time = 0.0
    walking_initialized = False

    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        # 시각화 옵션: 반투명 + CoM + 접촉력
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        while viewer.is_running():
            sim_time += dt

            # --- 바로 보행 시작: 현재 위치 기준으로 궤적 생성 ---
            if not walking_initialized:
                com_init = data.subtree_com[1].copy()
                lf_init = data.site("left_foot").xpos.copy()
                rf_init = data.site("right_foot").xpos.copy()
                z_c = com_init[2]

                planner = TrajectoryOptimization(
                    z_c=z_c, step_time=step_time, dsp_time=DSP_TIME,
                    step_height=STEP_HEIGHT, dt=dt
                )
                footsteps, dcm_ref, dcm_vel_ref, com_traj, com_vel_ref, lf_traj, rf_traj = \
                    planner.compute_all_trajectories(
                        n_steps=N_STEPS, step_length=STEP_LENGTH,
                        step_width=STEP_WIDTH,
                        init_com=com_init, init_lf=lf_init, init_rf=rf_init,
                    )

                sim_length = len(com_traj)
                samples_per_step = planner.samples_per_step

                left_shoulder_swing, right_shoulder_swing = generate_arm_swing_angles(
                    sim_length, step_time, dt)

                controller = SimplifiedModelControl(
                    z_c=z_c, k_dcm=K_DCM, ki_dcm=0.0,
                    k_zmp=K_ZMP, k_com=K_COM, dt=dt
                )

                q0 = data.qpos.copy()
                left_sh_init = q0[left_sh_qadr]
                right_sh_init = q0[right_sh_qadr]

                print(f"\n[보행 시작] CoM={com_init}, z_c={z_c:.4f}m")
                print(f"컨트롤러: DCM PI")
                print(f"궤적: {sim_length} samples ({sim_length * dt:.1f}s)")
                print(f"dcm_ref[0]={dcm_ref[0]}")
                print(f"footsteps[0]={footsteps[0]}")
                print(f"CoM_xy={com_init[:2]} (vel≈0 → DCM≈CoM)")
                walking_initialized = True

            # --- 보행 구간 ---
            t_walk = sim_time
            traj_idx = int(t_walk / dt)
            if traj_idx >= sim_length:
                break

            # 데이터 가져오기 (CoM, Vel, ZMP)
            meas_com_pos = data.subtree_com[1].copy()
            mujoco.mj_subtreeVel(model, data)
            meas_com_vel = data.subtree_linvel[1].copy()
            meas_zmp = compute_zmp_from_ft_sensors(model, data)

            # --- Layer 2: 피드백 (DCM PI controller) ---
            desired_com_vel, calc_zmp, curr_dcm = controller.control_step(
                meas_com_pos=meas_com_pos,
                meas_com_vel=meas_com_vel,
                meas_zmp=meas_zmp,
                ref_dcm=dcm_ref[traj_idx],
                ref_dcm_vel=dcm_vel_ref[traj_idx],
                ref_com_pos=com_traj[traj_idx, :2],
                ref_com_vel=com_vel_ref[traj_idx],
            )

            # --- Layer 3 목표 설정 ---
            target_com = com_traj[traj_idx].copy()
            target_com[0] += desired_com_vel[0] * dt
            target_com[1] += desired_com_vel[1] * dt

            target_lf = lf_traj[traj_idx].copy()
            target_rf = rf_traj[traj_idx].copy()

            # 팔 스윙
            posture_target = q0.copy()
            posture_target[left_sh_qadr] = left_sh_init + left_shoulder_swing[traj_idx]
            posture_target[right_sh_qadr] = right_sh_init + right_shoulder_swing[traj_idx]
            wbc.update_posture_target(posture_target)

             
            torso_pitch = np.radians(2.0)
            half = torso_pitch / 2.0
            # Y축 회전 quaternion (wxyz): [cos(θ/2), 0, sin(θ/2), 0]
            torso_quat = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
            torso_target_pos = data.body("torso_link").xpos.copy()
            wbc.torso_task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(torso_quat), torso_target_pos))

            # --- Layer 3: IK 반복 수렴 ---
            wbc.com_task.set_target(target_com)
            wbc.left_foot_task.set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3.identity(), target_lf))
            wbc.right_foot_task.set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3.identity(), target_rf))

            for ik_iter in range(max_iters):
                vel = mink.solve_ik(
                    wbc.configuration, wbc.tasks, ik_dt, wbc.solver,
                    damping=ik_damping, limits=wbc.limits,
                )
                wbc.configuration.integrate_inplace(vel, ik_dt)
                
                # CoM + 양발 모두 수렴해야 break
                com_err_ik = np.linalg.norm(
                    wbc.com_task.compute_error(wbc.configuration)[:3])
                lf_err = wbc.left_foot_task.compute_error(wbc.configuration)
                rf_err = wbc.right_foot_task.compute_error(wbc.configuration)
                if (com_err_ik <= pos_threshold and
                    np.linalg.norm(lf_err[:3]) <= pos_threshold and
                    np.linalg.norm(rf_err[:3]) <= pos_threshold):
                    break

            # --- Actuator: gravity comp + PD ---
            q_ik = wbc.configuration.q.copy()
            q_target = q_ik[all_qpos_ids]
            q_curr = data.qpos[all_qpos_ids]
            dq_curr = data.qvel[all_dof_ids]

            tau_grav = data.qfrc_bias[all_dof_ids].copy()
            for i, n in enumerate(all_joint_names):
                if n in ("waist_roll", "waist_pitch"):
                    tau_grav[i] = 0.0

            tau_fb = Kp_all * (q_target - q_curr) - Kd_all * dq_curr
            tau_cmd = tau_grav + tau_fb

            np.clip(tau_cmd, model.actuator_ctrlrange[all_actuator_ids, 0],
                    model.actuator_ctrlrange[all_actuator_ids, 1], out=tau_cmd)
            data.ctrl[all_actuator_ids] = tau_cmd

            for _ in range(n_physics_steps):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            # Closed-loop
            wbc.configuration.update(data.qpos)

            # --- 디버깅 (매 스텝) ---
            actual_com = data.subtree_com[1]
            actual_lf = data.site("left_foot").xpos
            actual_rf = data.site("right_foot").xpos
            com_err = np.linalg.norm(com_traj[traj_idx, :2] - actual_com[:2])
            dcm_err = np.linalg.norm(dcm_ref[traj_idx] - curr_dcm)
            zmp_err = np.linalg.norm(calc_zmp - meas_zmp)

            # 발 추적 오차
            lf_pos_err = np.linalg.norm(target_lf - actual_lf)
            rf_pos_err = np.linalg.norm(target_rf - actual_rf)

            # support phase
            phase = get_support_phase(traj_idx, samples_per_step)

            # 접촉 수
            n_contacts = data.ncon

            # 기본 출력 (매 스텝)
            print(
                f"t={t_walk:.2f}s | "
                f"CoM_err={com_err*1000:.1f}mm | "
                f"DCM_err={dcm_err*1000:.1f}mm | "
                f"ZMP_err={zmp_err*1000:.1f}mm | "
                f"vel_fb=({desired_com_vel[0]:.3f}, {desired_com_vel[1]:.3f}) | "
                f"zmp_des=({calc_zmp[0]:.4f}, {calc_zmp[1]:.4f}) | "
                f"zmp_meas=({meas_zmp[0]:.4f}, {meas_zmp[1]:.4f})"
            )

            # 상세 출력 (0.02s 간격)
            if traj_idx % 10 == 0:
                print(
                    f"  [상세] phase={phase} | ncon={n_contacts} | "
                    f"LF_err={lf_pos_err*1000:.2f}mm | RF_err={rf_pos_err*1000:.2f}mm | "
                    f"IK_iters={ik_iter+1} | CoM_IK_err={com_err_ik*1000:.2f}mm"
                )
                print(
                    f"  [CoM] ref=({com_traj[traj_idx,0]:.4f},{com_traj[traj_idx,1]:.4f},{com_traj[traj_idx,2]:.4f}) "
                    f"act=({actual_com[0]:.4f},{actual_com[1]:.4f},{actual_com[2]:.4f}) "
                    f"z_err={abs(com_traj[traj_idx,2]-actual_com[2])*1000:.1f}mm"
                )
                print(
                    f"  [DCM] ref=({dcm_ref[traj_idx,0]:.4f},{dcm_ref[traj_idx,1]:.4f}) "
                    f"act=({curr_dcm[0]:.4f},{curr_dcm[1]:.4f}) "
                    f"integral=({controller.dcm_error_sum[0]:.5f},{controller.dcm_error_sum[1]:.5f})"
                )
                print(
                    f"  [Foot] LF_tgt=({target_lf[0]:.4f},{target_lf[1]:.4f},{target_lf[2]:.4f}) "
                    f"LF_act=({actual_lf[0]:.4f},{actual_lf[1]:.4f},{actual_lf[2]:.4f})"
                )
                print(
                    f"  [Foot] RF_tgt=({target_rf[0]:.4f},{target_rf[1]:.4f},{target_rf[2]:.4f}) "
                    f"RF_act=({actual_rf[0]:.4f},{actual_rf[1]:.4f},{actual_rf[2]:.4f})"
                )
                print(
                    f"  [ComVel] meas=({meas_com_vel[0]:.4f},{meas_com_vel[1]:.4f},{meas_com_vel[2]:.4f}) "
                    f"ref=({com_vel_ref[traj_idx,0]:.4f},{com_vel_ref[traj_idx,1]:.4f})"
                )

            # --- 시각화 ---
            viewer.user_scn.ngeom = 0
            if walking_initialized:
                # CoM 현재 위치 (바닥 투영, 큰 빨간 구체)
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0],
                                    [actual_com[0], actual_com[1], 0.005], np.eye(3).flatten(),
                                    [1, 0.2, 0.2, 0.9])
                viewer.user_scn.ngeom += 1

                # footsteps 표시 (구체)
                for fi, fs in enumerate(footsteps):
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                        break
                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.015, 0, 0],
                                        [fs[0], fs[1], 0.005], np.eye(3).flatten(),
                                        [1, 0, 0, 0.8] if fi % 2 == 0 else [0, 0, 1, 0.8])
                    viewer.user_scn.ngeom += 1

                # 전체 CoM 목표 궤적 (노란선, CoM 높이에 그리기)
                for i in range(0, sim_length - 1, 5):
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 10:
                        break
                    p1 = com_traj[i]
                    p2 = com_traj[i+1]
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, width=0.002,
                        from_=p1, to=p2)
                    viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 1, 0, 0.6]
                    viewer.user_scn.ngeom += 1

                # 전체 DCM 목표 궤적 (마젠타선, 바닥에 깔기)
                for i in range(0, len(dcm_ref) - 1, 5):
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 10:
                        break
                    p1 = np.array([dcm_ref[i, 0], dcm_ref[i, 1], 0.008])
                    p2 = np.array([dcm_ref[i+1, 0], dcm_ref[i+1, 1], 0.008])
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, width=0.002,
                        from_=p1, to=p2)
                    viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 0, 1, 0.6]
                    viewer.user_scn.ngeom += 1

                # 발 궤적 표시 (노란=RF, 시안=LF) — 현재 주변만
                vis_start = max(0, traj_idx - 100)
                vis_end = min(sim_length - 1, traj_idx + 200)
                for i in range(vis_start, vis_end - 1, 3):
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 2:
                        break
                    # RF 궤적 (빨간)
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, width=0.002,
                        from_=rf_traj[i], to=rf_traj[i+1])
                    viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 0.3, 0.3, 0.7]
                    viewer.user_scn.ngeom += 1
                    # LF 궤적 (시안)
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, width=0.002,
                        from_=lf_traj[i], to=lf_traj[i+1])
                    viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [0, 1, 1, 0.7]
                    viewer.user_scn.ngeom += 1

                # CoM 현재 주변 궤적 (초록선)
                for i in range(vis_start, vis_end - 1):
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                        break
                    p1 = np.array([com_traj[i, 0], com_traj[i, 1], 0.01])
                    p2 = np.array([com_traj[i+1, 0], com_traj[i+1, 1], 0.01])
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                        width=0.003, from_=p1, to=p2)
                    viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [0, 1, 0, 0.5]
                    viewer.user_scn.ngeom += 1

            viewer.sync()
            rate.sleep()
            traj_idx += 1

        print("\n완료!")
