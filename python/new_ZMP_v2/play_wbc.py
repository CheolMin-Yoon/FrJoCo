"""
play_wbc.py — Task-Space Inverse Dynamics WBC 기반 보행 시뮬레이션

Layer1 (GaitGenerator)  → 상태 머신 + Raibert + Cycloid/Bezier 스윙 궤적
Layer2 (ExternalContactControl) → ZMP 기반 desired contact force
Layer3_qp (TaskSpaceWBC) → 단일 QP로 feedforward 토크 생성

최종 제어: tau = tau_ff + Kp*(q_ref - q) + Kd*(dq_ref - dq)
"""

import os
import time
import numpy as np
import mujoco
import mujoco.viewer

from Layer1 import GaitGenerator
from Layer2 import ExternalContactControl
from Layer3_qp import TaskSpaceWBC

from config import (
    dt, t_swing, t_stance,
    robot_mass, com_height, torso_height,
)


def main():
    # ── 모델 로드 ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.normpath(os.path.join(script_dir, "../../model/g1/scene_23dof.xml"))

    model = mujoco.MjModel.from_xml_path(xml_path)
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

    # ── 초기 자세 (keyframe) ──
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
    Kp = np.full(nu, 200.0)
    Kd = np.full(nu, 10.0)
    for i, n in enumerate(leg_names):
        if "ankle" in n:
            Kp[i], Kd[i] = 50.0, 5.0

    Kp_arm = np.full(len(arm_names), 100.0)
    Kd_arm = np.full(len(arm_names), 5.0)

    # ── Layer 초기화 ──
    layer1 = GaitGenerator(T_s=t_swing, T_st=t_stance)
    layer2 = ExternalContactControl(mass=robot_mass, com_height=com_height)
    layer3 = TaskSpaceWBC(model.nv, nu, model, data, actuator_dof_ids=dof_ids)

    # ── 이전 스텝 swing (더 이상 수치 미분 불필요) ──

    # ── 궤적 히스토리 (실시간 시각화용) ──
    TRAIL_LEN = 500  # 최근 N 샘플 (1초분)
    com_trail = np.zeros((TRAIL_LEN, 3))
    lf_trail = np.zeros((TRAIL_LEN, 3))
    rf_trail = np.zeros((TRAIL_LEN, 3))
    zmp_trail = np.zeros((TRAIL_LEN, 2))
    trail_idx = 0
    trail_filled = False

    print(f"[INFO] QP WBC 시뮬레이션 시작")
    print(f"  torso_height={torso_height}, com_height={com_height}")
    print(f"  dt={dt}, t_swing={t_swing}, t_stance={t_stance}")

    step_count = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
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

            torso_pos = data.body(torso_body).xpos.copy()
            torso_vel_full = data.cvel[torso_body]  # [ang(3), lin(3)]
            torso_vel = torso_vel_full[3:].copy()    # linear only

            lf_pos = data.site(lf_site).xpos.copy()
            rf_pos = data.site(rf_site).xpos.copy()

            # ============================================
            # 초기 안정화 (2초) — gravity comp + PD
            # ============================================
            if data.time < 0.0:
                tau_ff = data.qfrc_bias[dof_ids].copy()

                tau_fb = Kp * (q0_legs - q_curr[qpos_ids]) - Kd * dq_curr[dof_ids]
                tau_cmd = tau_ff + tau_fb

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

            # ============================================
            # 목표 속도 (점진적 증가)
            # ============================================
            desired_vel = np.array([0.3, 0.0])

            # ============================================
            # Layer1: Gait Generation
            # ============================================
            phase, contact_state, swing_leg_idx = layer1.state_machine(
                dt, data.time, lf_pos, rf_pos
            )

            stance_left_pos, stance_right_pos = layer1.get_stance_foot_pos(swing_leg_idx)

            # 발 목표 위치 계산
            T_ssp = layer1.T_ssp  # SSP 시간 (phase→시간 변환용)
            ref_swing_vel_3 = np.zeros(3)
            ref_swing_acc_3 = np.zeros(3)
            
            if swing_leg_idx == -1:
                ref_lf = stance_left_pos if stance_left_pos is not None else lf_pos
                ref_rf = stance_right_pos if stance_right_pos is not None else rf_pos
                swing_site_id = -1
                ref_swing_pos = None
            elif swing_leg_idx == 0:  # left swing
                next_fs = layer1.Raibert_Heuristic_foot_step_planner(
                    lf_pos, torso_pos, torso_vel, desired_vel, swing_leg_idx, data
                )
                swing_start = layer1.get_swing_start_pos(swing_leg_idx)
                init_L = swing_start if swing_start is not None else lf_pos
                ref_lf, s_vel, s_acc = layer1.generate_swing_trajectory(phase, init_L, next_fs)
                # phase 미분 → 시간 미분: ds/dt = 1/T_ssp
                ref_swing_vel_3 = s_vel / T_ssp
                ref_swing_acc_3 = s_acc / (T_ssp ** 2)
                ref_rf = stance_right_pos if stance_right_pos is not None else rf_pos
                swing_site_id = lf_site
                ref_swing_pos = ref_lf
            else:  # right swing
                next_fs = layer1.Raibert_Heuristic_foot_step_planner(
                    rf_pos, torso_pos, torso_vel, desired_vel, swing_leg_idx, data
                )
                swing_start = layer1.get_swing_start_pos(swing_leg_idx)
                init_R = swing_start if swing_start is not None else rf_pos
                ref_rf, s_vel, s_acc = layer1.generate_swing_trajectory(phase, init_R, next_fs)
                ref_swing_vel_3 = s_vel / T_ssp
                ref_swing_acc_3 = s_acc / (T_ssp ** 2)
                ref_lf = stance_left_pos if stance_left_pos is not None else lf_pos
                swing_site_id = rf_site
                ref_swing_pos = ref_rf

            # Swing 속도/가속도 (해석적, 6DoF로 확장 — angular은 0)
            ref_swing_vel_6 = np.zeros(6)
            ref_swing_acc_6 = np.zeros(6)
            if ref_swing_pos is not None:
                ref_swing_vel_6[:3] = ref_swing_vel_3
                ref_swing_acc_6[:3] = ref_swing_acc_3
            else:
                prev_swing_pos = None

            # ============================================
            # Layer2: ZMP → desired contact force
            # ============================================
            com_pos = data.subtree_com[1].copy()
            mujoco.mj_subtreeVel(model, data)
            com_vel = data.subtree_linvel[1].copy()

            # DSP 진행률 및 다음 stance 발 정보
            dsp_progress = layer1.get_dsp_progress()
            # 짝수 스텝: 오른발 스윙 → 왼발이 stance
            next_stance_is_left = (layer1.step_count % 2 == 0)

            fr_left, fr_right = layer2.compute_desired_force(
                com_pos, com_vel, np.zeros(3), contact_state,
                lf_pos, rf_pos, swing_leg_idx,
                dsp_progress=dsp_progress,
                next_stance_is_left=next_stance_is_left,
            )

            # ============================================
            # Layer3: QP WBC → feedforward torque
            # ============================================
            ref_torso = np.array([torso_pos[0], torso_pos[1], torso_height])

            tau_ff = layer3.compute_torque(
                fr_left=fr_left,
                fr_right=fr_right,
                left_foot_site_id=lf_site,
                right_foot_site_id=rf_site,
                torso_body_id=torso_body,
                ref_torso_pos=ref_torso,
                swing_foot_site_id=swing_site_id,
                ref_swing_pos=ref_swing_pos,
                ref_swing_vel=ref_swing_vel_6 if ref_swing_pos is not None else None,
                ref_swing_acc=ref_swing_acc_6 if ref_swing_pos is not None else None,
                dt=dt,
            )

            # ============================================
            # PD 피드백 + feedforward
            # ============================================
            # q_ref: 초기 자세 (보행 중에도 PD 기준점)
            tau_fb = Kp * (q0_legs - q_curr[qpos_ids]) - Kd * dq_curr[dof_ids]
            tau_cmd = tau_ff + tau_fb

            # 팔: 초기 자세 유지
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

            # ============================================
            # 디버그 출력 (0.5초마다)
            # ============================================
            step_count += 1
            if step_count % 250 == 0:
                com_z = data.subtree_com[1][2]
                pelvis_z = data.qpos[2]
                tau_ff_max = np.max(np.abs(tau_ff))
                tau_fb_max = np.max(np.abs(tau_fb))
                phase_name = layer1.gait_phase_name

                print(
                    f"[t={data.time:.2f}s] "
                    f"{phase_name} step={layer1.step_count} "
                    f"p={phase:.2f} swing={swing_leg_idx} "
                    f"pelvis_z={pelvis_z:.3f} com_z={com_z:.3f} "
                    f"|tau_ff|={tau_ff_max:.1f} |tau_fb|={tau_fb_max:.1f} "
                    f"vel_cmd=({desired_vel[0]:.2f},{desired_vel[1]:.2f})"
                )

                # 발 추적 오차
                if swing_leg_idx == 0 and ref_swing_pos is not None:
                    err = np.linalg.norm(lf_pos - ref_swing_pos)
                    print(f"  swing(L) err={err*1000:.1f}mm")
                elif swing_leg_idx == 1 and ref_swing_pos is not None:
                    err = np.linalg.norm(rf_pos - ref_swing_pos)
                    print(f"  swing(R) err={err*1000:.1f}mm")

            # ============================================
            # 시뮬레이션 스텝
            # ============================================
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            # 넘어짐 감지
            if data.qpos[2] < 0.3:
                print(f"[CRASH] t={data.time:.3f}s pelvis_z={data.qpos[2]:.3f}")
                break
            if np.any(np.isnan(data.qpos)):
                print(f"[FATAL] NaN at t={data.time:.3f}s")
                break

            # ============================================
            # 궤적 히스토리 업데이트
            # ============================================
            com_trail[trail_idx] = com_pos
            lf_trail[trail_idx] = lf_pos
            rf_trail[trail_idx] = rf_pos
            # ZMP 타겟 (Layer2에서 사용한 것과 동일하게 계산)
            if swing_leg_idx == -1:
                if next_stance_is_left:
                    alpha = 0.5 * (1 - np.cos(np.pi * dsp_progress))
                    zt = (1 - alpha) * (lf_pos[:2] + rf_pos[:2]) / 2.0 + alpha * lf_pos[:2]
                else:
                    alpha = 0.5 * (1 - np.cos(np.pi * dsp_progress))
                    zt = (1 - alpha) * (lf_pos[:2] + rf_pos[:2]) / 2.0 + alpha * rf_pos[:2]
            elif contact_state[0] == 1 and contact_state[1] == 0:
                zt = lf_pos[:2]
            elif contact_state[0] == 0 and contact_state[1] == 1:
                zt = rf_pos[:2]
            else:
                zt = (lf_pos[:2] + rf_pos[:2]) / 2.0
            zmp_trail[trail_idx] = zt

            trail_idx = (trail_idx + 1) % TRAIL_LEN
            if trail_idx == 0:
                trail_filled = True
            n_trail = TRAIL_LEN if trail_filled else trail_idx

            # ============================================
            # 실시간 시각화
            # ============================================
            viewer.user_scn.ngeom = 0
            max_geom = viewer.user_scn.maxgeom

            # --- (1) CoM 현재 위치 (빨간 구, 바닥 투영) ---
            if viewer.user_scn.ngeom < max_geom:
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0],
                    [com_pos[0], com_pos[1], 0.005],
                    np.eye(3).flatten(), [1, 0.2, 0.2, 0.9])
                viewer.user_scn.ngeom += 1

            # --- (2) ZMP 타겟 (파란 구, 바닥) ---
            if viewer.user_scn.ngeom < max_geom:
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0, 0],
                    [zt[0], zt[1], 0.005],
                    np.eye(3).flatten(), [0, 0.3, 1, 0.9])
                viewer.user_scn.ngeom += 1

            # --- (3) Swing 목표 (초록 구) ---
            if ref_swing_pos is not None and viewer.user_scn.ngeom < max_geom:
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0, 0],
                    ref_swing_pos,
                    np.eye(3).flatten(), [0, 1, 0, 0.8])
                viewer.user_scn.ngeom += 1

            # --- (4) Swing 착지 목표점 (노란 구, 바닥) ---
            if swing_leg_idx >= 0 and viewer.user_scn.ngeom < max_geom:
                landing = layer1.fixed_next_footstep_left if swing_leg_idx == 0 else layer1.fixed_next_footstep_right
                if landing is not None:
                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_initGeom(
                        g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.018, 0, 0],
                        [landing[0], landing[1], 0.003],
                        np.eye(3).flatten(), [1, 1, 0, 0.8])
                    viewer.user_scn.ngeom += 1

            # --- (5) CoM 궤적 (빨간 선, 바닥 투영) ---
            trail_step = 3  # 매 3샘플마다 선분
            for i in range(0, n_trail - trail_step, trail_step):
                if viewer.user_scn.ngeom >= max_geom - 20:
                    break
                i0 = (trail_idx - n_trail + i) % TRAIL_LEN
                i1 = (i0 + trail_step) % TRAIL_LEN
                p0 = np.array([com_trail[i0, 0], com_trail[i0, 1], 0.003])
                p1 = np.array([com_trail[i1, 0], com_trail[i1, 1], 0.003])
                if np.linalg.norm(p1 - p0) > 0.5:
                    continue  # 점프 방지
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002,
                    p0, p1)
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 0.3, 0.3, 0.6]
                viewer.user_scn.ngeom += 1

            # --- (6) ZMP 궤적 (파란 선, 바닥) ---
            for i in range(0, n_trail - trail_step, trail_step):
                if viewer.user_scn.ngeom >= max_geom - 20:
                    break
                i0 = (trail_idx - n_trail + i) % TRAIL_LEN
                i1 = (i0 + trail_step) % TRAIL_LEN
                p0 = np.array([zmp_trail[i0, 0], zmp_trail[i0, 1], 0.002])
                p1 = np.array([zmp_trail[i1, 0], zmp_trail[i1, 1], 0.002])
                if np.linalg.norm(p1 - p0) > 0.5:
                    continue
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0015,
                    p0, p1)
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [0, 0.3, 1, 0.5]
                viewer.user_scn.ngeom += 1

            # --- (7) 왼발 궤적 (시안 선) ---
            for i in range(0, n_trail - trail_step, trail_step):
                if viewer.user_scn.ngeom >= max_geom - 10:
                    break
                i0 = (trail_idx - n_trail + i) % TRAIL_LEN
                i1 = (i0 + trail_step) % TRAIL_LEN
                p0 = lf_trail[i0]
                p1 = lf_trail[i1]
                if np.linalg.norm(p1 - p0) > 0.5:
                    continue
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002,
                    p0, p1)
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [0, 1, 1, 0.6]
                viewer.user_scn.ngeom += 1

            # --- (8) 오른발 궤적 (주황 선) ---
            for i in range(0, n_trail - trail_step, trail_step):
                if viewer.user_scn.ngeom >= max_geom - 5:
                    break
                i0 = (trail_idx - n_trail + i) % TRAIL_LEN
                i1 = (i0 + trail_step) % TRAIL_LEN
                p0 = rf_trail[i0]
                p1 = rf_trail[i1]
                if np.linalg.norm(p1 - p0) > 0.5:
                    continue
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002,
                    p0, p1)
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 0.5, 0, 0.6]
                viewer.user_scn.ngeom += 1

            viewer.sync()

            elapsed = time.time() - step_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    print("[완료]")


if __name__ == "__main__":
    main()
