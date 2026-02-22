"""
play_kinematic.py — new_ZMP 순수 키네마틱 버전
Layer1 (GaitGenerator)  → Raibert heuristic 실시간 발자국 + 스윙 궤적
mink IK                 → CoM/발/토르소 목표 → 관절각 → qpos 직접 대입
"""

import mujoco
import mujoco.viewer
import numpy as np
import mink
import time
import os

from Layer1 import GaitGenerator
from config import (
    dt, t_swing, t_stance, com_height,
)

# IK 파라미터
IK_DT = 0.005
IK_DAMPING = 1e-3
IK_MAX_ITERS = 5


def main() -> None:
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
    arm_joint_ids = np.array([model.joint(n + "_joint").id for n in arm_names])

    qpos_ids = np.array([model.jnt_qposadr[j] for j in joint_ids])
    arm_qpos_ids = np.array([model.jnt_qposadr[j] for j in arm_joint_ids])

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

    # ── 초기 위치 캡처 ──
    init_com = data.subtree_com[0].copy()
    init_torso = data.body(torso_body).xpos.copy()
    lf_init = data.site(lf_site).xpos.copy()
    rf_init = data.site(rf_site).xpos.copy()

    torso_com_z_offset = init_torso[2] - init_com[2]
    target_torso_z = com_height + torso_com_z_offset

    print(f"[INIT] CoM={init_com}, LF={lf_init}, RF={rf_init}")
    print(f"[INIT] torso_z={init_torso[2]:.4f}, com_z={init_com[2]:.4f}, target_torso_z={target_torso_z:.4f}")

    # ── Layer1 ──
    layer1 = GaitGenerator(T_s=t_swing, T_st=t_stance)

    # ── mink IK 설정 ──
    configuration = mink.Configuration(model)
    configuration.update(data.qpos)
    mujoco.mj_forward(configuration.model, configuration.data)

    com_task = mink.ComTask(cost=100.0)
    left_foot_task = mink.FrameTask(
        frame_name="left_foot", frame_type="site",
        position_cost=200.0, orientation_cost=100.0, lm_damping=0.01)
    right_foot_task = mink.FrameTask(
        frame_name="right_foot", frame_type="site",
        position_cost=200.0, orientation_cost=100.0, lm_damping=0.01)
    torso_task = mink.FrameTask(
        frame_name="torso_link", frame_type="body",
        position_cost=50.0, orientation_cost=5.0, lm_damping=0.01)
    posture_task = mink.PostureTask(model, cost=1.0)
    arm_task = mink.PostureTask(model, cost=0.0)

    tasks = [com_task, left_foot_task, right_foot_task, torso_task, posture_task, arm_task]
    limits = [mink.ConfigurationLimit(model)]
    ik_solver = "daqp"

    # 초기 타겟 설정
    com_task.set_target(init_com)
    left_foot_task.set_target(
        mink.SE3.from_rotation_and_translation(mink.SO3.identity(), lf_init))
    right_foot_task.set_target(
        mink.SE3.from_rotation_and_translation(mink.SO3.identity(), rf_init))
    torso_task.set_target_from_configuration(configuration)
    posture_task.set_target(data.qpos.copy())
    arm_task.set_target(data.qpos.copy())

    # ── 제어 상태 ──
    ref_torso_pos = np.array([init_torso[0], init_torso[1], target_torso_z])
    virtual_com_pos = np.array([init_com[0], init_com[1], target_torso_z])


    # ── 메인 루프 ──
    step_count = 0

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

            com_pos = data.subtree_com[0].copy()
            torso_pos = com_pos
            torso_vel = data.qvel[:3].copy()
            left_foot_pos = data.site(lf_site).xpos.copy()
            right_foot_pos = data.site(rf_site).xpos.copy()

            # ── 목표 생성 (walking) ──
            desired_vel = np.array([0.3, 0.0])

            virtual_com_pos[0] = init_com[0] + desired_vel[0] * data.time
            virtual_com_pos[1] = init_com[1] + desired_vel[1] * data.time
            virtual_com_pos[2] = target_torso_z
            ref_torso_pos = virtual_com_pos.copy()

            # Layer1: Gait
            phase, contact_state, swing_leg_idx = layer1.state_machine(dt, data.time)

            if swing_leg_idx == 0:  # left swing
                next_fs = layer1.Raibert_Heuristic_foot_step_planner(
                    left_foot_pos, torso_pos, torso_vel, desired_vel, 0, data)
                swing_start = layer1.get_swing_start_pos(0)
                if swing_start is None:
                    swing_start = left_foot_pos.copy()
                target_lf = layer1.generate_swing_trajectory(phase, swing_start, next_fs)
                target_rf = right_foot_pos.copy()
            else:  # right swing
                next_fs = layer1.Raibert_Heuristic_foot_step_planner(
                    right_foot_pos, torso_pos, torso_vel, desired_vel, 1, data)
                swing_start = layer1.get_swing_start_pos(1)
                if swing_start is None:
                    swing_start = right_foot_pos.copy()
                target_rf = layer1.generate_swing_trajectory(phase, swing_start, next_fs)
                target_lf = left_foot_pos.copy()

            target_com = np.array([ref_torso_pos[0], ref_torso_pos[1], init_com[2]])
            ref_torso = ref_torso_pos.copy()

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

            # ── 순수 kinematic: IK 결과를 qpos에 직접 대입 ──
            q_ik = configuration.q.copy()
            # 팔은 스폰 자세 유지
            q_ik[arm_qpos_ids] = q0_arms
            data.qpos[:] = q_ik
            data.time += dt
            mujoco.mj_forward(model, data)

            # ── 디버그 출력 ──
            step_count += 1
            if step_count % 500 == 0:
                lf_err = np.linalg.norm(target_lf - left_foot_pos)
                rf_err = np.linalg.norm(target_rf - right_foot_pos)
                com_err = np.linalg.norm(target_com[:2] - com_pos[:2])
                print(f"[t={data.time:.2f}s] "
                      f"CoM_err={com_err*1000:.1f}mm "
                      f"LF_err={lf_err*1000:.1f}mm RF_err={rf_err*1000:.1f}mm")

            viewer.sync()
            elapsed = time.time() - step_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    print("[완료]")


if __name__ == "__main__":
    main()
