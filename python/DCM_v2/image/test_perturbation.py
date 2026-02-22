"""
외란 응답 테스트: Standing + WBC 파이프라인

사용법:
  1. python test_perturbation.py
  2. MuJoCo viewer에서 Ctrl + Right Click으로 로봇에 외란
  3. 뷰어 닫으면 자동으로 그래프 표시

관찰 포인트:
  - 외란 후 관절 각도가 원래로 돌아오는 속도 (settling time)
  - 오버슈트/진동 여부 → 감쇠비 확인
  - 진동 주기 T → 실제 고유진동수 f = 1/T
"""
import numpy as np
import mujoco
import mujoco.viewer
import time
import os

from config import (
    DT, GRAVITY, ROBOT_MASS, COM_HEIGHT, TORSO_HEIGHT,
    LEG_KP, LEG_KD, ANKLE_KP, ANKLE_KD, ARM_KP, ARM_KD,
)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.normpath(os.path.join(script_dir, "../../model/g1/scene_23dof.xml"))

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = DT

    # 관절 매핑
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

    # 초기 자세
    key_id = model.key("knees_bent").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    q0_legs = data.qpos[qpos_ids].copy()
    q0_arms = data.qpos[arm_qpos_ids].copy()

    lf_site = model.site("left_foot").id
    rf_site = model.site("right_foot").id
    torso_body = model.body("torso_link").id

    # PD 게인
    Kp = np.full(nu, LEG_KP)
    Kd = np.full(nu, LEG_KD)
    for i, n in enumerate(leg_names):
        if "ankle" in n:
            Kp[i], Kd[i] = ANKLE_KP, ANKLE_KD

    Kp_arm = np.full(len(arm_names), ARM_KP)
    Kd_arm = np.full(len(arm_names), ARM_KD)

    # 로깅 버퍼
    LOG_SIZE = int(60.0 / DT)  # 60초
    log_t = np.zeros(LOG_SIZE)
    log_q = np.zeros((LOG_SIZE, nu))       # 관절 각도 오차
    log_tau = np.zeros((LOG_SIZE, nu))     # 토크
    log_com = np.zeros((LOG_SIZE, 3))      # CoM 위치
    log_com_vel = np.zeros((LOG_SIZE, 3))  # CoM 속도
    log_idx = 0

    print("=== 외란 응답 테스트 ===")
    print("Ctrl + Right Click으로 로봇에 외란을 주세요")
    print("뷰어를 닫으면 그래프가 표시됩니다")
    print(f"PD 게인: LEG Kp={LEG_KP:.1f} Kd={LEG_KD:.1f}, ANKLE Kp={ANKLE_KP:.1f} Kd={ANKLE_KD:.1f}")

    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        while viewer.is_running() and log_idx < LOG_SIZE:
            step_start = time.time()

            q_curr = data.qpos.copy()
            dq_curr = data.qvel.copy()
            lf_pos = data.site(lf_site).xpos.copy()
            rf_pos = data.site(rf_site).xpos.copy()
            com_pos = data.subtree_com[1].copy()

            # Standing: gravity comp + PD (g1_wbc_dynamics_qp.py stabilize와 동일)
            tau_grav = data.qfrc_bias[dof_ids].copy()
            tau_fb = Kp * (q0_legs - q_curr[qpos_ids]) - Kd * dq_curr[dof_ids]
            tau_cmd = tau_grav + tau_fb

            # 팔
            tau_arms = (Kp_arm * (q0_arms - q_curr[arm_qpos_ids])
                        - Kd_arm * dq_curr[arm_dof_ids]
                        + data.qfrc_bias[arm_dof_ids])

            np.clip(tau_cmd, model.actuator_ctrlrange[actuator_ids, 0],
                    model.actuator_ctrlrange[actuator_ids, 1], out=tau_cmd)
            np.clip(tau_arms, model.actuator_ctrlrange[arm_actuator_ids, 0],
                    model.actuator_ctrlrange[arm_actuator_ids, 1], out=tau_arms)

            data.ctrl[actuator_ids] = tau_cmd
            data.ctrl[arm_actuator_ids] = tau_arms

            # 로깅
            log_t[log_idx] = data.time
            log_q[log_idx] = q_curr[qpos_ids] - q0_legs  # 초기 대비 오차
            log_tau[log_idx] = tau_cmd
            log_com[log_idx] = com_pos
            log_com_vel[log_idx] = data.qvel[:3].copy()
            log_idx += 1

            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            viewer.sync()

            elapsed = time.time() - step_start
            if DT - elapsed > 0:
                time.sleep(DT - elapsed)

    # 그래프
    n = log_idx
    plot_results(log_t[:n], log_q[:n], log_tau[:n], log_com[:n], log_com_vel[:n], leg_names)


def plot_results(t, q_err, tau, com, com_vel, joint_names):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle('Perturbation Response (Standing + WBC)')

    # 1. 관절 각도 오차 (hip pitch만)
    ax = axes[0, 0]
    for i, name in enumerate(joint_names):
        if 'hip_pitch' in name:
            ax.plot(t, np.degrees(q_err[:, i]), label=name)
    ax.set_ylabel('Joint Error (deg)')
    ax.set_title('Hip Pitch Response')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. 관절 각도 오차 (ankle pitch만)
    ax = axes[0, 1]
    for i, name in enumerate(joint_names):
        if 'ankle_pitch' in name:
            ax.plot(t, np.degrees(q_err[:, i]), label=name)
    ax.set_ylabel('Joint Error (deg)')
    ax.set_title('Ankle Pitch Response')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. 토크 (hip pitch)
    ax = axes[1, 0]
    for i, name in enumerate(joint_names):
        if 'hip_pitch' in name:
            ax.plot(t, tau[:, i], label=name)
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Hip Pitch Torque')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. 토크 (ankle pitch)
    ax = axes[1, 1]
    for i, name in enumerate(joint_names):
        if 'ankle_pitch' in name:
            ax.plot(t, tau[:, i], label=name)
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Ankle Pitch Torque')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. CoM 위치
    ax = axes[2, 0]
    ax.plot(t, com[:, 0], label='x')
    ax.plot(t, com[:, 1], label='y')
    ax.plot(t, com[:, 2], label='z')
    ax.set_ylabel('Position (m)')
    ax.set_xlabel('Time (s)')
    ax.set_title('CoM Position')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. CoM 속도
    ax = axes[2, 1]
    ax.plot(t, com_vel[:, 0], label='vx')
    ax.plot(t, com_vel[:, 1], label='vy')
    ax.plot(t, com_vel[:, 2], label='vz')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('CoM Velocity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'perturbation_response.png'), dpi=150)
    plt.show()
    print("그래프 저장: perturbation_response.png")


if __name__ == "__main__":
    main()
