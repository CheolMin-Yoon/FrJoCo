import mujoco
import mujoco.viewer
import numpy as np
import time

from Layer1 import GaitGenerator
from Layer2 import ExternalContactControl
from Layer3 import HierarchicalWholeBodyController

from config import dt, t_swing, t_stance, robot_mass, com_height


def main() -> None:
    
    import os
    # 절대 경로로 변환
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "../../../model/g1/scene_23dof.xml")
    xml_path = os.path.normpath(xml_path)
    
    # MuJoCo 모델 및 데이터 초기화
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = dt
    

    # 제어할 관절 이름 정의 (액추에이터 이름은 _joint 접미사 없음)
    control_joint_names = [
        # Left leg (6)
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
        "left_knee", "left_ankle_pitch", "left_ankle_roll",
        
        # Right leg (6)
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
        "right_knee", "right_ankle_pitch", "right_ankle_roll",
        
        # Waist (3)
        "waist_yaw", "waist_roll", "waist_pitch",
    ]
    
    arm_joint_names = [
        # Arms (8)
        "left_shoulder_pitch", "left_shoulder_roll", 
        "left_shoulder_yaw", "left_elbow",
        "right_shoulder_pitch", "right_shoulder_roll",
        "right_shoulder_yaw", "right_elbow",
    ]
    
    # Joint ID는 _joint 접미사 포함
    control_joint_names_with_suffix = [name + "_joint" for name in control_joint_names]
    arm_joint_names_with_suffix = [name + "_joint" for name in arm_joint_names]

    # Joint/Actuator ID 매핑
    joint_ids = np.array([model.joint(name).id for name in control_joint_names_with_suffix])
    actuator_ids = np.array([model.actuator(name).id for name in control_joint_names])
    arm_joint_ids = np.array([model.joint(name).id for name in arm_joint_names_with_suffix])
    arm_actuator_ids = np.array([model.actuator(name).id for name in arm_joint_names])
    
    # qpos/qvel 인덱스 (floating base 고려)
    qpos_ids = np.array([model.jnt_qposadr[jid] for jid in joint_ids])
    dof_ids = np.array([model.jnt_dofadr[jid] for jid in joint_ids])
    arm_qpos_ids = np.array([model.jnt_qposadr[jid] for jid in arm_joint_ids])
    arm_dof_ids = np.array([model.jnt_dofadr[jid] for jid in arm_joint_ids])
    

    # 초기 자세 설정
    key_name = "knees_bent"
    key_id = model.key(key_name).id
    
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    
    # 초기 관절 각도 저장 (제어용)
    q0_legs = data.qpos[qpos_ids].copy()
    q0_arms = data.qpos[arm_qpos_ids].copy()
    

    # Layer 초기화
    nv = model.nv
    nu = len(actuator_ids)
    
    layer1 = GaitGenerator(T_s=t_swing, T_st=t_stance)
    layer2 = ExternalContactControl(mass=robot_mass, com_height=com_height) 
    layer3 = HierarchicalWholeBodyController(nv, nu, model, data, actuator_dof_ids=dof_ids)
    
    # Body/Site ID 매핑
    left_foot_site_id = model.site("left_foot").id
    right_foot_site_id = model.site("right_foot").id
    torso_body_id = model.body("torso_link").id
    waist_joint_id = model.joint("waist_yaw_joint").id
    waist_dof_id = model.jnt_dofadr[waist_joint_id]  # DoF ID로 변환
    
    # PD 제어 게인 (mujoco_menagerie/unitree_g1/g1_mjx.xml 참고)
    # 기본: kp=75, kv=2 (position actuator용)
    # motor actuator는 gravity compensation 필수
    
    # 관절별 게인 설정
    # 2차 오차 동역학: ddq = Kp*(q_ref-q) + Kd*(dq_ref-dq)
    # 임계 감쇠 (ζ=1): Kd = 2*sqrt(Kp)
    Kp_array = np.full(nu, 75.0)   # 기본값
    Kd_array = 2.0 * np.sqrt(Kp_array)  # ζ=1 → ~17.3
    
    # ankle 관절 인덱스 찾기
    ankle_indices = []
    for i, name in enumerate(control_joint_names):
        if 'ankle' in name:
            ankle_indices.append(i)
    
    # ankle에 대해 낮은 게인 적용
    for idx in ankle_indices:
        Kp_array[idx] = 20.0
        Kd_array[idx] = 2.0 * np.sqrt(Kp_array[idx])  # ~8.9
    
    # 팔 게인
    Kp_arm_array = np.full(len(arm_joint_names), 75.0)
    Kd_arm_array = 2.0 * np.sqrt(Kp_arm_array)  # ζ=1
    
    # wrist 관절 인덱스 찾기
    wrist_indices = []
    for i, name in enumerate(arm_joint_names):
        if 'wrist' in name:
            wrist_indices.append(i)
    
    # wrist에 대해 낮은 게인 적용
    for idx in wrist_indices:
        Kp_arm_array[idx] = 20.0
        Kd_arm_array[idx] = 2.0 * np.sqrt(Kp_arm_array[idx])
    
    print(f"Leg gains: Kp={Kp_array[0]:.1f}, Kd={Kd_array[0]:.1f} (ankle: Kp={Kp_array[4]:.1f}, Kd={Kd_array[4]:.1f})")
    print(f"Arm gains: Kp={Kp_arm_array[0]:.1f}, Kd={Kd_arm_array[0]:.1f}")

    
    # Torso 목표 위치 초기화 (속도 적분용)
    init_torso_pos = data.body(torso_body_id).xpos.copy()
    ref_torso_pos = np.array([init_torso_pos[0], init_torso_pos[1], com_height])
    
    # 가상 CoM 위치 (desired_vel 적분용 — 전진 목표 생성)
    virtual_com_pos = np.array([init_torso_pos[0], init_torso_pos[1], com_height])


    # 제어 토크
    tau_ff = np.zeros(nu)
    tau_fb = np.zeros(nu)
    tau_cmd = np.zeros(nu)
    tau_arms = np.zeros(len(arm_actuator_ids))
    
    # 시각화 도구 초기화

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        
        # Site axis 안 보이게 설정
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        
        # 메인 루프
        step_count = 0
        while viewer.is_running():
            step_start = time.time()
            
            # 상태 측정
            q_current = data.qpos.copy()
            dq_current = data.qvel.copy()
            
            # CoM (전체 로봇 CoM = world body의 subtree_com)
            com_pos_measured = data.subtree_com[0].copy()  # 전체 로봇 CoM
            torso_pos = com_pos_measured  # 이후 코드 호환용 alias
            torso_vel = data.qvel[:3].copy()  # floating base linear velocity (CoM 속도 근사)
            
            # 발 위치
            left_foot_pos = data.site(left_foot_site_id).xpos.copy()
            right_foot_pos = data.site(right_foot_site_id).xpos.copy()
            
            # ============================================
            # 초기 대기 시간 (2초)
            # ============================================
            if data.time < 0.0:
                # 초기 자세 유지 (PD 제어 + Gravity compensation)
                tau_cmd = (Kp_array * (q0_legs - q_current[qpos_ids]) - 
                          Kd_array * dq_current[dof_ids] + 
                          data.qfrc_bias[dof_ids])
                
                tau_arms = (Kp_arm_array * (q0_arms - q_current[arm_qpos_ids]) - 
                           Kd_arm_array * dq_current[arm_dof_ids] + 
                           data.qfrc_bias[arm_dof_ids])
                
                np.clip(tau_cmd, model.actuator_ctrlrange[actuator_ids, 0], 
                        model.actuator_ctrlrange[actuator_ids, 1], out=tau_cmd)
                np.clip(tau_arms, model.actuator_ctrlrange[arm_actuator_ids, 0],
                        model.actuator_ctrlrange[arm_actuator_ids, 1], out=tau_arms)
                
                data.ctrl[actuator_ids] = tau_cmd
                data.ctrl[arm_actuator_ids] = tau_arms
                
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                viewer.sync()
                
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                continue
            
            # ============================================
            # 목표 속도 정의 (점진적 증가)
            # ============================================
            if data.time < 7.0:
                 #2초~7초: 0 -> 0.3 m/s로 점진적 증가
                ramp_progress = max(0.0, (data.time - 2.0) / 5.0)
                desired_vel = np.array([1.0 * ramp_progress, 0.0])
            else:
                # 7초 이후: 0.3 m/s 유지
                desired_vel = np.array([0.1, 0.0])
            
            # Layer1: Gait Generation
            phase, contact_state, swing_leg_idx = layer1.state_machine(dt, data.time)
            
            # Raibert Heuristic으로 다음 발자국 위치 계산 (스윙 시작 시 한 번만)
            if swing_leg_idx == 0:  # left swing
                next_footstep = layer1.Raibert_Heuristic_foot_step_planner(
                    left_foot_pos, torso_pos, torso_vel, desired_vel, swing_leg_idx, data
                )
                # Swing 시작 위치 사용 (반드시 존재해야 함)
                swing_start_pos = layer1.get_swing_start_pos(swing_leg_idx)
                if swing_start_pos is None:
                    # 첫 스윙이거나 초기화 안 된 경우 현재 위치 사용
                    swing_start_pos = left_foot_pos.copy()
                ref_foot_pos_L = layer1.generate_swing_trajectory(phase, swing_start_pos, next_footstep)
                ref_foot_pos_R = right_foot_pos  # stance
            else:  # right swing
                next_footstep = layer1.Raibert_Heuristic_foot_step_planner(
                    right_foot_pos, torso_pos, torso_vel, desired_vel, swing_leg_idx, data
                )
                # Swing 시작 위치 사용 (반드시 존재해야 함)
                swing_start_pos = layer1.get_swing_start_pos(swing_leg_idx)
                if swing_start_pos is None:
                    # 첫 스윙이거나 초기화 안 된 경우 현재 위치 사용
                    swing_start_pos = right_foot_pos.copy()
                ref_foot_pos_R = layer1.generate_swing_trajectory(phase, swing_start_pos, next_footstep)
                ref_foot_pos_L = left_foot_pos  # stance
            
            # ============================================
            # Layer2: ZMP Control - 목표 지면반력 계산
            # ============================================
            com_pos = com_pos_measured  # 전체 로봇 CoM (이미 위에서 계산)
            com_vel = torso_vel       # linear velocity (이미 위에서 수정)
            com_acc_ref = np.zeros(3)
            
            # ZMP target: contact_state에 따라 stance foot 선택
            if contact_state[0] == 1 and contact_state[1] == 0:
                # Left stance
                stance_foot_pos = left_foot_pos.copy()
            elif contact_state[0] == 0 and contact_state[1] == 1:
                # Right stance
                stance_foot_pos = right_foot_pos.copy()
            else:
                # Double support
                stance_foot_pos = (left_foot_pos + right_foot_pos) / 2.0
            
            fr_left, fr_right = layer2.compute_desired_force(
                com_pos, com_vel, com_acc_ref, stance_foot_pos, contact_state
            )
            
            # ============================================
            # Layer3: Kinematic WBC (Task 우선순위 기반)
            # ============================================
            
            # 가상 CoM을 desired_vel로 이동 (전진 목표 생성)
            virtual_com_pos[0] += desired_vel[0] * dt
            virtual_com_pos[1] += desired_vel[1] * dt
            virtual_com_pos[2] = com_height
            
            # Torso 목표 위치: 가상 CoM 위치
            ref_torso_pos = virtual_com_pos.copy()
            
            # Reference data 준비
            stance_foot_id = right_foot_site_id if swing_leg_idx == 0 else left_foot_site_id
            swing_foot_id = left_foot_site_id if swing_leg_idx == 0 else right_foot_site_id
            ref_swing_pos = ref_foot_pos_L if swing_leg_idx == 0 else ref_foot_pos_R
            
            ref_data = {
                'waist_joint_id': waist_dof_id,  # DoF ID 사용
                'stance_foot_id': stance_foot_id,
                'torso_body_id': torso_body_id,
                'swing_foot_id': swing_foot_id,
                'ref_swing_pos': ref_swing_pos,
                'ref_torso_pos': ref_torso_pos,
            }
            
            # Kinematic WBC 실행 (모든 Task 통합)
            q_cmd, dq_cmd = layer3.KinematicWBC(ref_data, dt)
            
            # ============================================
            # 5. Layer3: Dynamics WBC (토크 계산)
            # ============================================
            # 항상 left, right 순서로 전달 (Contact Jacobian 순서 일관성)
            tau_ff = layer3.DynamicsWBC(q_cmd, dq_cmd, fr_left, fr_right, 
                                        left_foot_site_id, right_foot_site_id, dt)
            
            # ============================================
            # 6. PVT Controller (피드백 제어)
            # ============================================
            # tau_cmd = tau_ff + tau_fb + gravity_compensation
            # tau_fb = Kp * (q_cmd - q_current) + Kd * (dq_cmd - dq_current) + qfrc_bias
            tau_fb = (Kp_array * (q_cmd[qpos_ids] - q_current[qpos_ids]) + 
                     Kd_array * (dq_cmd[dof_ids] - dq_current[dof_ids]) + 
                     data.qfrc_bias[dof_ids])
            tau_cmd = tau_ff + tau_fb
            

            # 팔 제어 (초기 자세 유지)
            tau_arms = (Kp_arm_array * (q0_arms - q_current[arm_qpos_ids]) - 
                       Kd_arm_array * dq_current[arm_dof_ids] + 
                       data.qfrc_bias[arm_dof_ids])
            
            # 토크 제한
            np.clip(tau_cmd, model.actuator_ctrlrange[actuator_ids, 0], 
                    model.actuator_ctrlrange[actuator_ids, 1], out=tau_cmd)
            np.clip(tau_arms, model.actuator_ctrlrange[arm_actuator_ids, 0],
                    model.actuator_ctrlrange[arm_actuator_ids, 1], out=tau_arms)
            
            # 제어 명령 전송
            data.ctrl[actuator_ids] = tau_cmd
            data.ctrl[arm_actuator_ids] = tau_arms
            
            # 디버그 출력 (5초마다)
            step_count += 1
            if step_count % 2500 == 0:  # 5초마다 (0.002 * 2500 = 5s)
                print(f"[t={data.time:.1f}s] phase={phase:.2f}, swing_leg={swing_leg_idx}, "
                      f"torso_pos=({torso_pos[0]:.3f}, {torso_pos[1]:.3f}, {torso_pos[2]:.3f}), "
                      f"desired_vel=({desired_vel[0]:.2f}, {desired_vel[1]:.2f})")
            
            # 시뮬레이션 스텝
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            viewer.sync()
            
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()