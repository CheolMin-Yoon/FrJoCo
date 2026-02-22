import mujoco
import mujoco.viewer
import numpy as np
import time
import math

from Layer1 import GaitGenerator
from Layer2 import ExternalContactControl
from Layer3 import HierarchicalWholeBodyController

from config import dt, t_swing, t_stance, robot_mass, com_height, torso_height


def main() -> None:
    
    import os
    # 절대 경로로 변환
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "../../model/g1/scene_23dof.xml")
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
    
    # 초기 상태 출력
    print("\n" + "="*60)
    print("초기 스폰 상태 (knees_bent keyframe)")
    print("="*60)
    
    # CoM 정보
    com_init = data.subtree_com[1].copy()
    print(f"\n[CoM]")
    print(f"  Position: ({com_init[0]:.4f}, {com_init[1]:.4f}, {com_init[2]:.4f}) m")
    
    # 발 위치
    left_foot_init = data.site(model.site("left_foot").id).xpos.copy()
    right_foot_init = data.site(model.site("right_foot").id).xpos.copy()
    print(f"\n[발 위치]")
    print(f"  Left foot:  ({left_foot_init[0]:.4f}, {left_foot_init[1]:.4f}, {left_foot_init[2]:.4f}) m")
    print(f"  Right foot: ({right_foot_init[0]:.4f}, {right_foot_init[1]:.4f}, {right_foot_init[2]:.4f}) m")
    
    # Support polygon 중심
    support_center = (left_foot_init[:2] + right_foot_init[:2]) / 2
    print(f"  Support center: ({support_center[0]:.4f}, {support_center[1]:.4f}) m")
    
    # CoM과 support center 오프셋
    com_offset = com_init[:2] - support_center
    print(f"  CoM offset from center: ({com_offset[0]:.4f}, {com_offset[1]:.4f}) m")
    
    # 발 간격
    foot_distance = np.linalg.norm(left_foot_init - right_foot_init)
    foot_width = abs(left_foot_init[1] - right_foot_init[1])
    foot_length = abs(left_foot_init[0] - right_foot_init[0])
    print(f"  Foot width (y): {foot_width:.4f} m")
    print(f"  Foot length (x): {foot_length:.4f} m")
    print(f"  Foot distance: {foot_distance:.4f} m")
    
    # Torso/Pelvis 정보
    torso_init = data.body(model.body("torso_link").id).xpos.copy()
    pelvis_init = data.xpos[1].copy()  # body ID 1 = pelvis
    print(f"\n[Body 위치]")
    print(f"  Pelvis: ({pelvis_init[0]:.4f}, {pelvis_init[1]:.4f}, {pelvis_init[2]:.4f}) m")
    print(f"  Torso:  ({torso_init[0]:.4f}, {torso_init[1]:.4f}, {torso_init[2]:.4f}) m")
    
    # 초기 속도
    print(f"\n[초기 속도]")
    print(f"  qvel (floating base): {data.qvel[:6]}")
    print(f"  qvel max (joints): {np.max(np.abs(data.qvel[6:])):.6f} rad/s")
    
    # 관절 각도
    print(f"\n[주요 관절 각도 (rad)]")
    joint_names_check = ["left_hip_pitch", "left_knee", "left_ankle_pitch",
                         "right_hip_pitch", "right_knee", "right_ankle_pitch"]
    for jname in joint_names_check:
        jid = model.joint(jname + "_joint").id
        qpos_idx = model.jnt_qposadr[jid]
        print(f"  {jname:20s}: {data.qpos[qpos_idx]:7.4f} rad ({np.degrees(data.qpos[qpos_idx]):6.2f}°)")
    
    # 안정성 지표
    print(f"\n[안정성 지표]")
    com_height_actual = com_init[2]
    print(f"  CoM height: {com_height_actual:.4f} m")
    print(f"  CoM/height ratio: {com_height_actual/foot_width:.2f}")
    
    # ZMP 마진 (CoM이 support polygon 중심에서 얼마나 떨어져 있는지)
    zmp_margin_x = foot_length / 2 - abs(com_offset[0])
    zmp_margin_y = foot_width / 2 - abs(com_offset[1])
    if foot_length > 1e-6:
        print(f"  ZMP margin X: {zmp_margin_x:.4f} m ({zmp_margin_x/foot_length*100:.1f}% of foot length)")
    else:
        print(f"  ZMP margin X: {zmp_margin_x:.4f} m (foot_length≈0, skip %)")
    print(f"  ZMP margin Y: {zmp_margin_y:.4f} m ({zmp_margin_y/foot_width*100:.1f}% of foot width)")
    
    if zmp_margin_x < 0 or zmp_margin_y < 0:
        print(f"  ⚠️  WARNING: CoM is outside support polygon!")
    
    print("="*60 + "\n")
    
    # 초기 관절 각도 저장 (제어용)
    q0_legs = data.qpos[qpos_ids].copy()
    q0_arms = data.qpos[arm_qpos_ids].copy()
    

    # Body/Site ID 매핑 (Layer 초기화 전에 먼저 정의)
    left_foot_site_id = model.site("left_foot").id
    right_foot_site_id = model.site("right_foot").id
    torso_body_id = model.body("torso_link").id
    pelvis_body_id = 1  # pelvis는 항상 body ID 1 (전체 로봇의 루트)
    waist_joint_id = model.joint("waist_yaw_joint").id
    waist_dof_id = model.jnt_dofadr[waist_joint_id]  # DoF ID로 변환

    # Layer 초기화
    nv = model.nv
    nu = len(actuator_ids)
    
    layer1 = GaitGenerator(T_s=t_swing, T_st=t_stance)
    layer2 = ExternalContactControl(mass=robot_mass, com_height=com_height) 
    layer3 = HierarchicalWholeBodyController(nv, nu, model, data, actuator_dof_ids=dof_ids)
    
    # 초기 목표 위치 저장 (보행 전 드리프트 방지)
    initial_com_pos = data.subtree_com[1].copy()
    initial_torso_pos = data.body(torso_body_id).xpos.copy()
    
    print(f"[INFO] Initial CoM height:   {initial_com_pos[2]:.4f}m")
    print(f"[INFO] Initial Torso height: {initial_torso_pos[2]:.4f}m")
    print(f"[INFO] Config com_height (LIPM): {com_height}m")
    print(f"[INFO] Config torso_height (WBC): {torso_height}m")
    
    # PD 제어 게인 (mujoco_menagerie/unitree_g1/g1_mjx.xml 참고)
    # 기본: kp=75, kv=2 (position actuator용)
    # motor actuator는 gravity compensation 필수
    
    # 관절별 게인 설정 — motor actuator이므로 높은 게인 필요
    # 35kg 로봇, 직접 토크 제어 → Kp=200, Kd=10 수준
    Kp_array = np.full(nu, 100.0)   # 기본값 (hip, knee, waist)
    Kd_array = np.full(nu, 10.0)    # 기본값
    
    # ankle 관절 인덱스 찾기
    ankle_indices = []
    for i, name in enumerate(control_joint_names):
        if 'ankle' in name:
            ankle_indices.append(i)
    
    # ankle에 대해 낮은 게인 적용 (토크 제한 ±50 Nm)
    for idx in ankle_indices:
        Kp_array[idx] = 50.0
        Kd_array[idx] = 5.0
    
    # waist 관절 인덱스 찾기
    waist_indices = []
    for i, name in enumerate(control_joint_names):
        if 'waist' in name:
            waist_indices.append(i)
    
    # waist_roll, waist_pitch는 더미 관절 (pos="0 0 20")이므로 게인 0
    for idx in waist_indices:
        jname = control_joint_names[idx]
        if jname in ['waist_roll', 'waist_pitch']:
            Kp_array[idx] = 0.0
            Kd_array[idx] = 0.0
    
    # 팔 게인
    Kp_arm_array = np.full(len(arm_joint_names), 100.0)
    Kd_arm_array = np.full(len(arm_joint_names), 5.0)
    
    # wrist 관절 인덱스 찾기
    wrist_indices = []
    for i, name in enumerate(arm_joint_names):
        if 'wrist' in name:
            wrist_indices.append(i)
    
    # wrist에 대해 낮은 게인 적용
    for idx in wrist_indices:
        Kp_arm_array[idx] = 20.0
        Kd_arm_array[idx] = 2.0
    
    print(f"Leg gains: Kp={Kp_array[0]}, Kd={Kd_array[0]} (ankle: Kp={Kp_array[ankle_indices[0]]}, Kd={Kd_array[ankle_indices[0]]})")
    print(f"Waist gains: waist_yaw Kp={Kp_array[waist_indices[0]]}, waist_roll Kp={Kp_array[waist_indices[1]]}, waist_pitch Kp={Kp_array[waist_indices[2]]}")
    print(f"Arm gains: Kp={Kp_arm_array[0]}, Kd={Kd_arm_array[0]}")

    
    # Torso 목표 위치 초기화 (Layer3 WBC용 - torso_height 사용)
    init_torso_pos = data.body(torso_body_id).xpos.copy()
    ref_torso_pos = np.array([init_torso_pos[0], init_torso_pos[1], torso_height])


    # WBC 메모리 할당 (Dynamics WBC용으로만 사용)
    
    # 동역학 관련
    M = np.zeros((nv, nv))
    M_inv = np.zeros((nv, nv))
    C = np.zeros(nv)
    g = np.zeros(nv)
    
    # Contact Jacobian
    J_contact_left = np.zeros((3, nv))
    J_contact_right = np.zeros((3, nv))
    
    # 제어 토크
    tau_ff = np.zeros(nu)
    tau_fb = np.zeros(nu)
    tau_cmd = np.zeros(nu)
    tau_arms = np.zeros(len(arm_actuator_ids))
    
    # 디버깅 유틸리티
    def dbg_check(label, arr, step=None):
        """NaN/Inf 체크 + 요약 출력"""
        a = np.asarray(arr)
        has_nan = np.any(np.isnan(a))
        has_inf = np.any(np.isinf(a))
        if has_nan or has_inf:
            prefix = f"[step={step}] " if step is not None else ""
            print(f"  ⚠️  {prefix}{label}: NaN={has_nan}, Inf={has_inf}, "
                  f"min={np.nanmin(a):.4f}, max={np.nanmax(a):.4f}")
            return True
        return False

    def dbg_summary(label, arr):
        """배열 요약 (min/max/norm)"""
        a = np.asarray(arr)
        print(f"  {label}: norm={np.linalg.norm(a):.4f}, "
              f"min={np.min(a):.4f}, max={np.max(a):.4f}")

    # 시각화 도구 초기화

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        
        # Site axis 안 보이게 설정
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        
        # 메인 루프
        step_count = 0
        print(f"[DEBUG] 메인 루프 시작, data.time={data.time:.4f}")
        while viewer.is_running():
            step_start = time.time()
            
            # 상태 측정
            q_current = data.qpos.copy()
            dq_current = data.qvel.copy()
            
            # Torso (CoM)
            torso_pos = data.body(torso_body_id).xpos.copy()
            torso_vel = data.cvel[torso_body_id][3:].copy()  # cvel = [ang(3), lin(3)]
            
            # 발 위치
            left_foot_pos = data.site(left_foot_site_id).xpos.copy()
            right_foot_pos = data.site(right_foot_site_id).xpos.copy()
            
            # ============================================
            # 초기 대기 시간 (3초) - WBC로 자세 유지 (속도 0)
            # ============================================
            if step_count == 0:
                print(f"[DEBUG] 첫 스텝: data.time={data.time:.4f}, 초기대기 조건(data.time<3.0)={'진입' if data.time < 3.0 else '스킵'}")
            if data.time < 0.0:
                # 목표: 초기 keyframe 자세 유지 (고정 목표)
                # gravity compensation + PD로 초기 자세 고정
                tau_ff = data.qfrc_bias[dof_ids].copy()  # 중력 보상
                
                # 더미 관절 (waist_roll, waist_pitch)의 bias force는 0으로
                for idx in waist_indices:
                    jname = control_joint_names[idx]
                    if jname in ['waist_roll', 'waist_pitch']:
                        tau_ff[idx] = 0.0
                
                # PD 피드백: 초기 자세(q0_legs)로 복원
                tau_fb = (Kp_array * (q0_legs - q_current[qpos_ids]) - 
                         Kd_array * dq_current[dof_ids])
                
                tau_cmd = tau_ff + tau_fb
                
                # 0.5초마다 로그 출력
                if step_count % 250 == 0 or step_count < 5:
                    q_err = np.max(np.abs(q0_legs - q_current[qpos_ids]))
                    print(f"[t={data.time:.2f}s] 초기안정화 | pelvis_z={q_current[2]:.4f} | "
                          f"joint_err_max={q_err:.6f} rad | tau_ff_max={np.max(np.abs(tau_ff)):.1f} | "
                          f"tau_fb_max={np.max(np.abs(tau_fb)):.1f} | tau_cmd_max={np.max(np.abs(tau_cmd)):.1f}")
                    if step_count < 5:
                        # 첫 몇 스텝: 관절별 상세 출력
                        for i, name in enumerate(control_joint_names):
                            print(f"  {name:20s}: q0={q0_legs[i]:7.4f} q={q_current[qpos_ids[i]]:7.4f} "
                                  f"err={q0_legs[i]-q_current[qpos_ids[i]]:8.5f} "
                                  f"tau_ff={tau_ff[i]:7.2f} tau_fb={tau_fb[i]:7.2f} tau={tau_cmd[i]:7.2f}")
                
                # 팔 제어
                tau_arms = (Kp_arm_array * (q0_arms - q_current[arm_qpos_ids]) - 
                           Kd_arm_array * dq_current[arm_dof_ids] + 
                           data.qfrc_bias[arm_dof_ids])
                
                # 토크 제한
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
                
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                continue
            
            # ============================================
            # 목표 속도 정의 (점진적 증가, 보행 시작 3초 기준)
            # ============================================
            if data.time < 8.0:
                # 3초~8초: 0 -> 0.3 m/s로 점진적 증가
                ramp_progress = np.clip((data.time - 3.0) / 5.0, 0.0, 1.0)
                desired_vel = np.array([0.3 * ramp_progress, 0.0])
            else:
                # 8초 이후: 0.3 m/s 유지
                desired_vel = np.array([0.3, 0.0])
            
            # Layer1: Gait Generation
            try:
                phase, contact_state, swing_leg_idx = layer1.state_machine(dt, data.time, left_foot_pos, right_foot_pos)
            except Exception as e:
                print(f"[ERROR] Layer1.state_machine 실패 (step={step_count}, t={data.time:.4f}): {e}")
                import traceback; traceback.print_exc()
                break
            
            if step_count < 5 or step_count % 500 == 0:
                print(f"[DEBUG step={step_count}, t={data.time:.3f}] Layer1: phase={phase:.3f}, contact={contact_state}, swing_leg={swing_leg_idx}")
            
            # Stance 발 고정 위치 가져오기
            stance_left_pos, stance_right_pos = layer1.get_stance_foot_pos(swing_leg_idx)
            
            # Raibert Heuristic으로 다음 발자국 위치 계산
            if swing_leg_idx == -1:
                # 초기 CoM Shift Phase: 발 움직임 없음
                ref_foot_pos_L = stance_left_pos if stance_left_pos is not None else left_foot_pos
                ref_foot_pos_R = stance_right_pos if stance_right_pos is not None else right_foot_pos
            elif swing_leg_idx == 0:  # left swing
                next_footstep = layer1.Raibert_Heuristic_foot_step_planner(
                    left_foot_pos, torso_pos, torso_vel, desired_vel, swing_leg_idx, data
                )
                # Swing 시작 위치 사용
                swing_start_pos = layer1.get_swing_start_pos(swing_leg_idx)
                init_pos_L = swing_start_pos if swing_start_pos is not None else left_foot_pos
                ref_foot_pos_L = layer1.generate_swing_trajectory(phase, init_pos_L, next_footstep)
                # Stance 발은 고정 위치 사용
                ref_foot_pos_R = stance_right_pos if stance_right_pos is not None else right_foot_pos
            else:  # right swing
                next_footstep = layer1.Raibert_Heuristic_foot_step_planner(
                    right_foot_pos, torso_pos, torso_vel, desired_vel, swing_leg_idx, data
                )
                # Swing 시작 위치 사용
                swing_start_pos = layer1.get_swing_start_pos(swing_leg_idx)
                init_pos_R = swing_start_pos if swing_start_pos is not None else right_foot_pos
                ref_foot_pos_R = layer1.generate_swing_trajectory(phase, init_pos_R, next_footstep)
                # Stance 발은 고정 위치 사용
                ref_foot_pos_L = stance_left_pos if stance_left_pos is not None else left_foot_pos
            
            # ============================================
            # Layer2: ZMP Control - 목표 지면반력 계산
            # ============================================
            # 전체 로봇의 CoM 사용 (pelvis subtree = 전체 로봇)
            com_pos = data.subtree_com[1].copy()  # 인덱스 1 = pelvis (전체 로봇)
            
            # CoM 속도 계산
            mujoco.mj_subtreeVel(model, data)
            com_vel = data.subtree_linvel[1].copy()
            com_acc_ref = np.zeros(3)
            
            # Layer2에 발 위치와 swing_leg_idx 전달
            try:
                fr_left, fr_right = layer2.compute_desired_force(
                    com_pos, com_vel, com_acc_ref, contact_state, 
                    left_foot_pos, right_foot_pos, swing_leg_idx
                )
            except Exception as e:
                print(f"[ERROR] Layer2.compute_desired_force 실패 (step={step_count}, t={data.time:.4f}): {e}")
                import traceback; traceback.print_exc()
                break
            
            if step_count < 5 or step_count % 500 == 0:
                print(f"[DEBUG step={step_count}] Layer2: fr_left={fr_left}, fr_right={fr_right}")
                dbg_check("fr_left", fr_left, step_count)
                dbg_check("fr_right", fr_right, step_count)
            
            # ZMP 타겟 계산 (시각화용)
            if swing_leg_idx == -1:
                zmp_target = right_foot_pos[:2]  # 초기: 오른발로 shift
            elif contact_state[0] == 1 and contact_state[1] == 0:
                zmp_target = left_foot_pos[:2]  # 왼발 stance
            elif contact_state[0] == 0 and contact_state[1] == 1:
                zmp_target = right_foot_pos[:2]  # 오른발 stance
            else:
                zmp_target = (left_foot_pos[:2] + right_foot_pos[:2]) / 2.0
            
            # ============================================
            # Layer3: Kinematic WBC (Task 우선순위 기반)
            # ============================================
            
            # Torso 목표 높이: config의 torso_height 사용 (initial_torso_pos[2] 아님)
            if swing_leg_idx == -1:
                ref_torso_pos = np.array([torso_pos[0], torso_pos[1], torso_height])
            else:
                ref_torso_pos = np.array([torso_pos[0], torso_pos[1], torso_height])
            
            # Reference data 준비 (Layer3.py용 - ID 기반)
            # swing_leg_idx == -1 (초기 shift): swing task 없이 stance만
            if swing_leg_idx == -1:
                ref_data = {
                    'swing_leg_idx':    -1,
                    'waist_joint_id':   waist_dof_id,
                    'stance_foot_id':   left_foot_site_id,
                    'torso_body_id':    torso_body_id,
                    'ref_torso_pos':    ref_torso_pos,
                }
            else:
                stance_foot_id = right_foot_site_id if swing_leg_idx == 0 else left_foot_site_id
                swing_foot_id  = left_foot_site_id  if swing_leg_idx == 0 else right_foot_site_id
                ref_swing_pos  = ref_foot_pos_L if swing_leg_idx == 0 else ref_foot_pos_R
                ref_data = {
                    'swing_leg_idx':    swing_leg_idx,
                    'waist_joint_id':   waist_dof_id,
                    'stance_foot_id':   stance_foot_id,
                    'torso_body_id':    torso_body_id,
                    'ref_torso_pos':    ref_torso_pos,
                    'swing_foot_id':    swing_foot_id,
                    'ref_swing_pos':    ref_swing_pos,
                }
            
            # Kinematic WBC 실행 (모든 Task 통합)
            if step_count < 5 or step_count % 500 == 0:
                print(f"[DEBUG step={step_count}] Layer3 KinematicWBC 호출, ref_data keys={list(ref_data.keys())}")
            try:
                q_cmd, dq_cmd = layer3.KinematicWBC(ref_data, dt)
            except Exception as e:
                print(f"[ERROR] Layer3.KinematicWBC 실패 (step={step_count}, t={data.time:.4f}): {e}")
                import traceback; traceback.print_exc()
                break
            
            if step_count < 5 or step_count % 500 == 0:
                bad = dbg_check("q_cmd", q_cmd, step_count) or dbg_check("dq_cmd", dq_cmd, step_count)
                if bad:
                    print(f"  ref_torso_pos={ref_torso_pos}")
                    print(f"  torso_pos={torso_pos}")
                    dbg_summary("dq_cmd", dq_cmd)
            
            # ============================================
            # 5. Layer3: Dynamics WBC (토크 계산)
            # ============================================
            # 항상 left, right 순서로 전달 (Contact Jacobian 순서 일관성)
            try:
                tau_ff = layer3.DynamicsWBC(q_cmd, dq_cmd, fr_left, fr_right, 
                                            left_foot_site_id, right_foot_site_id, dt)
            except Exception as e:
                print(f"[ERROR] Layer3.DynamicsWBC 실패 (step={step_count}, t={data.time:.4f}): {e}")
                import traceback; traceback.print_exc()
                break
            
            if step_count < 5 or step_count % 500 == 0:
                dbg_check("tau_ff", tau_ff, step_count)
            
            # ============================================
            # 6. PVT Controller (피드백 제어)
            # ============================================
            # tau_cmd = tau_ff + tau_fb + gravity_compensation
            # tau_fb = Kp * (q_cmd - q_current) + Kd * (dq_cmd - dq_current) + qfrc_bias
            tau_fb = Kp_array * (q_cmd[qpos_ids] - q_current[qpos_ids]) + Kd_array * (dq_cmd[dof_ids] - dq_current[dof_ids])
            tau_cmd = tau_ff + tau_fb
            
            if step_count < 5 or step_count % 500 == 0:
                dbg_check("tau_fb", tau_fb, step_count)
                dbg_check("tau_cmd", tau_cmd, step_count)
                if step_count < 5:
                    dbg_summary("tau_ff", tau_ff)
                    dbg_summary("tau_fb", tau_fb)
                    dbg_summary("tau_cmd", tau_cmd)
            

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
            
            # ============================================
            # 디버그 출력 (0.5초마다 = 250스텝)
            # ============================================
            step_count += 1
            LOG_INTERVAL = 250  # 0.5초마다 (0.002 * 250)
            if step_count % LOG_INTERVAL == 0:
                print(f"\n{'='*70}")
                print(f"[t={data.time:.3f}s] step={step_count}")
                print(f"{'='*70}")
                
                # Layer1: Gait 상태
                print(f"[Layer1 - Gait]")
                print(f"  phase={phase:.3f}, swing_leg={swing_leg_idx}, contact={contact_state}")
                print(f"  desired_vel=({desired_vel[0]:.3f}, {desired_vel[1]:.3f}) m/s")
                
                # 발 위치 (실제 vs 목표)
                print(f"[Foot Positions]")
                print(f"  L_foot actual =({left_foot_pos[0]:.4f}, {left_foot_pos[1]:.4f}, {left_foot_pos[2]:.4f})")
                print(f"  R_foot actual =({right_foot_pos[0]:.4f}, {right_foot_pos[1]:.4f}, {right_foot_pos[2]:.4f})")
                print(f"  L_foot ref    =({ref_foot_pos_L[0]:.4f}, {ref_foot_pos_L[1]:.4f}, {ref_foot_pos_L[2]:.4f})")
                print(f"  R_foot ref    =({ref_foot_pos_R[0]:.4f}, {ref_foot_pos_R[1]:.4f}, {ref_foot_pos_R[2]:.4f})")
                if swing_leg_idx == 0:
                    foot_err = np.linalg.norm(left_foot_pos - ref_foot_pos_L)
                    print(f"  swing(L) tracking err = {foot_err:.4f} m")
                elif swing_leg_idx == 1:
                    foot_err = np.linalg.norm(right_foot_pos - ref_foot_pos_R)
                    print(f"  swing(R) tracking err = {foot_err:.4f} m")
                
                # Layer2: ZMP / GRF
                print(f"[Layer2 - ZMP/GRF]")
                print(f"  CoM pos =({com_pos[0]:.4f}, {com_pos[1]:.4f}, {com_pos[2]:.4f})")
                print(f"  CoM vel =({com_vel[0]:.4f}, {com_vel[1]:.4f}, {com_vel[2]:.4f})")
                print(f"  ZMP tgt =({zmp_target[0]:.4f}, {zmp_target[1]:.4f})")
                print(f"  CoM-ZMP dist = {np.linalg.norm(com_pos[:2] - zmp_target):.4f} m")
                print(f"  fr_left  = ({fr_left[0]:.2f}, {fr_left[1]:.2f}, {fr_left[2]:.2f}) N")
                print(f"  fr_right = ({fr_right[0]:.2f}, {fr_right[1]:.2f}, {fr_right[2]:.2f}) N")
                print(f"  total Fz = {fr_left[2]+fr_right[2]:.2f} N (mg={robot_mass*9.81:.2f})")
                
                # Layer3: Torso 추종
                print(f"[Layer3 - WBC]")
                print(f"  Torso actual =({torso_pos[0]:.4f}, {torso_pos[1]:.4f}, {torso_pos[2]:.4f})")
                print(f"  Torso ref    =({ref_torso_pos[0]:.4f}, {ref_torso_pos[1]:.4f}, {ref_torso_pos[2]:.4f})")
                torso_err = np.linalg.norm(torso_pos - ref_torso_pos)
                print(f"  Torso tracking err = {torso_err:.4f} m")
                
                # Kinematic WBC 출력
                print(f"  dq_cmd max = {np.max(np.abs(dq_cmd)):.4f} rad/s")
                print(f"  q_cmd - q_curr (joints max) = {np.max(np.abs(q_cmd[qpos_ids] - q_current[qpos_ids])):.6f} rad")
                
                # 토크 정보
                print(f"[Torques]")
                print(f"  tau_ff  max={np.max(np.abs(tau_ff)):.2f}, mean={np.mean(np.abs(tau_ff)):.2f}")
                print(f"  tau_fb  max={np.max(np.abs(tau_fb)):.2f}, mean={np.mean(np.abs(tau_fb)):.2f}")
                print(f"  tau_cmd max={np.max(np.abs(tau_cmd)):.2f}, mean={np.mean(np.abs(tau_cmd)):.2f}")
                
                # Floating base 상태 (위치/속도 발산 감지)
                print(f"[Floating Base]")
                print(f"  qpos(xyz)  =({q_current[0]:.4f}, {q_current[1]:.4f}, {q_current[2]:.4f})")
                print(f"  qpos(quat) =({q_current[3]:.4f}, {q_current[4]:.4f}, {q_current[5]:.4f}, {q_current[6]:.4f})")
                print(f"  qvel(lin)  =({dq_current[0]:.4f}, {dq_current[1]:.4f}, {dq_current[2]:.4f})")
                print(f"  qvel(ang)  =({dq_current[3]:.4f}, {dq_current[4]:.4f}, {dq_current[5]:.4f})")
                print(f"  qvel max(joints) = {np.max(np.abs(dq_current[6:])):.4f} rad/s")
                
                # 안정성 경고
                if com_pos[2] < 0.4:
                    print(f"  ⚠️  CoM height LOW: {com_pos[2]:.3f}m (falling?)")
                if np.max(np.abs(dq_current[:6])) > 5.0:
                    print(f"  ⚠️  Base velocity HIGH: max={np.max(np.abs(dq_current[:6])):.3f}")
                if np.any(np.isnan(tau_cmd)):
                    print(f"  ❌ NaN detected in tau_cmd!")
                if np.any(np.isnan(dq_cmd)):
                    print(f"  ❌ NaN detected in dq_cmd!")
            
            # ============================================
            # 시각화 (CoM, ZMP 타겟)
            # ============================================
            viewer.user_scn.ngeom = 0
            
            # CoM 바닥 투영 (빨간색 구)
            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_SPHERE, 
                    [0.03, 0, 0],  # size
                    [com_pos[0], com_pos[1], 0.01],  # position
                    np.eye(3).flatten(),  # orientation
                    [1, 0, 0, 0.9]  # rgba (빨간색)
                )
                viewer.user_scn.ngeom += 1
            
            # ZMP 타겟 (파란색 구)
            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_SPHERE,
                    [0.025, 0, 0],  # size
                    [zmp_target[0], zmp_target[1], 0.01],  # position
                    np.eye(3).flatten(),
                    [0, 0, 1, 0.9]  # rgba (파란색)
                )
                viewer.user_scn.ngeom += 1
            
            # 시뮬레이션 스텝
            mujoco.mj_step(model, data)
            
            # 다음 제어 스텝을 위해 kinematics/dynamics 업데이트
            # (Jacobian, CoM, 동역학 행렬 등을 최신 qpos로 계산)
            mujoco.mj_forward(model, data)
            
            # 넘어짐/발산 감지
            pelvis_z = data.qpos[2]
            if step_count < 5 or step_count % 500 == 0:
                print(f"[DEBUG step={step_count}, t={data.time:.3f}] pelvis_z={pelvis_z:.4f}, "
                      f"qvel_max={np.max(np.abs(data.qvel)):.2f}")
            if pelvis_z < 0.3:
                print(f"[CRASH] 로봇 넘어짐 감지! step={step_count}, t={data.time:.3f}, pelvis_z={pelvis_z:.4f}")
                print(f"  qpos[:7]={data.qpos[:7]}")
                print(f"  qvel[:6]={data.qvel[:6]}")
                print(f"  last tau_cmd: norm={np.linalg.norm(tau_cmd):.2f}, max={np.max(np.abs(tau_cmd)):.2f}")
            if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
                print(f"[FATAL] NaN 발생! step={step_count}, t={data.time:.3f}")
                print(f"  qpos has NaN: {np.any(np.isnan(data.qpos))}")
                print(f"  qvel has NaN: {np.any(np.isnan(data.qvel))}")
                break
            
            viewer.sync()
            
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()