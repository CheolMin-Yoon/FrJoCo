'''
Integrated ZMP-WBC Framework for Dynamic Stability in
Humanoid Robot Locomotion의 구현체

Layer3은 논문에서 제안하는 Architechture의 Multi-Task Coordinated Control의 구현체로써 
논문의 Section 3.2 Whole-Body Control 기반으로 작성되었음

논문에서 주장하는 WBC는 A. Kinematic WBC와 B. Dynamics WBC로 분리됨
'''

import numpy as np
from qpsolvers import solve_qp
from config import (
    # Kinematic WBC
    wbc_waist_kp, wbc_contact_kp, 
    wbc_torso_kp_pos, wbc_torso_kp_ori,
    wbc_swing_kp_pos, wbc_swing_kp_ori,
    wbc_damping,
    # Dynamics WBC
    qp_w_ddq, qp_w_f,
    qp_friction_coef,
    qp_ddq_max, qp_force_max
)

class HierarchicalWholeBodyController():
    '''
    Hierarchical Whole-Body Controller
    
    A. Kinematic WBC: Null-Space 기반 Task 우선순위 제어 (1차 오차 동역학)
        1. Redundant DoF constraints (1 DoF) - Waist 고정
        2. Static Contact (3 DoF) - 지지발 위치 고정
        3. Floating Base Pose Tracking (6 DoF) - 몸통 자세 제어
        4. Swing Leg Pose Tracking (6 DoF) - 스윙발 궤적 추종
    
    B. Dynamics WBC: QP 기반 토크 최적화
        - 동역학 제약 만족
        - 마찰 원뿔 제약
        - Feedforward 토크 생성
    '''
    
    def __init__(self, nv, nu, model=None, data=None, actuator_dof_ids=None):
        self.nv = nv
        self.nu = nu
        self.model = model
        self.data = data
        self.actuator_dof_ids = actuator_dof_ids  # Actuated joints의 DoF ID
        
        # 초기 자세 저장 (waist constraint용)
        if data is not None:
            self.q0 = data.qpos.copy()
            self.dq0 = data.qvel.copy()
        else:
            self.q0 = np.zeros(nv)
            self.dq0 = np.zeros(nv)
        
        # Static Contact Task용 stance 시작 위치 저장
        self.stance_start_pos = None
    
    # A. Kinematic WBC
    def KinematicWBC(self, ref_data, dt):

        # Null-space 초기화
        N = np.eye(self.nv)
        dq_cmd = np.zeros(self.nv)
        
        # 디버그 카운터
        if not hasattr(self, '_wbc_call_count'):
            self._wbc_call_count = 0
        self._wbc_call_count += 1
        
        # Stance foot 변경 감지 (Static Contact Task 초기화용)
        if 'stance_foot_id' in ref_data:
            if not hasattr(self, 'prev_stance_foot_id'):
                self.prev_stance_foot_id = ref_data['stance_foot_id']
            elif self.prev_stance_foot_id != ref_data['stance_foot_id']:
                # Stance foot가 바뀌면 위치 리셋
                print(f"[WBC] Stance foot changed! Resetting contact position.")
                self.stance_start_pos = None
                self.prev_stance_foot_id = ref_data['stance_foot_id']
        
        # Task 1: Waist 고정 (최고 우선순위)
        if 'waist_joint_id' in ref_data:
            delta_dq, N = self._task_waist_constraint(
                ref_data['waist_joint_id'], N, dq_cmd
            )
            dq_cmd += delta_dq
        
        # Task 2: Static Contact (지지발 고정)
        if 'stance_foot_id' in ref_data:
            delta_dq, N = self._task_static_contact(
                ref_data['stance_foot_id'], N, dq_cmd
            )
            dq_cmd += delta_dq
        
        # Task 3: Floating Base Pose Tracking (몸통 자세)
        if 'torso_body_id' in ref_data and 'ref_torso_pos' in ref_data:
            delta_dq, N = self._task_floating_base(
                ref_data['torso_body_id'], 
                ref_data['ref_torso_pos'], 
                N, dq_cmd
            )
            dq_cmd += delta_dq
        
        # Task 4: Swing Leg Pose Tracking (스윙발 궤적 추종)
        if 'swing_foot_id' in ref_data and 'ref_swing_pos' in ref_data:
            delta_dq, N = self._task_swing_leg(
                ref_data['swing_foot_id'],
                ref_data['ref_swing_pos'],
                ref_data.get('ref_swing_quat', None),
                ref_data.get('ref_swing_vel', None),
                N, dq_cmd
            )
            dq_cmd += delta_dq
        
        # 적분하여 q 명령 생성
        q_cmd, dq_cmd = self.integrate_and_differentiate(dq_cmd, dt)
        
        # 디버그 로그 (매 초마다)
        if self._wbc_call_count % 500 == 0:  # 1초마다 (0.002 * 500 = 1s)
            dq_norm = np.linalg.norm(dq_cmd)
            print(f"[WBC] Kinematic: ||dq_cmd||={dq_norm:.4f}, "
                  f"swing_ref=[{ref_data.get('ref_swing_pos', [0,0,0])[0]:.3f}, "
                  f"{ref_data.get('ref_swing_pos', [0,0,0])[1]:.3f}, "
                  f"{ref_data.get('ref_swing_pos', [0,0,0])[2]:.3f}]")
        
        return q_cmd, dq_cmd
    
    # Task 메서드들 (Private) - 1차 오차 동역학
    def _task_waist_constraint(self, waist_dof_id, N_prev, dq_prev):

        if self.model is None or self.data is None:
            return np.zeros(self.nv), N_prev
        
        # 1. Jacobian 구성 (1 DoF - waist joint만 선택)
        J = np.zeros((1, self.nv))
        J[0, waist_dof_id] = 1.0
        
        # 2. 현재 상태 (qpos 사용 - 위치 제어)
        # waist_dof_id는 qvel 인덱스이므로, qpos 인덱스로 변환 필요
        waist_joint_id = None
        for jid in range(self.model.njnt):
            if self.model.jnt_dofadr[jid] == waist_dof_id:
                waist_joint_id = jid
                break
        
        if waist_joint_id is None:
            return np.zeros(self.nv), N_prev
        
        waist_qpos_id = self.model.jnt_qposadr[waist_joint_id]
        q_waist_curr = self.data.qpos[waist_qpos_id]
        
        # 3. 목표: 초기 위치 유지
        q_waist_ref = self.q0[waist_qpos_id]
        
        # 4. P 제어로 목표 속도 계산 (1차 오차 동역학)
        # 현재: kp * (dq_ref - dq_curr) → 현재 속도를 0으로 만들 뿐
        # 수정: **위치 제어**를 통해 속도 명령 생성
        kp = wbc_waist_kp
        
        dq_waist_des = kp * (q_waist_ref - q_waist_curr)
        
        # 5. Null-space 투영
        J_projected = J @ N_prev
        
        # 6. 유효 속도 오차
        eff_vel_error = dq_waist_des - (J @ dq_prev)[0]
        
        # 7. Damped Pseudo Inverse
        J_proj_pinv = J_projected.T / (J_projected @ J_projected.T + wbc_damping)
        
        # 8. 관절 속도 산출 (flatten to ensure 1D array)
        delta_dq = (J_proj_pinv * eff_vel_error).flatten()
        
        # 9. Null-space 업데이트
        N_next = N_prev @ (np.eye(self.nv) - J_proj_pinv @ J_projected)
        
        return delta_dq, N_next
    
    def _task_static_contact(self, stance_foot_id, N_prev, dq_prev):
   
        if self.model is None or self.data is None:
            return np.zeros(self.nv), N_prev
        
        import mujoco
        
        # 1. Jacobian 계산 (3 DoF - position only)
        J = np.zeros((3, self.nv))
        J_rot = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, J, J_rot, stance_foot_id)
        
        # 2. 현재 상태 (Task Space)
        x_curr = self.data.site(stance_foot_id).xpos.copy()
        
        # 3. 목표: Stance 시작 시점의 위치 유지
        if self.stance_start_pos is None:
            self.stance_start_pos = x_curr.copy()
            # stance foot은 지면에 있어야 하므로 z=0 강제
            self.stance_start_pos[2] = 0.0
        
        x_ref = self.stance_start_pos
        
        # 4. P 제어로 목표 속도 계산 (1차 오차 동역학)
        kp = wbc_contact_kp
        
        pos_err = x_ref - x_curr
        vel_des = kp * pos_err
        
        # 5. Null-space 투영
        J_projected = J @ N_prev
        
        # 6. 유효 속도 오차
        eff_vel_error = vel_des - (J @ dq_prev)
        
        # 7. Damped Pseudo Inverse
        J_proj_pinv = J_projected.T @ np.linalg.inv(
            J_projected @ J_projected.T + wbc_damping * np.eye(3)
        )
        
        # 8. 관절 속도 산출
        delta_dq = J_proj_pinv @ eff_vel_error
        
        # 9. Null-space 업데이트
        N_next = N_prev @ (np.eye(self.nv) - J_proj_pinv @ J_projected)
        
        return delta_dq, N_next
    
    def _task_floating_base(self, torso_body_id, ref_torso_pos, N_prev, dq_prev):

        if self.model is None or self.data is None:
            return np.zeros(self.nv), N_prev
        
        import mujoco
        
        # 1. Jacobian 계산 (6 DoF - position + orientation)
        J = np.zeros((6, self.nv))
        mujoco.mj_jacBody(self.model, self.data, J[:3], J[3:], torso_body_id)
        
        # 2. 현재 상태 (Task Space)
        x_curr = self.data.body(torso_body_id).xpos.copy()
        R_curr = self.data.body(torso_body_id).xmat.reshape(3, 3).copy()
        
        # 3. 목표 설정
        x_ref = ref_torso_pos
        R_ref = np.eye(3)  # upright
        
        # 4. 오차 계산
        
        # (A) 위치 오차
        pos_err = x_ref - x_curr
        
        # (B) 자세 오차 (Quaternion)
        curr_quat = np.zeros(4)
        mujoco.mju_mat2Quat(curr_quat, R_curr.flatten())
        
        ref_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ref_quat, R_ref.flatten())
        
        neg_curr_quat = np.zeros(4)
        mujoco.mju_negQuat(neg_curr_quat, curr_quat)
        
        err_quat = np.zeros(4)
        mujoco.mju_mulQuat(err_quat, ref_quat, neg_curr_quat)
        
        ori_err = np.zeros(3)
        mujoco.mju_quat2Vel(ori_err, err_quat, 1.0)
        
        # 5. P 제어로 목표 속도 계산 (1차 오차 동역학)
        kp_pos = wbc_torso_kp_pos
        kp_ori = wbc_torso_kp_ori
        
        vel_des = np.zeros(6)
        vel_des[:3] = kp_pos * pos_err
        vel_des[3:] = kp_ori * ori_err
        
        # 6. Null-space 투영
        J_projected = J @ N_prev
        
        # 7. 유효 속도 오차
        eff_vel_error = vel_des - (J @ dq_prev)
        
        # 8. Damped Pseudo Inverse
        J_proj_pinv = J_projected.T @ np.linalg.inv(
            J_projected @ J_projected.T + wbc_damping * np.eye(6)
        )
        
        # 9. 관절 속도 산출
        delta_dq = J_proj_pinv @ eff_vel_error
        
        # 10. Null-space 업데이트
        N_next = N_prev @ (np.eye(self.nv) - J_proj_pinv @ J_projected)
        
        return delta_dq, N_next
    
    def _task_swing_leg(self, swing_foot_id, ref_pos, ref_quat=None, 
                        ref_vel=None, N_prev=None, dq_prev=None):
 
        # 1. Jacobian 계산
        J = np.zeros((6, self.nv))
        if self.model is not None and self.data is not None:
            import mujoco
            mujoco.mj_jacSite(self.model, self.data, J[:3], J[3:], swing_foot_id)
        
        # 2. 현재 상태 측정 (Task Space)
        if self.model is not None and self.data is not None:
            x_curr = self.data.site(swing_foot_id).xpos.copy()
            R_curr = self.data.site(swing_foot_id).xmat.reshape(3, 3).copy()
        else:
            x_curr = np.zeros(3)
            R_curr = np.eye(3)
        
        # 3. 목표 속도 생성 (P 제어)
        
        # (A) 위치 오차
        pos_err = ref_pos - x_curr
        
        # (B) 자세 오차 (Quaternion Difference)
        if ref_quat is not None and self.model is not None and self.data is not None:
            import mujoco
            curr_quat = np.zeros(4)
            mujoco.mju_mat2Quat(curr_quat, R_curr.flatten())
            
            neg_curr_quat = np.zeros(4)
            mujoco.mju_negQuat(neg_curr_quat, curr_quat)
            
            err_quat = np.zeros(4)
            mujoco.mju_mulQuat(err_quat, ref_quat, neg_curr_quat)
            
            ori_err = np.zeros(3)
            mujoco.mju_quat2Vel(ori_err, err_quat, 1.0)
        else:
            ori_err = np.zeros(3)
        
        # (C) 피드백 속도 계산 (1차 오차 동역학)
        vel_des = np.zeros(6)
        if ref_vel is None:
            ref_vel = np.zeros(6)
        
        # P 게인
        kp_pos = wbc_swing_kp_pos
        kp_ori = wbc_swing_kp_ori
        
        # Linear + Angular
        vel_des[:3] = ref_vel[:3] + kp_pos * pos_err
        vel_des[3:] = ref_vel[3:] + kp_ori * ori_err
        
        # 4. 의사 역행렬 기반 해법
        
        # (A) 영공간 투영된 자코비안
        J_projected = J @ N_prev
        
        # (B) 유효 속도 오차
        eff_vel_error = vel_des - (J @ dq_prev)
        
        # (C) Damped Pseudo Inverse
        J_proj_pinv = J_projected.T @ np.linalg.inv(
            J_projected @ J_projected.T + wbc_damping * np.eye(6)
        )
        
        # (D) 관절 속도 산출
        delta_dq = J_proj_pinv @ eff_vel_error
        
        # 5. Null-space 업데이트
        N_next = N_prev @ (np.eye(self.nv) - J_proj_pinv @ J_projected)
        
        return delta_dq, N_next
    
    def integrate_and_differentiate(self, dq_cmd, dt):

        if self.data is not None and self.model is not None:
            import mujoco
            # MuJoCo의 적분 함수 사용 (quaternion 등을 올바르게 처리)
            q_cmd = self.data.qpos.copy()
            mujoco.mj_integratePos(self.model, q_cmd, dq_cmd, dt)
        else:
            q_cmd = np.zeros(self.model.nq if self.model else self.nv)
        
        return q_cmd, dq_cmd
    
    # ============================================
    # B. Dynamics WBC
    # ============================================
    
    def DynamicsWBC(self, q_cmd, dq_cmd, fr_left, fr_right, left_foot_id, right_foot_id, dt):

        if self.model is None or self.data is None or self.actuator_dof_ids is None:
            return np.zeros(self.nu)
        
        import mujoco
        
        # Step 1: 동역학 행렬 계산
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        C = self.data.qfrc_bias.copy()
        
        # Step 2: Selection Matrix S (Floating Base 선택)
        # Floating base는 qvel의 처음 6개 (3D position + 3D rotation)
        S = np.zeros((6, self.nv))
        S[0:6, 0:6] = np.eye(6)
        
        # Step 3: Contact Jacobian (양발)
        J_left = np.zeros((3, self.nv))
        J_right = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, J_left, None, left_foot_id)
        mujoco.mj_jacSite(self.model, self.data, J_right, None, right_foot_id)
        Jc = np.vstack([J_left, J_right])  # (6, nv)
        
        # Step 4: 명령 가속도 및 목표 힘
        ddq_cmd = (dq_cmd - self.data.qvel) / dt
        f_des = np.concatenate([fr_left, fr_right])  # (6,)
        
        # Step 5: QP 설정
        # 변수: x = [ddq (nv), f (6)]
        n_vars = self.nv + 6
        
        # 목적 함수: min 0.5 * x^T H x + g^T x
        # (ddq - ddq_cmd)^T Q (ddq - ddq_cmd) + (f - f_des)^T R (f - f_des)
        # = ddq^T Q ddq - 2*ddq_cmd^T Q ddq + ... + f^T R f - 2*f_des^T R f + ...
        
        Q_ddq = np.eye(self.nv) * qp_w_ddq  # 가속도 추종 가중치
        R_f = np.eye(6) * qp_w_f  # 힘 추종 가중치
        
        H = np.block([
            [Q_ddq, np.zeros((self.nv, 6))],
            [np.zeros((6, self.nv)), R_f]
        ])
        
        g = np.concatenate([
            -Q_ddq @ ddq_cmd,
            -R_f @ f_des
        ])
        
        # Step 6: 등식 제약 (Floating base 동역학)
        # S*M*ddq + S*C = S*Jc^T*f
        # S*M*ddq - S*Jc^T*f = -S*C
        
        A_eq = np.hstack([
            S @ M,           # (6, nv)
            -S @ Jc.T        # (6, 6)
        ])
        b_eq = -S @ C
        
        # Step 7: 부등식 제약 (마찰 원뿔)
        # 각 발에 대해:
        # - fz >= 0 (수직 방향 힘은 양수)
        # - |fx| <= mu * fz
        # - |fy| <= mu * fz
        # 
        # 선형 부등식으로 변환:
        # -fz <= 0
        # fx - mu*fz <= 0
        # -fx - mu*fz <= 0
        # fy - mu*fz <= 0
        # -fy - mu*fz <= 0
        
        mu = qp_friction_coef  # 마찰 계수
        
        # 한 발에 대한 마찰 원뿔 (5개 부등식)
        C_friction_single = np.array([
            [0, 0, -1],           # -fz <= 0
            [1, 0, -mu],          # fx - mu*fz <= 0
            [-1, 0, -mu],         # -fx - mu*fz <= 0
            [0, 1, -mu],          # fy - mu*fz <= 0
            [0, -1, -mu]          # -fy - mu*fz <= 0
        ])
        
        # 양발에 대한 마찰 원뿔 (10개 부등식)
        C_friction = np.block([
            [C_friction_single, np.zeros((5, 3))],
            [np.zeros((5, 3)), C_friction_single]
        ])
        
        # G*x <= h 형태로 변환
        # [0, C_friction] * [ddq, f]^T <= 0
        G = np.hstack([
            np.zeros((10, self.nv)),  # ddq에 대한 제약 없음
            C_friction                 # f에 대한 마찰 원뿔
        ])
        h = np.zeros(10)
        
        # Step 8: 변수 범위 (합리적인 범위 설정)
        lb = np.concatenate([
            np.full(self.nv, -qp_ddq_max),  # ddq 범위
            np.full(6, -qp_force_max)        # f 범위
        ])
        ub = np.concatenate([
            np.full(self.nv, qp_ddq_max),   # ddq 범위
            np.full(6, qp_force_max)         # f 범위
        ])
        
        # Step 9: QP 풀기 (OSQP solver)
        try:
            solution = solve_qp(
                P=H, q=g,
                A=A_eq, b=b_eq,
                G=G, h=h,
                lb=lb, ub=ub,
                solver="osqp",
                verbose=False
            )
            
            if solution is not None:
                ddq_opt = solution[:self.nv]
                f_opt = solution[self.nv:]
            else:
                print("[WARNING] QP solver failed, using commanded values")
                ddq_opt = ddq_cmd
                f_opt = f_des
        except Exception as e:
            print(f"[ERROR] QP solver exception: {e}")
            ddq_opt = ddq_cmd
            f_opt = f_des
        
        # Step 10: 최종 토크 계산 (논문 구조)
        tau_full = M @ ddq_opt + C - Jc.T @ f_opt
        tau_ff = tau_full[self.actuator_dof_ids]
        
        # f_opt 저장 (외부에서 f_actual과 비교용)
        self.last_f_opt = f_opt.copy()
        self.last_Jc = Jc.copy()
        
        # 디버그 로그 (매 초마다)
        if not hasattr(self, '_dwbc_call_count'):
            self._dwbc_call_count = 0
        self._dwbc_call_count += 1
        
        if self._dwbc_call_count % 500 == 0:
            f_left_norm = np.linalg.norm(f_opt[:3])
            f_right_norm = np.linalg.norm(f_opt[3:])
            tau_norm = np.linalg.norm(tau_ff)
            print(f"[WBC] Dynamics: F_left={f_left_norm:.1f}N, F_right={f_right_norm:.1f}N, "
                  f"||tau_ff||={tau_norm:.1f}Nm, QP={'OK' if solution is not None else 'FAIL'}")
        
        return tau_ff
