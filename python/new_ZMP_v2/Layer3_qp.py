"""
Layer3_qp.py — Task-Space Inverse Dynamics WBC (단일 QP)

논문 Section B. Task Space Inverse Dynamics Control + C. Whole-Body Control Formulation

mink IK 없이, 가속도 레벨에서 직접 QP를 풀어 feedforward 토크를 생성한다.

QP 변수: x = [ddq (nv), f_c (n_contact)]

목적함수 (식 6):
  min  Σ_i  w_i * || ẍ_{i,d} - (J_i * ddq + dJ_i * dq) ||²

  여기서 ẍ_{i,d} = ẍ_{i,ref} + K_d(ẋ_{i,ref} - ẋ_i) + K_p(x_{i,ref} - x_i)  (식 4)

등식 제약:
  (6a) M*ddq + C = S^T*tau + J_c^T*f_c
       → floating base 행: S*M*ddq + S*C = S*J_c^T*f_c  (S = [I_6, 0])
  (6b) J_c*ddq + dJ_c*dq = 0  (접촉 가속도 = 0)

부등식 제약:
  (6c) 마찰 원뿔 (linearized friction cone)
  (6d) 토크 한계 (ddq 범위로 근사)
"""

import numpy as np
import mujoco
from qpsolvers import solve_qp

from config import (
    qp_w_ddq, qp_w_f,
    qp_friction_coef,
    qp_ddq_max, qp_force_max,
)


# ============================================================
# Task 게인 (식 4의 K_p, K_d)
# ============================================================
# Torso (floating base) task
TORSO_KP_POS = 1000.0
TORSO_KP_ORI = 1000.0
TORSO_KD_POS = 20.0
TORSO_KD_ORI = 20.0

# Swing foot task
SWING_KP_POS = 200.0
SWING_KP_ORI = 50.0
SWING_KD_POS = 30.0
SWING_KD_ORI = 10.0

# QP task 가중치 (식 6의 W_i)
W_TORSO_POS = 1000.0
W_TORSO_ORI = 1000.0
W_SWING_POS = 50.0
W_SWING_ORI = 1.0
W_REGULARIZE = 0.001   # ddq regularization


class TaskSpaceWBC:
    """Task-Space Inverse Dynamics WBC — 단일 QP로 feedforward 토크 생성."""

    def __init__(self, nv, nu, model, data, actuator_dof_ids):
        self.nv = nv
        self.nu = nu
        self.model = model
        self.data = data
        self.actuator_dof_ids = actuator_dof_ids

        # 이전 스텝 task-space 속도 (dJ*dq 수치 근사용)
        self._prev_torso_vel = np.zeros(6)
        self._prev_swing_vel = np.zeros(6)
        
        # 이전 프레임 접촉 자코비안 (dJ*dq 수치 미분용)
        self._prev_Jc = None
        self._prev_dt = None

    # ==========================================================
    # 유틸: orientation error (rotation matrix → angular error)
    # ==========================================================
    @staticmethod
    def _ori_error(R_ref, R_curr):
        """SO(3) 오차를 axis-angle 벡터로 반환."""
        R_err = R_ref @ R_curr.T
        # Rodrigues: angle*axis from rotation matrix
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        if abs(angle) < 1e-8:
            return np.zeros(3)
        axis = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]) / (2 * np.sin(angle))
        return angle * axis

    # ==========================================================
    # 메인: compute_torque
    # ==========================================================
    def compute_torque(
        self,
        # 접촉 정보
        fr_left: np.ndarray,       # (3,) desired left foot force
        fr_right: np.ndarray,      # (3,) desired right foot force
        left_foot_site_id: int,
        right_foot_site_id: int,
        # Torso task
        torso_body_id: int,
        ref_torso_pos: np.ndarray,  # (3,)
        ref_torso_vel: np.ndarray = None,  # (6,) [lin, ang], optional
        # Swing task
        swing_foot_site_id: int = -1,
        ref_swing_pos: np.ndarray = None,  # (3,)
        ref_swing_vel: np.ndarray = None,  # (6,) [lin, ang], optional
        ref_swing_acc: np.ndarray = None,  # (6,) [lin, ang], optional (해석적 가속도)
        # 기타
        dt: float = 0.002,
    ) -> np.ndarray:
        """QP를 풀어 actuated joint feedforward 토크 (nu,)를 반환."""

        nv = self.nv
        model, data = self.model, self.data

        # ── 1. 동역학 행렬 ──
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(model, M, data.qM)
        C = data.qfrc_bias.copy()  # C*dq + g

        # ── 2. 접촉 자코비안 ──
        J_lf = np.zeros((3, nv))
        J_rf = np.zeros((3, nv))
        mujoco.mj_jacSite(model, data, J_lf, None, left_foot_site_id)
        mujoco.mj_jacSite(model, data, J_rf, None, right_foot_site_id)

        left_contact = fr_left[2] > 1.0
        right_contact = fr_right[2] > 1.0

        # 접촉 자코비안 스택 (contact 발만)
        Jc_list = []
        fc_des_list = []
        if left_contact:
            Jc_list.append(J_lf)
            fc_des_list.append(fr_left)
        if right_contact:
            Jc_list.append(J_rf)
            fc_des_list.append(fr_right)

        if len(Jc_list) == 0:
            # 공중 — 토크 0
            return np.zeros(self.nu)

        Jc = np.vstack(Jc_list)          # (3*n_contact, nv)
        n_c = Jc.shape[0]
        f_des = np.concatenate(fc_des_list)  # (3*n_contact,)

        # dJc*dq 수치 근사: (Jc - Jc_prev)/dt * dq
        dq = data.qvel.copy()
        if self._prev_Jc is not None and self._prev_Jc.shape == Jc.shape and self._prev_dt is not None:
            dJc = (Jc - self._prev_Jc) / self._prev_dt
            dJc_dq = dJc @ dq
        else:
            dJc_dq = np.zeros(n_c)
        self._prev_Jc = Jc.copy()
        self._prev_dt = dt

        # ── 3. Task-space 가속도 목표 (식 4) ──
        # 각 task: J_i, desired_ddx_i, weight_i
        tasks = []  # list of (J_i, ddx_des_i, W_i)

        # (A) Torso task (6 DoF: position + orientation)
        J_torso = np.zeros((6, nv))
        mujoco.mj_jacBody(model, data, J_torso[:3], J_torso[3:], torso_body_id)

        torso_pos = data.body(torso_body_id).xpos.copy()
        torso_mat = data.body(torso_body_id).xmat.reshape(3, 3)
        R_ref_torso = np.eye(3)  # upright

        # 현재 task-space 속도
        torso_vel = J_torso @ data.qvel  # (6,)

        # 레퍼런스 속도/가속도
        if ref_torso_vel is None:
            ref_torso_vel = np.zeros(6)
        # 가속도 레퍼런스: 수치 미분
        torso_ddx_ref = (ref_torso_vel - self._prev_torso_vel) / dt
        self._prev_torso_vel = ref_torso_vel.copy()

        # 식 (4): ẍ_d = ẍ_ref + K_d*(ẋ_ref - ẋ) + K_p*(x_ref - x)
        pos_err_torso = ref_torso_pos - torso_pos
        ori_err_torso = self._ori_error(R_ref_torso, torso_mat)

        ddx_des_torso = np.zeros(6)
        ddx_des_torso[:3] = (torso_ddx_ref[:3]
                             + TORSO_KD_POS * (ref_torso_vel[:3] - torso_vel[:3])
                             + TORSO_KP_POS * pos_err_torso)
        ddx_des_torso[3:] = (torso_ddx_ref[3:]
                             + TORSO_KD_ORI * (ref_torso_vel[3:] - torso_vel[3:])
                             + TORSO_KP_ORI * ori_err_torso)

        W_torso = np.diag([W_TORSO_POS]*3 + [W_TORSO_ORI]*3)
        tasks.append((J_torso, ddx_des_torso, W_torso))

        # (B) Swing foot task (SSP일 때만)
        if swing_foot_site_id >= 0 and ref_swing_pos is not None:
            J_swing = np.zeros((6, nv))
            mujoco.mj_jacSite(model, data, J_swing[:3], J_swing[3:], swing_foot_site_id)

            swing_pos = data.site(swing_foot_site_id).xpos.copy()
            swing_mat = data.site(swing_foot_site_id).xmat.reshape(3, 3)

            swing_vel = J_swing @ data.qvel
            if ref_swing_vel is None:
                ref_swing_vel = np.zeros(6)
            
            # 해석적 가속도 사용 (수치 미분 제거)
            if ref_swing_acc is not None:
                swing_ddx_ref = ref_swing_acc
            else:
                swing_ddx_ref = (ref_swing_vel - self._prev_swing_vel) / dt
            self._prev_swing_vel = ref_swing_vel.copy()

            pos_err_swing = ref_swing_pos - swing_pos
            ori_err_swing = self._ori_error(np.eye(3), swing_mat)

            ddx_des_swing = np.zeros(6)
            ddx_des_swing[:3] = (swing_ddx_ref[:3]
                                 + SWING_KD_POS * (ref_swing_vel[:3] - swing_vel[:3])
                                 + SWING_KP_POS * pos_err_swing)
            ddx_des_swing[3:] = (swing_ddx_ref[3:]
                                 + SWING_KD_ORI * (ref_swing_vel[3:] - swing_vel[3:])
                                 + SWING_KP_ORI * ori_err_swing)

            W_swing = np.diag([W_SWING_POS]*3 + [W_SWING_ORI]*3)
            tasks.append((J_swing, ddx_des_swing, W_swing))

        # ── 4. QP 구성 ──
        # 변수: x = [ddq (nv), f_c (n_c)]
        n_vars = nv + n_c

        # 목적함수: Σ_i || W_i^{1/2} (ddx_des_i - J_i*ddq) ||²  + w_reg*||ddq||²
        # = Σ_i (ddx_i - J_i*ddq)^T W_i (ddx_i - J_i*ddq)
        # 전개: ddq^T (Σ J_i^T W_i J_i) ddq - 2*(Σ ddx_i^T W_i J_i) ddq + const
        # + w_f * ||f - f_des||²

        H_ddq = np.zeros((nv, nv))
        g_ddq = np.zeros(nv)
        for J_i, ddx_i, W_i in tasks:
            H_ddq += J_i.T @ W_i @ J_i
            g_ddq -= J_i.T @ W_i @ ddx_i
        # regularization
        H_ddq += W_REGULARIZE * np.eye(nv)

        # 접촉력 추종: w_f * ||f - f_des||²
        R_f = qp_w_f * np.eye(n_c)

        H = np.block([
            [H_ddq,                    np.zeros((nv, n_c))],
            [np.zeros((n_c, nv)),      R_f                ],
        ])
        g_vec = np.concatenate([g_ddq, -R_f @ f_des])

        # ── 5. 등식 제약 ──
        # (6a) Floating base dynamics: S*M*ddq - S*Jc^T*f = -S*C
        S = np.zeros((6, nv))
        S[:6, :6] = np.eye(6)

        A_dyn = np.hstack([S @ M, -S @ Jc.T])
        b_dyn = -S @ C

        # (6b) Contact acceleration = 0: Jc*ddq = -dJc*dq
        A_contact = np.hstack([Jc, np.zeros((n_c, n_c))])
        b_contact = -dJc_dq

        A_eq = np.vstack([A_dyn, A_contact])
        b_eq = np.concatenate([b_dyn, b_contact])

        # ── 6. 부등식 제약: 마찰 원뿔 ──
        mu = qp_friction_coef
        C_fric_single = np.array([
            [0,  0, -1],
            [1,  0, -mu],
            [-1, 0, -mu],
            [0,  1, -mu],
            [0, -1, -mu],
        ])

        fric_rows = []
        h_rows = []
        col_offset = 0
        for i in range(len(Jc_list)):
            blk = np.zeros((5, n_c))
            blk[:, col_offset:col_offset+3] = C_fric_single
            fric_rows.append(blk)
            h_rows.extend([0.0]*5)
            col_offset += 3

        C_fric_all = np.vstack(fric_rows)
        n_ineq = C_fric_all.shape[0]
        G = np.hstack([np.zeros((n_ineq, nv)), C_fric_all])
        h = np.array(h_rows)

        # ── 7. 변수 범위 ──
        lb = np.concatenate([np.full(nv, -qp_ddq_max), np.full(n_c, -qp_force_max)])
        ub = np.concatenate([np.full(nv,  qp_ddq_max), np.full(n_c,  qp_force_max)])

        # ── 8. QP 풀기 ──
        try:
            sol = solve_qp(
                P=H, q=g_vec,
                A=A_eq, b=b_eq,
                G=G, h=h,
                lb=lb, ub=ub,
                solver="osqp",
                verbose=False,
            )
            if sol is not None:
                ddq_opt = sol[:nv]
                f_opt = sol[nv:]
            else:
                ddq_opt = np.zeros(nv)
                f_opt = f_des
        except Exception as e:
            print(f"[QP ERROR] {e}")
            ddq_opt = np.zeros(nv)
            f_opt = f_des

        # ── 9. 역동역학 → 토크 ──
        # tau_full = M*ddq + C - Jc^T*f
        tau_full = M @ ddq_opt + C - Jc.T @ f_opt
        tau_ff = tau_full[self.actuator_dof_ids]

        return tau_ff
