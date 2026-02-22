"""
Layer 3: Task-Space Inverse Dynamics WBC (단일 QP)

DCM 플래너(Layer1)의 오프라인 궤적 + DCM PI(Layer2)의 피드백을
받아서 역동역학 QP로 feedforward 토크를 생성한다.

mink IK 제거 → 가속도 레벨 QP 직접 풀기.
"""

import numpy as np
import mujoco
from qpsolvers import solve_qp

from config import (
    QP_W_DDQ, QP_W_F,
    QP_FRICTION_COEF,
    QP_DDQ_MAX, QP_FORCE_MAX,
    TORSO_KP_POS, TORSO_KD_POS,
    TORSO_KP_ORI, TORSO_KD_ORI,
    SWING_KP_POS, SWING_KD_POS,
    SWING_KP_ORI, SWING_KD_ORI,
    W_TORSO_POS, W_TORSO_ORI,
    W_SWING_POS, W_SWING_ORI,
)
W_REGULARIZE = 0.01


class TaskSpaceWBC:
    """Task-Space Inverse Dynamics WBC — 단일 QP로 feedforward 토크 생성."""

    def __init__(self, nv, nu, model, data, actuator_dof_ids):
        self.nv = nv
        self.nu = nu
        self.model = model
        self.data = data
        self.actuator_dof_ids = actuator_dof_ids

        self._prev_torso_vel = np.zeros(6)
        self._prev_swing_vel = np.zeros(6)

    @staticmethod
    def _ori_error(R_ref, R_curr):
        """SO(3) 오차 → axis-angle 벡터."""
        R_err = R_ref @ R_curr.T
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        if abs(angle) < 1e-8:
            return np.zeros(3)
        axis = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]) / (2 * np.sin(angle))
        return angle * axis

    def compute_torque(
        self,
        fr_left, fr_right,
        left_foot_site_id, right_foot_site_id,
        torso_body_id, ref_torso_pos,
        ref_torso_vel=None,
        swing_foot_site_id=-1,
        ref_swing_pos=None,
        ref_swing_vel=None,
        dt=0.002,
    ):
        """QP를 풀어 actuated joint feedforward 토크 (nu,)를 반환."""
        nv = self.nv
        model, data = self.model, self.data

        # 1. 동역학 행렬
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(model, M, data.qM)
        C = data.qfrc_bias.copy()

        # 2. 접촉 자코비안
        J_lf = np.zeros((3, nv))
        J_rf = np.zeros((3, nv))
        mujoco.mj_jacSite(model, data, J_lf, None, left_foot_site_id)
        mujoco.mj_jacSite(model, data, J_rf, None, right_foot_site_id)

        left_contact = fr_left[2] > 1.0
        right_contact = fr_right[2] > 1.0

        Jc_list, fc_des_list = [], []
        if left_contact:
            Jc_list.append(J_lf)
            fc_des_list.append(fr_left)
        if right_contact:
            Jc_list.append(J_rf)
            fc_des_list.append(fr_right)

        if len(Jc_list) == 0:
            return np.zeros(self.nu)

        Jc = np.vstack(Jc_list)
        n_c = Jc.shape[0]
        f_des = np.concatenate(fc_des_list)
        dJc_dq = np.zeros(n_c)

        # 3. Task-space 가속도 목표
        tasks = []

        # (A) Torso task
        J_torso = np.zeros((6, nv))
        mujoco.mj_jacBody(model, data, J_torso[:3], J_torso[3:], torso_body_id)

        torso_pos = data.body(torso_body_id).xpos.copy()
        torso_mat = data.body(torso_body_id).xmat.reshape(3, 3)
        torso_vel = J_torso @ data.qvel

        if ref_torso_vel is None:
            ref_torso_vel = np.zeros(6)
        torso_ddx_ref = (ref_torso_vel - self._prev_torso_vel) / dt
        self._prev_torso_vel = ref_torso_vel.copy()

        pos_err = ref_torso_pos - torso_pos
        ori_err = self._ori_error(np.eye(3), torso_mat)

        ddx_torso = np.zeros(6)
        ddx_torso[:3] = torso_ddx_ref[:3] + TORSO_KD_POS * (ref_torso_vel[:3] - torso_vel[:3]) + TORSO_KP_POS * pos_err
        ddx_torso[3:] = torso_ddx_ref[3:] + TORSO_KD_ORI * (ref_torso_vel[3:] - torso_vel[3:]) + TORSO_KP_ORI * ori_err

        tasks.append((J_torso, ddx_torso, np.diag([W_TORSO_POS]*3 + [W_TORSO_ORI]*3)))

        # (B) Swing foot task
        if swing_foot_site_id >= 0 and ref_swing_pos is not None:
            J_sw = np.zeros((6, nv))
            mujoco.mj_jacSite(model, data, J_sw[:3], J_sw[3:], swing_foot_site_id)

            sw_pos = data.site(swing_foot_site_id).xpos.copy()
            sw_mat = data.site(swing_foot_site_id).xmat.reshape(3, 3)
            sw_vel = J_sw @ data.qvel

            if ref_swing_vel is None:
                ref_swing_vel = np.zeros(6)
            sw_ddx_ref = (ref_swing_vel - self._prev_swing_vel) / dt
            self._prev_swing_vel = ref_swing_vel.copy()

            ddx_sw = np.zeros(6)
            ddx_sw[:3] = sw_ddx_ref[:3] + SWING_KD_POS * (ref_swing_vel[:3] - sw_vel[:3]) + SWING_KP_POS * (ref_swing_pos - sw_pos)
            ddx_sw[3:] = sw_ddx_ref[3:] + SWING_KD_ORI * (ref_swing_vel[3:] - sw_vel[3:]) + SWING_KP_ORI * self._ori_error(np.eye(3), sw_mat)

            tasks.append((J_sw, ddx_sw, np.diag([W_SWING_POS]*3 + [W_SWING_ORI]*3)))

        # 4. QP 구성
        n_vars = nv + n_c

        H_ddq = W_REGULARIZE * np.eye(nv)
        g_ddq = np.zeros(nv)
        for J_i, ddx_i, W_i in tasks:
            H_ddq += J_i.T @ W_i @ J_i
            g_ddq -= J_i.T @ W_i @ ddx_i

        R_f = QP_W_F * np.eye(n_c)
        H = np.block([
            [H_ddq,                np.zeros((nv, n_c))],
            [np.zeros((n_c, nv)),  R_f                ],
        ])
        g_vec = np.concatenate([g_ddq, -R_f @ f_des])

        # 5. 등식 제약
        S = np.zeros((6, nv))
        S[:6, :6] = np.eye(6)

        A_eq = np.vstack([
            np.hstack([S @ M, -S @ Jc.T]),
            np.hstack([Jc, np.zeros((n_c, n_c))]),
        ])
        b_eq = np.concatenate([-S @ C, -dJc_dq])

        # 6. 마찰 원뿔
        mu = QP_FRICTION_COEF
        C_fric = np.array([
            [0, 0, -1], [1, 0, -mu], [-1, 0, -mu], [0, 1, -mu], [0, -1, -mu],
        ])
        fric_rows, h_rows = [], []
        col_off = 0
        for _ in range(len(Jc_list)):
            blk = np.zeros((5, n_c))
            blk[:, col_off:col_off+3] = C_fric
            fric_rows.append(blk)
            h_rows.extend([0.0]*5)
            col_off += 3

        G = np.hstack([np.zeros((len(h_rows), nv)), np.vstack(fric_rows)])
        h = np.array(h_rows)

        lb = np.concatenate([np.full(nv, -QP_DDQ_MAX), np.full(n_c, -QP_FORCE_MAX)])
        ub = np.concatenate([np.full(nv,  QP_DDQ_MAX), np.full(n_c,  QP_FORCE_MAX)])

        # 7. QP 풀기
        try:
            sol = solve_qp(P=H, q=g_vec, A=A_eq, b=b_eq, G=G, h=h,
                           lb=lb, ub=ub, solver="osqp", verbose=False)
            if sol is not None:
                ddq_opt, f_opt = sol[:nv], sol[nv:]
            else:
                ddq_opt, f_opt = np.zeros(nv), f_des
        except Exception as e:
            print(f"[QP ERROR] {e}")
            ddq_opt, f_opt = np.zeros(nv), f_des

        # 8. 역동역학 → 토크
        tau_full = M @ ddq_opt + C - Jc.T @ f_opt
        return tau_full[self.actuator_dof_ids]
