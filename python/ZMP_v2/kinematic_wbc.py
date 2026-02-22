"""
Kinematic WBC — ctrl(lib_ZMPctrl.py)의 numik을 최대한 충실히 재현

ctrl numik (delt < 1 분기):
  1순위 (eqnJ1): CoM(3) + 양발 위치(6) + 양발 방향(6) = 15 DoF
  2순위 (eqnJ2): Hip 방향(3) + 상체 잠금(len(ubjnts))
  → np.linalg.pinv 사용
  → dq = dqN1 + pinv(J2 @ N1) @ (delx2 - J2 @ dqN1)
  → mj_integratePos(q, dq, delt) 1회 → 리턴

ZAM은 Avec에 포함되지만 eqnJ2에 포함 안 됨 → 실질적으로 미사용.
"""

import numpy as np
import mujoco


class KinematicWBC:
    """ctrl numik 재현 — 1-step IK."""

    def __init__(self, model, data, actuator_dof_ids,
                 max_iters=10, tol=1e-6, damping=1e-6):
        self.model = model
        self.data = data
        self.nv = model.nv
        self.nq = model.nq
        self.actuator_dof_ids = actuator_dof_ids

        # site IDs 캐시
        self.lf_id = model.site("left_foot_site").id
        self.rf_id = model.site("right_foot_site").id

    def solve(self, ref_com, ref_lf, ref_rf,
              upper_body_dof_ids=None, q0_euler=None):
        """ctrl numik (delt<1) 재현.

        Args:
            ref_com: (3,) 목표 CoM
            ref_lf: (3,) 목표 왼발
            ref_rf: (3,) 목표 오른발
            upper_body_dof_ids: 상체 DoF IDs (ctrl의 ubjnts)

        Returns:
            q_new: (nq,) 새 qpos
            dq: (nv,) 관절 속도
        """
        model, data = self.model, self.data
        dt = model.opt.timestep

        mujoco.mj_fwdPosition(model, data)

        # ── 현재 상태 읽기 ──
        ocmi = data.subtree_com[0].copy()
        olefti = data.site(self.lf_id).xpos.copy()
        orighti = data.site(self.rf_id).xpos.copy()

        # ── 방향 오차 ──
        err_ori_hip = self._quat_error_from_quat(
            np.array([1, 0, 0, 0]), data.qpos[3:7].copy())
        err_ori_left = self._quat_error_from_xmat(
            data.site(self.lf_id).xmat)
        err_ori_right = self._quat_error_from_xmat(
            data.site(self.rf_id).xmat)

        # ── 자코비안 ──
        Jcm = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeCom(model, data, Jcm, 0)

        Jwb = np.zeros((3, self.nv))
        Jwb[0:3, 3:6] = np.eye(3)

        Jvleft = np.zeros((3, self.nv))
        Jwleft = np.zeros((3, self.nv))
        mujoco.mj_jacSite(model, data, Jvleft, Jwleft, self.lf_id)

        Jvright = np.zeros((3, self.nv))
        Jwright = np.zeros((3, self.nv))
        mujoco.mj_jacSite(model, data, Jvright, Jwright, self.rf_id)

        # ── Avec, bvec 구성 (ctrl과 동일) ──
        n_ub = len(upper_body_dof_ids) if upper_body_dof_ids is not None else 0
        total_rows = 18 + n_ub + 3  # 18(기본) + ub + ZAM(3)

        Avec = np.zeros((total_rows, self.nv))
        bvec = np.zeros(total_rows)

        # CoM (0:3)
        Avec[0:3] = Jcm
        bvec[0:3] = ref_com - ocmi

        # Hip orient (3:6)
        Avec[3:6] = Jwb
        bvec[3:6] = err_ori_hip

        # Left foot pos (6:9)
        Avec[6:9] = Jvleft
        bvec[6:9] = ref_lf - olefti

        # Left foot ori (9:12)
        Avec[9:12] = Jwleft
        bvec[9:12] = err_ori_left

        # Right foot pos (12:15)
        Avec[12:15] = Jvright
        bvec[12:15] = ref_rf - orighti

        # Right foot ori (15:18)
        Avec[15:18] = Jwright
        bvec[15:18] = err_ori_right

        # Upper body lock (18:18+n_ub)
        if n_ub > 0:
            Jub = np.zeros((n_ub, self.nv))
            for i, dof in enumerate(upper_body_dof_ids):
                Jub[i, dof] = 1.0
            Avec[18:18+n_ub] = Jub
            # ctrl: bvec = (qref[ubjnts] - q0[ubjnts]) / 1, qref=0
            # q0[ubjnts]는 현재 관절 값 → 0으로 되돌리려는 힘
            # 여기서는 상체를 현재 위치에 유지 (변화 0)
            bvec[18:18+n_ub] = 0.0

        # ZAM (18+n_ub:18+n_ub+3)
        Iwb = np.zeros((3, self.nv))
        mujoco.mj_angmomMat(model, data, Iwb, 0)
        Avec[18+n_ub:18+n_ub+3] = Iwb
        bvec[18+n_ub:18+n_ub+3] = 0.0

        # ── 1순위: CoM(0,1,2) + 양발 위치+방향(6~17) = 15 ──
        eqnJ1 = np.concatenate([np.array([0, 1, 2]), np.arange(6, 18)])

        J1 = Avec[eqnJ1].copy()
        delx1 = bvec[eqnJ1].copy() / dt

        dqN1 = np.linalg.pinv(J1) @ delx1
        InJ1 = np.eye(self.nv) - np.linalg.pinv(J1) @ J1

        # ── 2순위: Hip(3,4,5) + 상체잠금(18:18+n_ub) ──
        # ctrl: eqnJ2 = [3,4,5] + [18:18+len(ubjnts)]  (ZAM 미포함)
        eqnJ2 = np.concatenate([np.array([3, 4, 5]), np.arange(18, 18+n_ub)])

        J2 = Avec[eqnJ2].copy()
        delx2 = bvec[eqnJ2].copy() / dt

        Jt2 = J2 @ InJ1
        dq = dqN1 + np.linalg.pinv(Jt2) @ (delx2 - J2 @ dqN1)

        # ── 적분 (ctrl: mj_integratePos 1회) ──
        q_new = data.qpos.copy()
        mujoco.mj_integratePos(model, q_new, dq, dt)
        data.qpos[:] = q_new
        mujoco.mj_fwdPosition(model, data)

        return data.qpos.copy(), dq

    def _quat_error_from_quat(self, quat_des, quat_curr):
        """quaternion 오차 → axis-angle (3,)."""
        neg_curr = np.zeros(4)
        mujoco.mju_negQuat(neg_curr, quat_curr)
        err_quat = np.zeros(4)
        mujoco.mju_mulQuat(err_quat, quat_des, neg_curr)
        err_vel = np.zeros(3)
        mujoco.mju_quat2Vel(err_vel, err_quat, 1.0)
        return err_vel

    def _quat_error_from_xmat(self, xmat_flat):
        """site xmat → identity와의 방향 오차 (3,)."""
        curr_quat = np.zeros(4)
        mujoco.mju_mat2Quat(curr_quat, xmat_flat)
        return self._quat_error_from_quat(np.array([1, 0, 0, 0]), curr_quat)

    def compute_total_error(self, ref_com, ref_lf, ref_rf):
        """디버그용 총 오차."""
        data = self.data
        com_pos = data.subtree_com[0].copy()
        lf_pos = data.site(self.lf_id).xpos.copy()
        rf_pos = data.site(self.rf_id).xpos.copy()
        lf_ori = self._quat_error_from_xmat(data.site(self.lf_id).xmat)
        rf_ori = self._quat_error_from_xmat(data.site(self.rf_id).xmat)
        return (np.linalg.norm(ref_com - com_pos)
                + np.linalg.norm(ref_lf - lf_pos)
                + np.linalg.norm(lf_ori)
                + np.linalg.norm(ref_rf - rf_pos)
                + np.linalg.norm(rf_ori))
