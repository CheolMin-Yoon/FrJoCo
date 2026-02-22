"""
Layer 2 MPC: JAX + daqp 기반 LIPM Model Predictive Control

★ 구조:
  - JAX: QP 행렬 사전 계산 + gradient 벡터 계산 (jit 가속)
  - daqp: 부등식 제약 QP solver (ZMP ∈ 지지 다각형)
  - 인터페이스(control_step)는 기존 Layer2와 동일

★ LIPM 이산 시스템 (1D, X/Y 독립):
  상태: x = [pos, vel]
  입력: u = ZMP 위치
  x(k+1) = A·x(k) + B·u(k)

★ 목적함수:
  min_U  J = Σ_{k=0}^{N-1} [ q_com·(com_k - com_ref_k)²
                             + q_dcm·(dcm_k - dcm_ref_k)²
                             + r_zmp·(u_k - zmp_ref_k)²
                             + s_dzmp·(u_k - u_{k-1})² ]

  Dense form: J = ½ U^T H U + f^T U
  s.t.  zmp_lb ≤ u_k ≤ zmp_ub  (지지 다각형 제약)

★ 지지 다각형 제약:
  - DSP: 양발 convex hull → 넓은 범위
  - SSP (left support):  x: foot_x ± foot_lx,  y: foot_y ± foot_ly
  - SSP (right support): x: foot_x ± foot_lx,  y: foot_y ± foot_ly
  → horizon 내 phase 변화에 따라 각 스텝별 bound가 다름
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Tuple, Optional
import daqp

from core.config import GRAVITY, DT


class MPCControl:
    """JAX + daqp 기반 LIPM MPC 컨트롤러"""

    def __init__(
        self,
        z_c: float,
        g: float = GRAVITY,
        dt: float = 0.02,           # MPC 내부 예측 dt
        ctrl_dt: float = DT,        # 실제 제어 dt
        horizon: int = 30,
        q_com: float = 100.0,
        q_dcm: float = 50.0,
        r_zmp: float = 1.0,
        s_dzmp: float = 10.0,
        # 발 크기 (지지 다각형용)
        foot_half_length: float = 0.10,   # 발 앞뒤 반폭 (x)
        foot_half_width: float = 0.05,    # 발 좌우 반폭 (y)
    ):
        self.z_c = z_c
        self.omega = np.sqrt(g / z_c)
        self.dt = dt
        self.ctrl_dt = ctrl_dt
        self.N = horizon

        self.q_com = q_com
        self.q_dcm = q_dcm
        self.r_zmp = r_zmp
        self.s_dzmp = s_dzmp

        self.foot_half_length = foot_half_length
        self.foot_half_width = foot_half_width

        # ── LIPM 이산화 (정확한 행렬 지수) ──
        w = self.omega
        T = dt
        cosh_wT = np.cosh(w * T)
        sinh_wT = np.sinh(w * T)

        A_np = np.array([
            [cosh_wT,      sinh_wT / w],
            [w * sinh_wT,  cosh_wT    ]
        ])
        B_np = np.array([
            [1.0 - cosh_wT],
            [-w * sinh_wT ]
        ])

        self.A_np = A_np
        self.B_np = B_np

        # ── QP 행렬 사전 계산 (numpy로 한 번만) ──
        self._build_qp_matrices(A_np, B_np)

        # ── JAX용 상수 변환 ──
        self._Psi_j = jnp.array(self.Psi)
        self._C_pos_j = jnp.array(self.C_pos)
        self._C_dcm_j = jnp.array(self.C_dcm_full)
        self._G_com_T_j = jnp.array(self.G_com.T)
        self._G_dcm_T_j = jnp.array(self.G_dcm.T)
        self._D_T_j = jnp.array(self.D.T)

        # ── jit 컴파일된 gradient 계산 함수 ──
        self._compute_gradient_jit = jit(self._compute_gradient_jax)

        # ── daqp 설정 ──
        # H는 고정 → numpy로 유지
        self.H_upper = np.array(self.H, dtype=np.float64, order='C')

        # 이전 ZMP
        self.prev_zmp = np.zeros(2)
        # 디버깅용
        self.dcm_error_sum = np.zeros(2)

        print(f"[MPC] JAX+daqp | horizon={horizon} | dt={dt}s | "
              f"lookahead={horizon*dt:.2f}s | ω={self.omega:.3f}")

    # ================================================================== #
    # QP 행렬 사전 계산 (numpy, 초기화 시 1회)
    # ================================================================== #
    def _build_qp_matrices(self, A, B):
        N = self.N

        # Psi (2N x 2): [A; A²; ...; A^N]
        Psi = np.zeros((2 * N, 2))
        Ak = np.eye(2)
        for k in range(N):
            Ak = A @ Ak if k > 0 else A.copy()
            Psi[2*k:2*k+2, :] = Ak

        # Phi (2N x N): 하삼각 Toeplitz
        Phi = np.zeros((2 * N, N))
        AjB = [B.copy()]
        for j in range(1, N):
            AjB.append(A @ AjB[j-1])
        for k in range(N):
            for j in range(k + 1):
                Phi[2*k:2*k+2, j:j+1] = AjB[k - j]

        # CoM 추출: C_pos (N x 2N)
        C_pos = np.zeros((N, 2 * N))
        for k in range(N):
            C_pos[k, 2*k] = 1.0

        # DCM 추출: C_dcm (N x 2N)
        C_dcm_full = np.zeros((N, 2 * N))
        for k in range(N):
            C_dcm_full[k, 2*k] = 1.0
            C_dcm_full[k, 2*k+1] = 1.0 / self.omega

        # 차분 행렬 D (N x N)
        D = np.eye(N)
        for k in range(1, N):
            D[k, k-1] = -1.0

        # 강제 응답 행렬
        G_com = C_pos @ Phi
        G_dcm = C_dcm_full @ Phi

        # Hessian (고정)
        H = (self.q_com * G_com.T @ G_com +
             self.q_dcm * G_dcm.T @ G_dcm +
             self.r_zmp * np.eye(N) +
             self.s_dzmp * D.T @ D)

        self.Psi = Psi
        self.Phi = Phi
        self.C_pos = C_pos
        self.C_dcm_full = C_dcm_full
        self.D = D
        self.G_com = G_com
        self.G_dcm = G_dcm
        self.H = H

    # ================================================================== #
    # JAX jit: gradient 벡터 계산
    # ================================================================== #
    @partial(jit, static_argnums=(0,))
    def _compute_gradient_jax(
        self,
        x0: jnp.ndarray,        # (2,)
        com_ref: jnp.ndarray,    # (N,)
        dcm_ref: jnp.ndarray,    # (N,)
        zmp_ref: jnp.ndarray,    # (N,)
        prev_zmp: float,
    ) -> jnp.ndarray:
        """gradient 벡터 f 계산 (JAX jit 가속)"""
        # 자유 응답
        X_free = self._Psi_j @ x0
        com_free = self._C_pos_j @ X_free
        dcm_free = self._C_dcm_j @ X_free

        # 오차
        e_com = com_free - com_ref
        e_dcm = dcm_free - dcm_ref

        # Δu 초기값
        d0 = jnp.zeros(self.N).at[0].set(-prev_zmp)

        # gradient
        f = (self.q_com * self._G_com_T_j @ e_com +
             self.q_dcm * self._G_dcm_T_j @ e_dcm -
             self.r_zmp * zmp_ref +
             self.s_dzmp * self._D_T_j @ d0)

        return f

    # ================================================================== #
    # 1D QP solve (daqp, 제약 포함)
    # ================================================================== #
    def _solve_1d_constrained(
        self,
        x0: np.ndarray,
        com_ref: np.ndarray,
        dcm_ref: np.ndarray,
        zmp_ref: np.ndarray,
        prev_zmp: float,
        zmp_lb: np.ndarray,       # (N,) 각 스텝별 ZMP 하한
        zmp_ub: np.ndarray,       # (N,) 각 스텝별 ZMP 상한
    ) -> float:
        """daqp로 제약 QP 풀기"""
        N = self.N

        # JAX로 gradient 계산
        f_jax = self._compute_gradient_jit(
            jnp.array(x0),
            jnp.array(com_ref),
            jnp.array(dcm_ref),
            jnp.array(zmp_ref),
            prev_zmp,
        )
        f_np = np.asarray(f_jax, dtype=np.float64)

        # ── daqp QP: min ½ x^T H x + f^T x  s.t.  lb ≤ x ≤ ub ──
        A_ineq = np.eye(N, dtype=np.float64)
        sense = np.zeros(N, dtype=np.int32)  # 0 = inequality

        result = daqp.solve(
            self.H_upper, f_np,
            A_ineq,
            np.asarray(zmp_ub, dtype=np.float64),
            np.asarray(zmp_lb, dtype=np.float64),
            sense,
        )

        # daqp.solve returns (x, fval, exitflag, info)
        x_sol, fval, exitflag, info = result

        if exitflag == 1:  # optimal
            return x_sol[0]
        else:
            # fallback: unconstrained (Cholesky)
            neg_f = -f_np
            try:
                U_opt = np.linalg.solve(self.H, neg_f)
                return float(np.clip(U_opt[0], zmp_lb[0], zmp_ub[0]))
            except np.linalg.LinAlgError:
                return float(zmp_ref[0])

    # ================================================================== #
    # 1D QP solve (unconstrained, 빠른 fallback)
    # ================================================================== #
    def _solve_1d_unconstrained(
        self,
        x0: np.ndarray,
        com_ref: np.ndarray,
        dcm_ref: np.ndarray,
        zmp_ref: np.ndarray,
        prev_zmp: float,
    ) -> float:
        """제약 없는 QP (Cholesky solve)"""
        f_jax = self._compute_gradient_jit(
            jnp.array(x0),
            jnp.array(com_ref),
            jnp.array(dcm_ref),
            jnp.array(zmp_ref),
            prev_zmp,
        )
        f_np = np.asarray(f_jax, dtype=np.float64)
        U_opt = np.linalg.solve(self.H, -f_np)
        return float(U_opt[0])

    # ================================================================== #
    # DCM 계산
    # ================================================================== #
    def calculate_current_dcm(self, com_pos: np.ndarray, com_vel: np.ndarray) -> np.ndarray:
        pos_xy = com_pos[:2] if len(com_pos) > 2 else com_pos
        vel_xy = com_vel[:2] if len(com_vel) > 2 else com_vel
        return pos_xy + vel_xy / self.omega

    # ================================================================== #
    # ZMP bounds 생성 (지지 다각형)
    # ================================================================== #
    def compute_zmp_bounds(
        self,
        axis: int,                    # 0=x, 1=y
        support_feet: np.ndarray,     # (N, 3) [phase, lf_pos, rf_pos]
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = self.N
        lb = np.zeros(N)
        ub = np.zeros(N)

        if axis == 0:
            half = self.foot_half_length
        else:
            half = self.foot_half_width

        for k in range(N):
            phase = int(support_feet[k, 0])
            lf = support_feet[k, 1]
            rf = support_feet[k, 2]

            if phase == 0:  # DSP
                lb[k] = min(lf, rf) - half
                ub[k] = max(lf, rf) + half
            elif phase == 1:  # left support
                lb[k] = lf - half
                ub[k] = lf + half
            else:  # right support
                lb[k] = rf - half
                ub[k] = rf + half

        return lb, ub

    # ================================================================== #
    # Main Interface
    # ================================================================== #
    def control_step(
        self,
        meas_com_pos: np.ndarray,
        meas_com_vel: np.ndarray,
        meas_zmp: np.ndarray,
        ref_dcm: np.ndarray,
        ref_dcm_vel: np.ndarray,
        ref_com_pos: np.ndarray,
        ref_com_vel: np.ndarray,
        ref_com_horizon: Optional[np.ndarray] = None,
        ref_dcm_horizon: Optional[np.ndarray] = None,
        ref_zmp_horizon: Optional[np.ndarray] = None,
        zmp_bounds_x: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        zmp_bounds_y: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        반환: (desired_com_vel, desired_zmp, current_dcm)
        """
        pos_xy = meas_com_pos[:2] if len(meas_com_pos) > 2 else meas_com_pos
        vel_xy = meas_com_vel[:2] if len(meas_com_vel) > 2 else meas_com_vel

        curr_dcm = self.calculate_current_dcm(meas_com_pos, meas_com_vel)

        N = self.N
        if ref_com_horizon is None:
            ref_com_horizon = np.tile(ref_com_pos[:2], (N, 1))
        if ref_dcm_horizon is None:
            ref_dcm_horizon = np.tile(ref_dcm[:2], (N, 1))
        if ref_zmp_horizon is None:
            zmp_ff = ref_dcm - ref_dcm_vel / self.omega
            ref_zmp_horizon = np.tile(zmp_ff[:2], (N, 1))

        # X, Y 독립 QP
        desired_zmp = np.zeros(2)
        use_constraints = (zmp_bounds_x is not None) or (zmp_bounds_y is not None)

        for axis in range(2):
            x0 = np.array([pos_xy[axis], vel_xy[axis]])
            com_ref_seq = ref_com_horizon[:, axis]
            dcm_ref_seq = ref_dcm_horizon[:, axis]
            zmp_ref_seq = ref_zmp_horizon[:, axis]

            if use_constraints:
                bounds = zmp_bounds_x if axis == 0 else zmp_bounds_y
                if bounds is not None:
                    lb, ub = bounds
                    optimal_zmp = self._solve_1d_constrained(
                        x0, com_ref_seq, dcm_ref_seq, zmp_ref_seq,
                        self.prev_zmp[axis], lb, ub
                    )
                else:
                    optimal_zmp = self._solve_1d_unconstrained(
                        x0, com_ref_seq, dcm_ref_seq, zmp_ref_seq,
                        self.prev_zmp[axis]
                    )
            else:
                optimal_zmp = self._solve_1d_unconstrained(
                    x0, com_ref_seq, dcm_ref_seq, zmp_ref_seq,
                    self.prev_zmp[axis]
                )

            desired_zmp[axis] = optimal_zmp

        self.prev_zmp = desired_zmp.copy()

        # desired CoM velocity
        desired_com_acc = self.omega**2 * (pos_xy - desired_zmp)
        desired_com_vel = ref_com_vel[:2] + desired_com_acc * self.ctrl_dt

        com_pos_err = ref_com_pos[:2] - pos_xy
        desired_com_vel += 1.0 * com_pos_err

        return desired_com_vel, desired_zmp, curr_dcm
