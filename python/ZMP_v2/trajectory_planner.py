"""
Layer1: ZMP Preview Control 기반 오프라인 궤적 생성

Kajita 2003 ZMP preview control (lib_ZMPctrl.py mpc2humn 방식):
  - LIPM 이산 상태공간 + Riccati → preview gain
  - x/y 독립 preview 제어 → CoM 궤적
  - footstep 시퀀스 → ZMP 레퍼런스

발 궤적:
  - Cycloid XY + Bezier Z (이착지 속도 0 보장)
  - DSP 구간: 양발 고정
  - SSP 구간: stance 고정, swing은 Cycloid+Bezier

lib_ZMPctrl.py의 연속성 조건:
  - CubicSpline(bc_type='clamped') → 시작/끝 속도 0
  - Cycloid: θ-sin(θ) → p=0,1에서 자동으로 속도 0
  - 여기서는 Bezier Z도 이착지 속도 0이 되도록 제어점 배치
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from typing import List, Tuple


class TrajectoryPlanner:
    """ZMP Preview Control 기반 오프라인 궤적 생성기."""

    def __init__(self, z_c, dt=0.002, step_time=0.7, dsp_ratio=0.1,
                 step_height=0.08, preview_horizon=1000):
        self.z_c = z_c              # CoM 높이 (LIPM)
        self.dt = dt
        self.step_time = step_time
        self.dsp_ratio = dsp_ratio  # DSP 비율 (0~1)
        self.step_height = step_height
        self.N = preview_horizon    # preview horizon (samples)
        self.g = 9.81
        self.omega = np.sqrt(self.g / self.z_c)

        # DSP/SSP 시간
        self.dsp_time = step_time * dsp_ratio
        self.ssp_time = step_time - self.dsp_time
        self.samples_per_step = int(round(step_time / dt))
        self.dsp_samples = int(round(self.dsp_time / dt))
        self.ssp_samples = self.samples_per_step - self.dsp_samples

        # Preview gain 사전 계산
        self._compute_preview_gains()

    def _compute_preview_gains(self):
        """Kajita 2003: LIPM 이산 상태공간 + Riccati → preview gain."""
        dt, h, g = self.dt, self.z_c, self.g

        # 이산 LIPM 상태공간: x = [pos, vel, acc]
        A = np.array([[1, dt, dt**2 / 2],
                      [0, 1, dt],
                      [0, 0, 1]])
        B = np.array([[dt**3 / 6],
                      [dt**2 / 2],
                      [dt]])
        C = np.array([[1, 0, -h / g]])

        # 확장 시스템 (적분기 포함)
        nx = A.shape[0]
        A_aug = np.vstack([
            np.hstack([np.eye(1), C @ A]),
            np.hstack([np.zeros((nx, 1)), A])
        ])
        B_aug = np.vstack([C @ B, B])

        # 비용 가중치
        Qe = 1.0
        R = 1e-6
        Q = np.zeros((nx + 1, nx + 1))
        Q[0, 0] = Qe

        # Riccati
        P = solve_discrete_are(A_aug, B_aug, Q, R)
        K = np.linalg.inv(B_aug.T @ P @ B_aug + R) @ (B_aug.T @ P @ A_aug)

        self._Gi = K[0, 0]          # 적분 게인
        self._Gx = K[0, 1:]         # 상태 피드백 게인

        # Preview gain
        AcBK = A_aug - B_aug @ K
        X = -AcBK.T @ P @ np.array([[1], [0], [0], [0]])
        inv_term = np.linalg.inv(B_aug.T @ P @ B_aug + R)
        self._Gd = np.zeros(self.N)
        for i in range(self.N):
            self._Gd[i] = (inv_term @ (B_aug.T @ X)).item()
            X = AcBK.T @ X

        # 상태공간 행렬 저장
        self._A = A
        self._B = B
        self._C = C

    # ================================================================== #
    # Footstep Planning
    # ================================================================== #
    def plan_footsteps(self, n_steps, step_length, step_width,
                       init_lf_xy, init_rf_xy):
        """좌우 교대 footstep 시퀀스 생성.
        
        ctrl 방식: 첫 스텝은 초기 DSP (양발 고정, CoM만 이동)
        
        Returns:
            footsteps: list of (x, y) — ZMP 레퍼런스용
            foot_targets: list of (x, y) — 실제 발 착지 위치
        """
        footsteps = []   # ZMP ref (발 중심)
        foot_targets = []  # 실제 발 착지 위치

        # 첫 스텝: 초기 DSP (양발 중심)
        # ctrl: spno=2로 시작, 첫 step_time 동안 양발 고정
        init_center_x = (init_lf_xy[0] + init_rf_xy[0]) / 2
        init_center_y = (init_lf_xy[1] + init_rf_xy[1]) / 2
        footsteps.append((init_center_x, init_center_y))
        foot_targets.append(None)  # 첫 스텝은 발 이동 없음

        # 이후 스텝: 좌우 교대
        for i in range(1, n_steps):
            x = init_lf_xy[0] + (i - 1) * step_length
            if i % 2 == 1:  # 홀수: 왼발
                y = init_lf_xy[1]
            else:  # 짝수: 오른발
                y = init_rf_xy[1]
            footsteps.append((x, y))
            foot_targets.append((x, y))

        return footsteps, foot_targets

    # ================================================================== #
    # ZMP Preview Control → CoM 궤적
    # ================================================================== #
    def compute_com_trajectory(self, zmp_ref, init_pos, init_vel=0.0):
        """1축 ZMP preview control → CoM 궤적.

        Args:
            zmp_ref: (M,) ZMP 레퍼런스 (1축)
            init_pos: 초기 CoM 위치 (스칼라)
            init_vel: 초기 CoM 속도 (스칼라)

        Returns:
            com_pos: (L,) CoM 위치
            com_vel: (L,) CoM 속도
            zmp_actual: (L,) 실제 ZMP 출력
        """
        A, B, C = self._A, self._B, self._C
        Gi, Gx, Gd, N = self._Gi, self._Gx, self._Gd, self.N

        M = len(zmp_ref)
        L = M - N  # preview 가능한 길이
        if L <= 0:
            raise ValueError(f"ZMP ref ({M}) must be longer than preview horizon ({N})")

        x = np.array([[init_pos], [init_vel], [0.0]])
        com_pos = np.zeros(L)
        com_vel = np.zeros(L)
        zmp_actual = np.zeros(L)
        e_sum = 0.0

        for k in range(L):
            p = (C @ x)[0, 0]
            e = p - zmp_ref[k]
            e_sum += e

            preview_sum = 0.0
            for j in range(N):
                idx = k + j + 1
                ref_val = zmp_ref[idx] if idx < M else zmp_ref[-1]
                preview_sum += Gd[j] * ref_val

            u = -Gi * e_sum - Gx @ x.flatten() - preview_sum
            x = A @ x + B * u

            com_pos[k] = x[0, 0]
            com_vel[k] = x[1, 0]
            zmp_actual[k] = p

        return com_pos, com_vel, zmp_actual

    def _build_zmp_reference(self, footsteps, n_steps):
        """footstep 시퀀스 → ZMP 레퍼런스 배열 (x, y 각각).

        각 스텝 동안 ZMP는 해당 footstep 위치에 고정.
        preview horizon 분량의 여유를 추가.
        """
        extra = self.N + self.samples_per_step  # preview 여유
        total = n_steps * self.samples_per_step + extra

        zmp_x = np.zeros(total)
        zmp_y = np.zeros(total)

        for i, (fx, fy) in enumerate(footsteps):
            start = i * self.samples_per_step
            end = start + self.samples_per_step
            zmp_x[start:end] = fx
            zmp_y[start:end] = fy

        # 마지막 footstep으로 나머지 채우기
        last_filled = n_steps * self.samples_per_step
        zmp_x[last_filled:] = footsteps[-1][0]
        zmp_y[last_filled:] = footsteps[-1][1]

        return zmp_x, zmp_y

    # ================================================================== #
    # Foot Trajectory Helpers
    # ================================================================== #
    @staticmethod
    def _cycloid_xy(p, start, end):
        """사이클로이드 XY 보간. p∈[0,1].

        θ = 2π·p → mod = (θ - sin(θ))/(2π)
        dp/dt at p=0,1 → dmod/dp = (1 - cos(2πp)) → 0 at p=0,1
        ∴ 이착지 속도 자동 0.
        """
        theta = 2 * np.pi * p
        mod = (theta - np.sin(theta)) / (2 * np.pi)
        return start + (end - start) * mod

    def _bezier_z(self, p, z_start, z_end, step_height=None):
        """5차 베지어 Z 궤적. p∈[0,1].

        6개 제어점으로 이착지 속도 0 보장:
          P0 = P1 = z_start   → dz/dp(0) = 5(P1-P0) = 0
          P4 = P5 = z_end     → dz/dp(1) = 5(P5-P4) = 0
          P2 = P3 = step_height (최대 높이)

        cf. g1_dcm_wbc의 3차 베지어(P0,P1=h,P2=h,P3)는
        dz/dp(0) = 3(h - z_start) ≠ 0 → 이착지 속도 불연속.
        lib_ZMPctrl.py는 CubicSpline(clamped)로 해결.
        여기서는 5차 베지어로 해결.
        """
        h = step_height if step_height is not None else self.step_height
        P = np.array([z_start, z_start, h, h, z_end, z_end])
        # 5차 베지어: B(t) = Σ C(5,i) * (1-t)^(5-i) * t^i * P[i]
        t = p
        omt = 1 - t
        z = (omt**5 * P[0]
             + 5 * omt**4 * t * P[1]
             + 10 * omt**3 * t**2 * P[2]
             + 10 * omt**2 * t**3 * P[3]
             + 5 * omt * t**4 * P[4]
             + t**5 * P[5])
        return z

    def _generate_swing(self, p, init_pos, target_pos):
        """Cycloid(XY) + Bezier(Z) 스윙 궤적. p∈[0,1]."""
        xy = self._cycloid_xy(p, init_pos[:2], target_pos[:2])
        z = self._bezier_z(p, init_pos[2], target_pos[2])
        return np.array([xy[0], xy[1], z])

    # ================================================================== #
    # Foot Trajectories
    # ================================================================== #
    def compute_foot_trajectories(self, footsteps, init_lf, init_rf):
        """DSP(양발 고정) + SSP(Cycloid XY + Bezier Z) 발 궤적 생성.

        Args:
            footsteps: list of (x, y) — plan_footsteps 결과
            init_lf: (3,) 초기 왼발 [x, y, z]
            init_rf: (3,) 초기 오른발 [x, y, z]

        Returns:
            left_traj: (T, 3)
            right_traj: (T, 3)
        """
        n_steps = len(footsteps)
        total = n_steps * self.samples_per_step

        left_traj = np.zeros((total, 3))
        right_traj = np.zeros((total, 3))

        left_pos = np.array(init_lf, dtype=float)
        right_pos = np.array(init_rf, dtype=float)
        ground_z_lf = init_lf[2]
        ground_z_rf = init_rf[2]

        for i in range(n_steps):
            start_idx = i * self.samples_per_step
            
            # 첫 스텝(i=0): 초기 DSP, 양발 고정
            if i == 0:
                for k in range(self.samples_per_step):
                    idx = start_idx + k
                    left_traj[idx] = left_pos
                    right_traj[idx] = right_pos
                continue
            
            # i >= 1: 실제 걸음
            is_right_swing = (i % 2 == 0)

            # 다음 스텝 착지 목표
            if i + 1 < n_steps:
                fx, fy = footsteps[i + 1]
                if is_right_swing:
                    swing_target = np.array([fx, fy, ground_z_rf])
                else:
                    swing_target = np.array([fx, fy, ground_z_lf])
            else:
                swing_target = None

            for k in range(self.samples_per_step):
                t = k * self.dt
                idx = start_idx + k

                # DSP 구간 또는 마지막 스텝: 양발 고정
                if t < self.dsp_time or swing_target is None:
                    left_traj[idx] = left_pos
                    right_traj[idx] = right_pos
                    continue

                # SSP 구간: swing phase 계산
                p = np.clip((t - self.dsp_time) / self.ssp_time, 0.0, 1.0)

                if is_right_swing:
                    right_traj[idx] = self._generate_swing(p, right_pos, swing_target)
                    left_traj[idx] = left_pos
                else:
                    left_traj[idx] = self._generate_swing(p, left_pos, swing_target)
                    right_traj[idx] = right_pos

            # 스텝 끝: 착지 위치 업데이트
            if swing_target is not None:
                if is_right_swing:
                    right_pos = swing_target.copy()
                else:
                    left_pos = swing_target.copy()

        return left_traj, right_traj

    # ================================================================== #
    # Phase Info
    # ================================================================== #
    def _build_phase_info(self, n_steps):
        """각 타임스텝의 위상 정보.

        Returns:
            list of (phase_name, swing_leg_idx, phase_ratio)
            - phase_name: 'dsp' | 'ssp'
            - swing_leg_idx: -1(dsp), 0(왼발 스윙), 1(오른발 스윙)
            - phase_ratio: 0.0~1.0
        """
        phase_info = []
        for i in range(n_steps):
            for k in range(self.samples_per_step):
                t = k * self.dt
                if t < self.dsp_time:
                    ratio = t / self.dsp_time if self.dsp_time > 0 else 0.0
                    phase_info.append(('dsp', -1, ratio))
                else:
                    ratio = (t - self.dsp_time) / self.ssp_time if self.ssp_time > 0 else 0.0
                    swing_idx = 1 if i % 2 == 0 else 0
                    phase_info.append(('ssp', swing_idx, np.clip(ratio, 0.0, 1.0)))
        return phase_info

    # ================================================================== #
    # Orchestrator
    # ================================================================== #
    def compute_all_trajectories(self, n_steps, step_length, step_width,
                                 init_com, init_lf, init_rf):
        """전체 오프라인 궤적 생성.

        Args:
            n_steps: 걸음 수
            step_length: 보폭 (x)
            step_width: 보폭 (y, 좌우 간격)
            init_com: (3,) 초기 CoM [x, y, z]
            init_lf: (3,) 초기 왼발 [x, y, z]
            init_rf: (3,) 초기 오른발 [x, y, z]

        Returns:
            dict: footsteps, ref_com_pos_x/y, ref_com_vel_x/y,
                  left_foot_traj, right_foot_traj, phase_info, zmp_ref_x/y
        """
        # 1. Footstep 계획
        init_lf_xy = init_lf[:2]
        init_rf_xy = init_rf[:2]
        footsteps, foot_targets = self.plan_footsteps(
            n_steps, step_length, step_width, init_lf_xy, init_rf_xy
        )

        # 2. ZMP 레퍼런스 생성
        zmp_ref_x, zmp_ref_y = self._build_zmp_reference(footsteps, n_steps)

        # 3. CoM 궤적 (x, y 독립)
        com_x, com_vx, zmp_ax = self.compute_com_trajectory(
            zmp_ref_x, init_com[0], 0.0
        )
        com_y, com_vy, zmp_ay = self.compute_com_trajectory(
            zmp_ref_y, init_com[1], 0.0
        )

        # CoM 궤적 길이를 발 궤적 길이에 맞춤
        total_foot = n_steps * self.samples_per_step
        L = min(len(com_x), total_foot)

        # 4. 발 궤적
        left_foot_traj, right_foot_traj = self.compute_foot_trajectories(
            footsteps, init_lf, init_rf
        )

        # 5. Phase info
        phase_info = self._build_phase_info(n_steps)

        # 길이 통일
        self.trajectories = {
            'footsteps': footsteps,
            'foot_targets': foot_targets,
            'ref_com_pos_x': com_x[:L],
            'ref_com_pos_y': com_y[:L],
            'ref_com_vel_x': com_vx[:L],
            'ref_com_vel_y': com_vy[:L],
            'zmp_ref_x': zmp_ref_x,
            'zmp_ref_y': zmp_ref_y,
            'zmp_actual_x': zmp_ax[:L],
            'zmp_actual_y': zmp_ay[:L],
            'left_foot_traj': left_foot_traj[:L],
            'right_foot_traj': right_foot_traj[:L],
            'phase_info': phase_info[:L],
            'length': L,
        }
        return self.trajectories

