"""Layer1: DCM 기반 궤적 플래너 + DSP 상태 머신 + Raibert 스윙 보정

DCM/Layer1.py의 TrajectoryOptimization과 new_ZMP/Layer1.py의
Raibert/Cycloid/Bezier를 결합한 통합 플래너.
"""

import numpy as np
from typing import List, Tuple


class DCMPlanner:
    """DCM 기반 궤적 플래너 + DSP 상태 머신 + Raibert 스윙 보정"""

    def __init__(
        self,
        z_c: float,
        g: float,
        step_time: float,
        dsp_time: float,
        step_height: float,
        dt: float,
        arm_swing_amp: float,
        init_dsp_extra: float,
        raibert_kp: float,
        foot_height: float,
    ):
        # DCM 파라미터 (DCM/Layer1.py에서)
        self.omega = np.sqrt(g / z_c)
        self.step_time = step_time
        self.dsp_time = dsp_time
        self.ssp_time = step_time - dsp_time
        self.step_height = step_height
        self.dt = dt
        self.arm_swing_amp = arm_swing_amp
        self.init_dsp_extra = init_dsp_extra

        # Raibert 파라미터 (new_ZMP/Layer1.py에서)
        self.raibert_kp = raibert_kp
        self.foot_height = foot_height

        # 상태 머신 변수
        self.trajectory_index = 0

        # Raibert 보정 상태
        self.fixed_swing_target = None  # SSP 시작 시 1회 고정
        self.swing_start_pos = None     # SSP 시작 시 발 위치

        # 사전 계산 궤적 (compute_all_trajectories에서 채워짐)
        self.trajectories = None

    # ================================================================== #
    # Helper: 스텝별 시간/샘플 수 (첫 스텝만 DSP 확장)
    # ================================================================== #

    def _step_time_for(self, step_idx: int) -> float:
        """step_idx번째 스텝의 총 시간."""
        if step_idx == 0 and self.init_dsp_extra > 0:
            return self.step_time + self.init_dsp_extra
        return self.step_time

    def _dsp_time_for(self, step_idx: int) -> float:
        """step_idx번째 스텝의 DSP 시간."""
        if step_idx == 0 and self.init_dsp_extra > 0:
            return self.dsp_time + self.init_dsp_extra
        return self.dsp_time

    def _samples_for(self, step_idx: int) -> int:
        """step_idx번째 스텝의 샘플 수."""
        return int(self._step_time_for(step_idx) / self.dt)

    def _total_samples(self, n_steps: int) -> int:
        """n_steps 스텝의 총 샘플 수."""
        return sum(self._samples_for(i) for i in range(n_steps))

    def _step_start_idx(self, step_idx: int) -> int:
        """step_idx번째 스텝의 시작 인덱스."""
        return sum(self._samples_for(i) for i in range(step_idx))

    # ================================================================== #
    # 사전 계산 (시작 전 1회 호출) — 후속 태스크에서 구현
    # ================================================================== #

    def plan_footsteps(self, n_steps, step_length, step_width, init_xy):
        """명목 발자국 계획. DCM/Layer1.py의 plan_footsteps와 동일.

        Args:
            n_steps: 발자국 수
            step_length: 보폭 (전진 거리)
            step_width: 좌우 폭 (CoM 기준)
            init_xy: 초기 CoM XY 위치 (2,)

        Returns:
            list of (x, y) 튜플, 길이 n_steps
        """
        footsteps = []
        for i in range(n_steps):
            if i == 0:
                x = init_xy[0]
                y = step_width  # 첫발은 왼발 (지지발 쪽)
            else:
                x = init_xy[0] + i * step_length
                if i % 2 != 0:
                    y = -step_width  # 오른발
                else:
                    y = step_width   # 왼발
            footsteps.append((x, y))
        return footsteps

    def compute_dcm_trajectory(self, footsteps):
        """역방향 재귀 → 순방향 DCM 궤적. DCM/Layer1.py와 동일.

        Args:
            footsteps: list of (x, y) 튜플

        Returns:
            ref_dcm (N, 2): DCM 기준 궤적
            ref_dcm_vel (N, 2): DCM 속도 기준 궤적
        """
        n_steps = len(footsteps)
        total_samples = self._total_samples(n_steps)

        ref_dcm = np.zeros((total_samples, 2))
        ref_dcm_vel = np.zeros((total_samples, 2))

        # DCM end-of-step (역방향 계산) — 각 스텝의 step_time 사용
        dcm_eos = np.zeros((n_steps, 2))
        dcm_eos[-1] = np.array(footsteps[-1])

        for i in range(n_steps - 2, -1, -1):
            next_zmp = np.array(footsteps[i + 1])
            exp_neg = np.exp(-self.omega * self._step_time_for(i + 1))
            dcm_eos[i] = next_zmp + (dcm_eos[i + 1] - next_zmp) * exp_neg

        # DCM 순방향 생성
        for i in range(n_steps):
            start_idx = self._step_start_idx(i)
            samples_i = self._samples_for(i)
            step_time_i = self._step_time_for(i)
            current_zmp = np.array(footsteps[i])
            xi_end = dcm_eos[i]

            for k in range(samples_i):
                idx = start_idx + k
                t = k * self.dt
                t_remaining = step_time_i - t

                current_dcm = current_zmp + (xi_end - current_zmp) * np.exp(
                    -self.omega * t_remaining
                )
                ref_dcm[idx] = current_dcm
                ref_dcm_vel[idx] = self.omega * (current_dcm - current_zmp)

        return ref_dcm, ref_dcm_vel

    def compute_com_trajectory(self, ref_dcm, init_com_xy):
        """DCM 오일러 적분 → CoM 궤적. DCM/Layer1.py와 동일.

        CoM 속도: com_vel[k] = omega * (ref_dcm[k] - com_pos[k])
        CoM 위치: com_pos[k+1] = com_pos[k] + com_vel[k] * dt

        Args:
            ref_dcm: (N, 2) DCM 기준 궤적
            init_com_xy: (2,) 초기 CoM XY 위치

        Returns:
            ref_com_pos (N, 2): CoM 위치 기준 궤적
            ref_com_vel (N, 2): CoM 속도 기준 궤적
        """
        total_samples = len(ref_dcm)
        ref_com_pos = np.zeros((total_samples, 2))
        ref_com_vel = np.zeros((total_samples, 2))

        current_com = init_com_xy[:2].copy()

        for k in range(total_samples):
            dx = self.omega * (ref_dcm[k] - current_com)
            ref_com_pos[k] = current_com
            ref_com_vel[k] = dx
            current_com = current_com + dx * self.dt

        return ref_com_pos, ref_com_vel

    def compute_foot_trajectories(self, footsteps, init_lf, init_rf, step_length):
        """DSP(양발 고정) + SSP(Cycloid XY + Bezier Z 스윙) 명목 발 궤적.

        각 스텝 i에 대해:
        - DSP 구간: 양발 현재 위치 고정
        - SSP 구간: Cycloid(XY) + Bezier(Z)로 스윙
          - 짝수 스텝(i%2==0): 오른발 스윙, 왼발 고정
          - 홀수 스텝(i%2!=0): 왼발 스윙, 오른발 고정

        Args:
            footsteps: list of (x, y) 튜플
            init_lf: (3,) 초기 왼발 위치 [x, y, z]
            init_rf: (3,) 초기 오른발 위치 [x, y, z]
            step_length: 보폭

        Returns:
            left_traj (N, 3), right_traj (N, 3)
        """
        n_steps = len(footsteps)
        total_samples = self._total_samples(n_steps)

        left_traj = np.zeros((total_samples, 3))
        right_traj = np.zeros((total_samples, 3))

        left_pos = init_lf.copy()
        right_pos = init_rf.copy()
        ground_z_lf = init_lf[2]
        ground_z_rf = init_rf[2]
        foot_x_start = init_lf[0]

        foot_targets = []
        for i in range(n_steps):
            if i == 0:
                fx = foot_x_start
            else:
                fx = foot_x_start + i * step_length
            fy = footsteps[i][1]
            foot_targets.append((fx, fy))

        for i in range(n_steps):
            start_idx = self._step_start_idx(i)
            samples_i = self._samples_for(i)
            dsp_time_i = self._dsp_time_for(i)
            step_time_i = self._step_time_for(i)
            ssp_time_i = step_time_i - dsp_time_i
            is_right_swing = (i % 2 == 0)

            if i + 1 < n_steps:
                swing_target_xy = np.array(foot_targets[i + 1])
            else:
                swing_target_xy = None

            for k in range(samples_i):
                t = k * self.dt
                idx = start_idx + k

                if t < dsp_time_i or swing_target_xy is None:
                    left_traj[idx] = left_pos
                    right_traj[idx] = right_pos
                    continue

                # SSP: Cycloid(XY) + Bezier(Z)
                swing_phase = np.clip((t - dsp_time_i) / ssp_time_i, 0.0, 1.0)

                if is_right_swing:
                    swing_init = right_pos.copy()
                    swing_target = np.array([swing_target_xy[0], swing_target_xy[1], ground_z_rf])
                    pos = self.generate_swing_trajectory(swing_phase, swing_init, swing_target)
                    right_traj[idx] = pos
                    left_traj[idx] = left_pos
                else:
                    swing_init = left_pos.copy()
                    swing_target = np.array([swing_target_xy[0], swing_target_xy[1], ground_z_lf])
                    pos = self.generate_swing_trajectory(swing_phase, swing_init, swing_target)
                    left_traj[idx] = pos
                    right_traj[idx] = right_pos

            if swing_target_xy is not None:
                if is_right_swing:
                    right_pos = np.array([swing_target_xy[0], swing_target_xy[1], ground_z_rf])
                else:
                    left_pos = np.array([swing_target_xy[0], swing_target_xy[1], ground_z_lf])

        return left_traj, right_traj

    def _build_phase_info(self, n_steps):
        """각 타임스텝의 위상 정보 사전 계산.
        반환: phase_info[k] = (phase_name, swing_leg_idx, phase_ratio)
        """
        phase_info = []
        for i in range(n_steps):
            samples_i = self._samples_for(i)
            dsp_time_i = self._dsp_time_for(i)
            step_time_i = self._step_time_for(i)
            ssp_time_i = step_time_i - dsp_time_i

            for k in range(samples_i):
                t = k * self.dt
                if t < dsp_time_i:
                    phase_name = "DSP_init" if i == 0 else "DSP_transition"
                    swing_leg_idx = -1
                    phase_ratio = t / dsp_time_i if dsp_time_i > 0 else 0.0
                else:
                    phase_name = "SSP"
                    swing_leg_idx = 1 if i % 2 == 0 else 0  # 짝수 스텝: 오른발 스윙
                    phase_ratio = (t - dsp_time_i) / ssp_time_i if ssp_time_i > 0 else 0.0
                phase_info.append((phase_name, swing_leg_idx, phase_ratio))
        return phase_info

    def compute_all_trajectories(self, n_steps, step_length, step_width,
                                 init_com, init_lf, init_rf):
        """전체 기준 궤적 + phase_info 사전 계산.
        반환: dict with footsteps, ref_dcm, ref_dcm_vel, ref_com_pos, ref_com_vel,
              left_foot_traj, right_foot_traj, arm_left, arm_right, phase_info
        """
        footsteps = self.plan_footsteps(n_steps, step_length, step_width, init_com[:2])
        ref_dcm, ref_dcm_vel = self.compute_dcm_trajectory(footsteps)
        ref_com_pos, ref_com_vel = self.compute_com_trajectory(ref_dcm, init_com[:2])
        left_foot_traj, right_foot_traj = self.compute_foot_trajectories(
            footsteps, init_lf, init_rf, step_length
        )
        total = self._total_samples(n_steps)
        arm_left, arm_right = self.compute_arm_swing_trajectory(total)
        phase_info = self._build_phase_info(n_steps)

        self.trajectories = {
            'footsteps': footsteps,
            'ref_dcm': ref_dcm,
            'ref_dcm_vel': ref_dcm_vel,
            'ref_com_pos': ref_com_pos,
            'ref_com_vel': ref_com_vel,
            'left_foot_traj': left_foot_traj,
            'right_foot_traj': right_foot_traj,
            'arm_left': arm_left,
            'arm_right': arm_right,
            'phase_info': phase_info,
        }
        return self.trajectories

    # ================================================================== #
    # 실시간 상태 머신 (매 루프 호출) — 후속 태스크에서 구현
    # ================================================================== #

    def step(self):
        """trajectory_index 증가 및 사전 계산 데이터 반환. Task 3.1에서 구현."""
        pass

    def compute_raibert_correction(self, torso_vel, desired_vel,
                                   nominal_footstep_xy):
        """Raibert Heuristic 스윙 착지점 보정. Task 3.2에서 구현."""
        pass

    def generate_swing_trajectory(self, phase_ratio, init_pos, target_pos):
        """Cycloid(XY) + Bezier(Z) 스윙 궤적 생성.

        Args:
            phase_ratio: 0.0~1.0 SSP 진행률
            init_pos: (3,) 스윙 시작 위치
            target_pos: (3,) 스윙 목표 위치

        Returns:
            (3,) — [x, y, z]
        """
        xy = self.Cycloid_Trajectory(phase_ratio, init_pos[:2], target_pos[:2])
        z = self.Bezier_Curve_interpolation(phase_ratio, init_pos[2], target_pos[2])
        return np.array([xy[0], xy[1], z])

    def Cycloid_Trajectory(self, p, init_pos, next_xy):
        """수평 XY 궤적 (사이클로이드). new_ZMP/Layer1.py와 동일.

        theta = 2π·p, cycloid_mod = (theta - sin(theta))/(2π)
        xy = init + (target - init) * cycloid_mod
        """
        theta = 2 * np.pi * p
        cycloid_mod = (theta - np.sin(theta)) / (2 * np.pi)
        x = init_pos[0] + (next_xy[0] - init_pos[0]) * cycloid_mod
        y = init_pos[1] + (next_xy[1] - init_pos[1]) * cycloid_mod
        return np.array([x, y])

    def Bezier_Curve_interpolation(self, s, init_z=0.0, target_z=0.0):
        """수직 Z 궤적 — 단일 3차 베지어 곡선.

        제어점: P0=init_z, P1=step_height, P2=step_height, P3=target_z
        s=0 → init_z, s≈0.5 → step_height 부근, s=1 → target_z
        """
        P0 = init_z
        P1 = self.step_height
        P2 = self.step_height
        P3 = target_z
        t = s
        z = ((1 - t)**3 * P0 +
             3 * (1 - t)**2 * t * P1 +
             3 * (1 - t) * t**2 * P2 +
             t**3 * P3)
        return z
