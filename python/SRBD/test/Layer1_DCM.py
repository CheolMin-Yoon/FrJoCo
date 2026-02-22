# A Benchmarking of DCM Based Architectures for Position and Velocity Controlled Walking of Humanoid Robots
# DCM Trajectory Planner 저자가 설명하는 첫번째 Layer

# 이 레이어에서는 

import numpy as np
from typing import List, Tuple

from config import GRAVITY, STEP_HEIGHT, DSP_TIME, DT, STEP_TIME, INIT_DSP_EXTRA

class TrajectoryOptimization:

    def __init__(
        self,
        z_c: float,
        g: float = GRAVITY,
        step_time: float = STEP_TIME,
        dsp_time: float = DSP_TIME,
        step_height: float = STEP_HEIGHT,
        dt: float = DT,
        arm_swing_amp: float = 0.15,
        init_dsp_extra: float = INIT_DSP_EXTRA,
    ):
        self.z_c = z_c
        self.omega = np.sqrt(g / z_c)
        self.step_time = step_time
        self.dsp_time = dsp_time
        self.ssp_time = step_time - dsp_time
        self.step_height = step_height
        self.dt = dt
        self.samples_per_step = int(step_time / dt)
        self.arm_swing_amp = arm_swing_amp
        self.init_dsp_extra = init_dsp_extra  # 첫 스텝 DSP 추가 시간

    # ========================================================================= #
    # Helper: 스텝별 시간/샘플 수 (첫 스텝만 DSP 확장)
    # ========================================================================= #
    def _step_time_for(self, i: int) -> float:
        """i번째 스텝의 총 시간"""
        if i == 0 and self.init_dsp_extra > 0:
            return self.step_time + self.init_dsp_extra
        return self.step_time

    def _dsp_time_for(self, i: int) -> float:
        """i번째 스텝의 DSP 시간"""
        if i == 0 and self.init_dsp_extra > 0:
            return self.dsp_time + self.init_dsp_extra
        return self.dsp_time

    def _samples_for(self, i: int) -> int:
        return int(self._step_time_for(i) / self.dt)

    def _total_samples(self, n_steps: int) -> int:
        return sum(self._samples_for(i) for i in range(n_steps))

    def _step_start_idx(self, step: int) -> int:
        return sum(self._samples_for(i) for i in range(step))

    # ========================================================================= #
    # 1. Footstep Plan
    # ========================================================================= #
    def plan_footsteps(
        self,
        n_steps: int,
        step_length: float,
        step_width: float,
        init_xy: np.ndarray = np.array([0.035, 0.0]) # 근데 이게 아마 실제로 0.0351, 0.0 일껄
    ) -> List[Tuple[float, float]]:                  # init_xy는 초기 CoM
        
        # 리스트로 저장
        footsteps = []
        
        # 첫발은 왼발부터 시작
        for i in range(n_steps):
            if i == 0:
                x = init_xy[0]
                y = step_width  # sway가 CoM을 왼발(지지발)로 이동시키므로 정확히 step_width
            else:
                x = init_xy[0] + i * step_length
                
                if i % 2 != 0:
                    y = -step_width   # 오른발
                else:
                    y = step_width  # 왼발
            footsteps.append((x,y))
        return footsteps

    # ========================================================================= #
    # 2. DCM Trajectory
    # ========================================================================= #
    def compute_dcm_trajectory(
        self,
        footsteps: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
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

                current_dcm = current_zmp + (xi_end - current_zmp) * np.exp(-self.omega * t_remaining)
                ref_dcm[idx] = current_dcm
                ref_dcm_vel[idx] = self.omega * (current_dcm - current_zmp)

        return ref_dcm, ref_dcm_vel

    # ========================================================================= #
    # 3. CoM Trajectory (DCM Integration)
    # ========================================================================= #
    def compute_com_trajectory(
        self,
        ref_dcm: np.ndarray,
        init_com_xy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    # ========================================================================= #
    # 4. Foot Trajectory (월드 좌표계)
    # ========================================================================= #
    def compute_foot_trajectories(
        self,
        footsteps: List[Tuple[float, float]],
        init_lf: np.ndarray,
        init_rf: np.ndarray,
        step_length: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """발 궤적 생성. 발 위치 기준으로 swing target 계산."""
        n_steps = len(footsteps)
        total_samples = self._total_samples(n_steps)

        left_traj = np.zeros((total_samples, 3))
        right_traj = np.zeros((total_samples, 3))

        # 초기 발 위치 (월드 좌표)
        left_pos = init_lf.copy()
        right_pos = init_rf.copy()
        ground_z_lf = init_lf[2]
        ground_z_rf = init_rf[2]
        foot_x_start = init_lf[0]  # 양발 x 동일 (≈-0.0014)

        # ★ 발 기준 착지점 (footsteps와 독립)
        # footsteps는 CoM 기준 → ZMP/DCM용
        # foot_targets는 발 기준 → IK 목표용
        foot_targets = []
        for i in range(n_steps):
            if i == 0:
                fx = foot_x_start
            else:
                fx = foot_x_start + i * step_length
            # y는 footsteps와 동일 (좌우 발 위치)
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

                swing_phase = np.clip((t - dsp_time_i) / ssp_time_i, 0.0, 1.0)
                progress = 0.5 * (1 - np.cos(np.pi * swing_phase))

                if is_right_swing:
                    curr_x = right_pos[0] + (swing_target_xy[0] - right_pos[0]) * progress
                    curr_y = right_pos[1] + (swing_target_xy[1] - right_pos[1]) * progress
                    curr_z = ground_z_rf + self.step_height * np.sin(np.pi * swing_phase)
                    right_traj[idx] = [curr_x, curr_y, curr_z]
                    left_traj[idx] = left_pos
                else:
                    curr_x = left_pos[0] + (swing_target_xy[0] - left_pos[0]) * progress
                    curr_y = left_pos[1] + (swing_target_xy[1] - left_pos[1]) * progress
                    curr_z = ground_z_lf + self.step_height * np.sin(np.pi * swing_phase)
                    left_traj[idx] = [curr_x, curr_y, curr_z]
                    right_traj[idx] = right_pos

            if swing_target_xy is not None:
                if is_right_swing:
                    right_pos = np.array([swing_target_xy[0], swing_target_xy[1], ground_z_rf])
                else:
                    left_pos = np.array([swing_target_xy[0], swing_target_xy[1], ground_z_lf])

        return left_traj, right_traj

    # ========================================================================= #
    # Main Wrapper
    # ========================================================================= #
    def compute_all_trajectories(
        self,
        n_steps: int,
        step_length: float,
        step_width: float,
        init_com: np.ndarray,     # (3,) 월드 좌표 CoM
        init_lf: np.ndarray,      # (3,) 월드 좌표 왼발
        init_rf: np.ndarray,      # (3,) 월드 좌표 오른발
    ):
        # 1. 발자국 계획
        footsteps = self.plan_footsteps(n_steps, step_length, step_width, init_xy=init_com[:2])

        # 2. DCM 궤적
        ref_dcm, ref_dcm_vel = self.compute_dcm_trajectory(footsteps)

        # 3. CoM 궤적
        com_xy, com_vel = self.compute_com_trajectory(ref_dcm, init_com[:2])
        com_pos = np.column_stack([com_xy, np.full(len(com_xy), init_com[2])])

        # 4. 발 궤적 (발 위치 기준)
        l_foot, r_foot = self.compute_foot_trajectories(footsteps, init_lf, init_rf, step_length)

        return footsteps, ref_dcm, ref_dcm_vel, com_pos, com_vel, l_foot, r_foot
    
    # ========================================================================= #
    # 5. 팔 스윙 궤적
    # ========================================================================= #
    def compute_arm_swing_trajectory(self, trajectory_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        팔 스윙 궤적 생성 (보행과 동기화된 정현파)
        
        Args:
            trajectory_length: 전체 궤적 길이 (샘플 수)
        
        Returns:
            left_arm: 왼팔 shoulder_pitch 각도 (rad)
            right_arm: 오른팔 shoulder_pitch 각도 (rad)
        """
        left_traj = np.zeros(trajectory_length)
        right_traj = np.zeros(trajectory_length)
        period = 2 * self.step_time  # 두 걸음에 한 주기
        
        for k in range(trajectory_length):
            t = k * self.dt
            phase = 2 * np.pi * t / period
            swing = self.arm_swing_amp * np.cos(phase)
            
            # envelope: 처음 한 주기 동안 부드럽게 증가
            if t < period:
                envelope = min(1.0, 0.5 * (1 - np.cos(np.pi * t / period)))
            else:
                envelope = 1.0
            
            left_traj[k] = swing * envelope
            right_traj[k] = -swing * envelope  # 반대 위상
        
        return left_traj, right_traj
