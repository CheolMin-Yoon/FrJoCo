"""Task 2.5: compute_foot_trajectories 단위 테스트

DSP(양발 고정) + SSP(코사인 보간 스윙) 명목 발 궤적 검증.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Layer1 import DCMPlanner


@pytest.fixture
def planner():
    """기본 파라미터로 DCMPlanner 생성."""
    return DCMPlanner(
        z_c=0.69,
        g=9.81,
        step_time=0.7,
        dsp_time=0.08,
        step_height=0.08,
        dt=0.002,
        arm_swing_amp=0.15,
        init_dsp_extra=0.12,
        raibert_kp=0.5,
        foot_height=0.08,
    )


@pytest.fixture
def basic_setup(planner):
    """기본 발자국 계획 + 초기 발 위치."""
    n_steps = 6
    step_length = 0.1
    step_width = 0.1185
    init_xy = np.array([0.035, 0.0])
    footsteps = planner.plan_footsteps(n_steps, step_length, step_width, init_xy)
    init_lf = np.array([-0.0014, 0.1185, 0.0])
    init_rf = np.array([-0.0014, -0.1185, 0.0])
    return footsteps, init_lf, init_rf, step_length, n_steps


class TestReturnShape:
    def test_returns_tuple_of_two_arrays(self, planner, basic_setup):
        footsteps, init_lf, init_rf, step_length, n_steps = basic_setup
        result = planner.compute_foot_trajectories(footsteps, init_lf, init_rf, step_length)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_shape_matches_total_samples(self, planner, basic_setup):
        footsteps, init_lf, init_rf, step_length, n_steps = basic_setup
        left, right = planner.compute_foot_trajectories(footsteps, init_lf, init_rf, step_length)
        total = planner._total_samples(n_steps)
        assert left.shape == (total, 3)
        assert right.shape == (total, 3)


class TestDSPFixedFeet:
    """DSP 구간에서 양발이 고정되는지 검증."""

    def test_first_step_dsp_both_feet_fixed(self, planner, basic_setup):
        footsteps, init_lf, init_rf, step_length, _ = basic_setup
        left, right = planner.compute_foot_trajectories(footsteps, init_lf, init_rf, step_length)

        dsp_samples = int(planner._dsp_time_for(0) / planner.dt)
        for k in range(dsp_samples):
            np.testing.assert_array_almost_equal(left[k], init_lf)
            np.testing.assert_array_almost_equal(right[k], init_rf)

    def test_second_step_dsp_both_feet_fixed(self, planner, basic_setup):
        footsteps, init_lf, init_rf, step_length, _ = basic_setup
        left, right = planner.compute_foot_trajectories(footsteps, init_lf, init_rf, step_length)

        step1_start = planner._step_start_idx(1)
        dsp_samples = int(planner._dsp_time_for(1) / planner.dt)
        # During step 1 DSP, both feet should be constant
        for k in range(dsp_samples):
            idx = step1_start + k
            np.testing.assert_array_almost_equal(left[idx], left[step1_start])
            np.testing.assert_array_almost_equal(right[idx], right[step1_start])


class TestSSPSwing:
    """SSP 구간에서 스윙 발이 이동하는지 검증."""

    def test_even_step_right_foot_swings(self, planner, basic_setup):
        """짝수 스텝(i=0): 오른발 스윙, 왼발 고정."""
        footsteps, init_lf, init_rf, step_length, _ = basic_setup
        left, right = planner.compute_foot_trajectories(footsteps, init_lf, init_rf, step_length)

        start = planner._step_start_idx(0)
        dsp_samples = int(planner._dsp_time_for(0) / planner.dt)
        samples_0 = planner._samples_for(0)

        # SSP 중간 지점에서 왼발은 고정
        mid_idx = start + dsp_samples + (samples_0 - dsp_samples) // 2
        np.testing.assert_array_almost_equal(left[mid_idx], init_lf)

        # 오른발은 SSP 중간에서 Z > 0 (스윙 높이)
        assert right[mid_idx, 2] > 0.0

    def test_odd_step_left_foot_swings(self, planner, basic_setup):
        """홀수 스텝(i=1): 왼발 스윙, 오른발 고정."""
        footsteps, init_lf, init_rf, step_length, _ = basic_setup
        left, right = planner.compute_foot_trajectories(footsteps, init_lf, init_rf, step_length)

        start = planner._step_start_idx(1)
        dsp_samples = int(planner._dsp_time_for(1) / planner.dt)
        samples_1 = planner._samples_for(1)

        mid_idx = start + dsp_samples + (samples_1 - dsp_samples) // 2
        # 왼발은 SSP 중간에서 Z > 0
        assert left[mid_idx, 2] > 0.0
        # 오른발은 고정 (step 0에서 스윙 후 착지한 위치)
        np.testing.assert_array_almost_equal(right[mid_idx], right[start])


class TestSwingZProfile:
    """스윙 발 Z축 프로파일 검증."""

    def test_swing_z_starts_at_ground(self, planner, basic_setup):
        footsteps, init_lf, init_rf, step_length, _ = basic_setup
        left, right = planner.compute_foot_trajectories(footsteps, init_lf, init_rf, step_length)

        # Step 0 SSP 시작: 오른발 Z ≈ ground
        start = planner._step_start_idx(0)
        dsp_samples = int(planner._dsp_time_for(0) / planner.dt)
        ssp_start_idx = start + dsp_samples
        assert np.isclose(right[ssp_start_idx, 2], init_rf[2], atol=1e-6)

    def test_swing_z_returns_to_ground(self, planner, basic_setup):
        footsteps, init_lf, init_rf, step_length, _ = basic_setup
        left, right = planner.compute_foot_trajectories(footsteps, init_lf, init_rf, step_length)

        # Step 0 SSP 끝: 오른발 Z ≈ ground
        end_idx = planner._step_start_idx(1) - 1
        assert np.isclose(right[end_idx, 2], init_rf[2], atol=5e-3)

    def test_swing_z_peak_near_step_height(self, planner, basic_setup):
        footsteps, init_lf, init_rf, step_length, _ = basic_setup
        left, right = planner.compute_foot_trajectories(footsteps, init_lf, init_rf, step_length)

        # Step 0 SSP 중간: 오른발 Z ≈ step_height
        start = planner._step_start_idx(0)
        dsp_samples = int(planner._dsp_time_for(0) / planner.dt)
        ssp_samples = planner._samples_for(0) - dsp_samples
        mid_idx = start + dsp_samples + ssp_samples // 2
        assert np.isclose(right[mid_idx, 2], planner.step_height, atol=0.01)


class TestFootTargetPositions:
    """스윙 종료 후 발 위치가 올바른지 검증."""

    def test_right_foot_lands_at_target_after_step0(self, planner, basic_setup):
        footsteps, init_lf, init_rf, step_length, _ = basic_setup
        left, right = planner.compute_foot_trajectories(footsteps, init_lf, init_rf, step_length)

        # Step 0 종료 후 오른발 위치: foot_targets[1]
        step1_start = planner._step_start_idx(1)
        expected_x = init_lf[0] + 1 * step_length  # foot_targets[1].x
        expected_y = footsteps[1][1]  # -step_width
        assert np.isclose(right[step1_start, 0], expected_x, atol=1e-6)
        assert np.isclose(right[step1_start, 1], expected_y, atol=1e-6)
        assert np.isclose(right[step1_start, 2], init_rf[2], atol=1e-6)


class TestLastStepNoSwing:
    """마지막 스텝에서 스윙 타겟이 없으면 양발 고정."""

    def test_last_step_both_feet_fixed(self, planner):
        n_steps = 2
        footsteps = planner.plan_footsteps(n_steps, 0.1, 0.1185, np.array([0.035, 0.0]))
        init_lf = np.array([0.0, 0.1185, 0.0])
        init_rf = np.array([0.0, -0.1185, 0.0])
        left, right = planner.compute_foot_trajectories(footsteps, init_lf, init_rf, 0.1)

        last_start = planner._step_start_idx(n_steps - 1)
        last_samples = planner._samples_for(n_steps - 1)
        # 마지막 스텝은 swing_target_xy=None이므로 전체 구간 양발 고정
        for k in range(last_samples):
            idx = last_start + k
            np.testing.assert_array_almost_equal(left[idx], left[last_start])
            np.testing.assert_array_almost_equal(right[idx], right[last_start])


class TestCosineInterpolationXY:
    """XY 코사인 보간이 올바른지 검증."""

    def test_swing_xy_at_midpoint(self, planner, basic_setup):
        """SSP 50% 지점에서 XY는 시작과 끝의 중간."""
        footsteps, init_lf, init_rf, step_length, _ = basic_setup
        left, right = planner.compute_foot_trajectories(footsteps, init_lf, init_rf, step_length)

        start = planner._step_start_idx(0)
        dsp_samples = int(planner._dsp_time_for(0) / planner.dt)
        ssp_samples = planner._samples_for(0) - dsp_samples
        mid_idx = start + dsp_samples + ssp_samples // 2

        # At swing_phase=0.5, progress = 0.5*(1-cos(pi*0.5)) = 0.5
        target_x = init_lf[0] + step_length  # foot_targets[1].x
        target_y = footsteps[1][1]
        expected_x = init_rf[0] + (target_x - init_rf[0]) * 0.5
        expected_y = init_rf[1] + (target_y - init_rf[1]) * 0.5
        assert np.isclose(right[mid_idx, 0], expected_x, atol=0.005)
        assert np.isclose(right[mid_idx, 1], expected_y, atol=0.005)
