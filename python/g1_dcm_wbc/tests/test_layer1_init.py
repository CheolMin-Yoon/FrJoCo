"""Task 2.1: DCMPlanner __init__ 및 헬퍼 메서드 테스트"""

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


class TestInit:
    def test_omega(self, planner):
        expected = np.sqrt(9.81 / 0.69)
        assert np.isclose(planner.omega, expected)

    def test_ssp_time(self, planner):
        assert np.isclose(planner.ssp_time, 0.7 - 0.08)

    def test_initial_state(self, planner):
        assert planner.trajectory_index == 0
        assert planner.fixed_swing_target is None
        assert planner.swing_start_pos is None
        assert planner.trajectories is None


class TestStepTimeFor:
    def test_first_step_extended(self, planner):
        expected = 0.7 + 0.12
        assert np.isclose(planner._step_time_for(0), expected)

    def test_other_steps_normal(self, planner):
        assert np.isclose(planner._step_time_for(1), 0.7)
        assert np.isclose(planner._step_time_for(5), 0.7)

    def test_no_extra_when_zero(self):
        p = DCMPlanner(0.69, 9.81, 0.7, 0.08, 0.08, 0.002, 0.15, 0.0, 0.5, 0.08)
        assert np.isclose(p._step_time_for(0), 0.7)


class TestDspTimeFor:
    def test_first_step_extended(self, planner):
        expected = 0.08 + 0.12
        assert np.isclose(planner._dsp_time_for(0), expected)

    def test_other_steps_normal(self, planner):
        assert np.isclose(planner._dsp_time_for(1), 0.08)


class TestSamplesFor:
    def test_first_step(self, planner):
        expected = int((0.7 + 0.12) / 0.002)
        assert planner._samples_for(0) == expected

    def test_other_steps(self, planner):
        expected = int(0.7 / 0.002)
        assert planner._samples_for(1) == expected


class TestTotalSamples:
    def test_single_step(self, planner):
        assert planner._total_samples(1) == planner._samples_for(0)

    def test_multiple_steps(self, planner):
        n = 5
        expected = planner._samples_for(0) + 4 * planner._samples_for(1)
        assert planner._total_samples(n) == expected


class TestStepStartIdx:
    def test_first_step(self, planner):
        assert planner._step_start_idx(0) == 0

    def test_second_step(self, planner):
        assert planner._step_start_idx(1) == planner._samples_for(0)

    def test_third_step(self, planner):
        expected = planner._samples_for(0) + planner._samples_for(1)
        assert planner._step_start_idx(2) == expected


class TestStubMethods:
    """스텁 메서드가 존재하고 에러 없이 호출 가능한지 확인."""

    def test_plan_footsteps_implemented(self, planner):
        result = planner.plan_footsteps(10, 0.1, 0.1, np.zeros(2))
        assert isinstance(result, list) and len(result) == 10

    def test_compute_dcm_trajectory_implemented(self, planner):
        # 빈 리스트는 IndexError → 구현됨을 확인
        with pytest.raises(IndexError):
            planner.compute_dcm_trajectory([])

    def test_compute_com_trajectory_implemented(self, planner):
        result = planner.compute_com_trajectory(np.zeros((10, 2)), np.zeros(2))
        assert result is not None

    def test_compute_foot_trajectories_implemented(self, planner):
        result = planner.compute_foot_trajectories([], np.zeros(3), np.zeros(3), 0.1)
        assert result is not None

    def test_compute_arm_swing_trajectory(self, planner):
        left, right = planner.compute_arm_swing_trajectory(100)
        assert left.shape == (100,)
        assert right.shape == (100,)
        # 반대 위상: left ≈ -right
        np.testing.assert_allclose(left, -right, atol=1e-12)

    def test_build_phase_info_returns_list(self, planner):
        result = planner._build_phase_info(5)
        assert isinstance(result, list)
        assert len(result) == planner._total_samples(5)

    def test_build_phase_info_valid_entries(self, planner):
        result = planner._build_phase_info(3)
        valid_phases = {"DSP_init", "SSP", "DSP_transition"}
        valid_swing = {-1, 0, 1}
        for phase_name, swing_leg_idx, phase_ratio in result:
            assert phase_name in valid_phases
            assert swing_leg_idx in valid_swing
            assert 0.0 <= phase_ratio <= 1.0 + 1e-9

    def test_compute_all_trajectories_returns_dict(self, planner):
        result = planner.compute_all_trajectories(
            5, 0.1, 0.1185, np.array([0.0, 0.0, 0.69]),
            np.array([0.0, 0.1185, 0.0]), np.array([0.0, -0.1185, 0.0])
        )
        assert isinstance(result, dict)
        expected_keys = {
            'footsteps', 'ref_dcm', 'ref_dcm_vel',
            'ref_com_pos', 'ref_com_vel',
            'left_foot_traj', 'right_foot_traj',
            'arm_left', 'arm_right', 'phase_info',
        }
        assert set(result.keys()) == expected_keys

    def test_compute_all_trajectories_stores_trajectories(self, planner):
        result = planner.compute_all_trajectories(
            5, 0.1, 0.1185, np.array([0.0, 0.0, 0.69]),
            np.array([0.0, 0.1185, 0.0]), np.array([0.0, -0.1185, 0.0])
        )
        assert planner.trajectories is result

    def test_compute_all_trajectories_consistent_lengths(self, planner):
        n_steps = 5
        result = planner.compute_all_trajectories(
            n_steps, 0.1, 0.1185, np.array([0.0, 0.0, 0.69]),
            np.array([0.0, 0.1185, 0.0]), np.array([0.0, -0.1185, 0.0])
        )
        total = planner._total_samples(n_steps)
        assert len(result['footsteps']) == n_steps
        assert result['ref_dcm'].shape[0] == total
        assert result['ref_com_pos'].shape[0] == total
        assert result['left_foot_traj'].shape[0] == total
        assert result['arm_left'].shape[0] == total
        assert len(result['phase_info']) == total

    def test_step_stub(self, planner):
        assert planner.step() is None

    def test_compute_raibert_correction_stub(self, planner):
        assert planner.compute_raibert_correction(
            np.zeros(3), np.zeros(2), np.zeros(2)
        ) is None

    def test_generate_swing_trajectory_implemented(self, planner):
        result = planner.generate_swing_trajectory(0.5, np.zeros(3), np.ones(3))
        assert result is not None and result.shape == (3,)

    def test_cycloid_trajectory_implemented(self, planner):
        result = planner.Cycloid_Trajectory(0.5, np.zeros(2), np.ones(2))
        assert result is not None and result.shape == (2,)

    def test_bezier_curve_implemented(self, planner):
        result = planner.Bezier_Curve_interpolation(0.5)
        assert isinstance(result, float) or result is not None
