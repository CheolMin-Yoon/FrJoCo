"""Task 2.3: compute_dcm_trajectory 단위 테스트

역방향 재귀 dcm_eos → 순방향 ref_dcm, ref_dcm_vel 생성 검증.
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
def simple_planner():
    """init_dsp_extra=0인 단순 플래너 (모든 스텝 동일 시간)."""
    return DCMPlanner(
        z_c=0.69,
        g=9.81,
        step_time=0.7,
        dsp_time=0.08,
        step_height=0.08,
        dt=0.002,
        arm_swing_amp=0.15,
        init_dsp_extra=0.0,
        raibert_kp=0.5,
        foot_height=0.08,
    )


class TestReturnShape:
    """반환 배열의 shape 검증."""

    def test_shape_matches_total_samples(self, planner):
        footsteps = planner.plan_footsteps(5, 0.1, 0.1185, np.array([0.0, 0.0]))
        ref_dcm, ref_dcm_vel = planner.compute_dcm_trajectory(footsteps)
        total = planner._total_samples(len(footsteps))
        assert ref_dcm.shape == (total, 2)
        assert ref_dcm_vel.shape == (total, 2)

    def test_single_step_shape(self, simple_planner):
        footsteps = [(0.0, 0.1)]
        ref_dcm, ref_dcm_vel = simple_planner.compute_dcm_trajectory(footsteps)
        expected = simple_planner._samples_for(0)
        assert ref_dcm.shape == (expected, 2)

    def test_two_steps_shape(self, planner):
        footsteps = [(0.0, 0.1), (0.1, -0.1)]
        ref_dcm, ref_dcm_vel = planner.compute_dcm_trajectory(footsteps)
        total = planner._total_samples(2)
        assert ref_dcm.shape == (total, 2)


class TestDCMEndOfStep:
    """DCM 궤적의 마지막 값이 마지막 발자국과 일치하는지 검증 (Property 2)."""

    def test_last_dcm_equals_last_footstep(self, planner):
        footsteps = planner.plan_footsteps(5, 0.1, 0.1185, np.array([0.0, 0.0]))
        ref_dcm, _ = planner.compute_dcm_trajectory(footsteps)
        last_footstep = np.array(footsteps[-1])
        # 마지막 타임스텝의 DCM은 마지막 발자국에 수렴
        np.testing.assert_allclose(ref_dcm[-1], last_footstep, atol=1e-6)

    def test_single_step_dcm_converges(self, simple_planner):
        footsteps = [(0.5, 0.1)]
        ref_dcm, _ = simple_planner.compute_dcm_trajectory(footsteps)
        np.testing.assert_allclose(ref_dcm[-1], [0.5, 0.1], atol=1e-6)


class TestBackwardRecursion:
    """역방향 재귀 dcm_eos 값 검증."""

    def test_two_step_dcm_eos(self, simple_planner):
        """2-step 케이스에서 dcm_eos[0] 수동 계산 검증."""
        footsteps = [(0.0, 0.1), (0.1, -0.1)]
        omega = simple_planner.omega
        step_time = simple_planner.step_time

        # dcm_eos[-1] = footsteps[-1] = (0.1, -0.1)
        # dcm_eos[0] = footsteps[1] + exp(-omega * step_time_for(1)) * (dcm_eos[1] - footsteps[1])
        # step_time_for(1) = 0.7 (no extra for step 1)
        exp_neg = np.exp(-omega * step_time)
        expected_eos_0 = np.array([0.1, -0.1]) + exp_neg * (np.array([0.1, -0.1]) - np.array([0.1, -0.1]))
        # dcm_eos[1] == footsteps[1], so dcm_eos[0] == footsteps[1]
        np.testing.assert_allclose(expected_eos_0, [0.1, -0.1], atol=1e-10)

        # 실제 궤적의 마지막 스텝 시작 시점 DCM 검증
        ref_dcm, _ = simple_planner.compute_dcm_trajectory(footsteps)
        step1_start = simple_planner._step_start_idx(1)
        # t=0에서: dcm = zmp[1] + (dcm_eos[1] - zmp[1]) * exp(-omega * step_time)
        expected_dcm_at_step1_start = np.array([0.1, -0.1]) + (
            np.array([0.1, -0.1]) - np.array([0.1, -0.1])
        ) * np.exp(-omega * step_time)
        np.testing.assert_allclose(ref_dcm[step1_start], expected_dcm_at_step1_start, atol=1e-10)

    def test_three_step_backward_recursion(self, simple_planner):
        """3-step 케이스에서 역방향 재귀 검증."""
        footsteps = [(0.0, 0.1), (0.1, -0.1), (0.2, 0.1)]
        omega = simple_planner.omega
        step_time = simple_planner.step_time

        # dcm_eos[2] = (0.2, 0.1)
        # dcm_eos[1] = footsteps[2] + exp(-omega*step_time) * (dcm_eos[2] - footsteps[2])
        #            = (0.2, 0.1) + exp(-omega*0.7) * (0, 0) = (0.2, 0.1)
        # dcm_eos[0] = footsteps[1] + exp(-omega*step_time) * (dcm_eos[1] - footsteps[1])
        exp_neg = np.exp(-omega * step_time)
        dcm_eos_1 = np.array([0.2, 0.1])  # same as footsteps[2]
        dcm_eos_0 = np.array([0.1, -0.1]) + exp_neg * (dcm_eos_1 - np.array([0.1, -0.1]))

        # 첫 스텝 끝에서의 DCM은 dcm_eos[0]에 수렴해야 함
        ref_dcm, _ = simple_planner.compute_dcm_trajectory(footsteps)
        step0_end_idx = simple_planner._step_start_idx(1) - 1
        np.testing.assert_allclose(ref_dcm[step0_end_idx], dcm_eos_0, atol=0.01)


class TestForwardDCM:
    """순방향 DCM 생성 검증."""

    def test_dcm_vel_formula(self, simple_planner):
        """ref_dcm_vel = omega * (ref_dcm - current_zmp) 관계 검증."""
        footsteps = [(0.0, 0.1), (0.1, -0.1)]
        ref_dcm, ref_dcm_vel = simple_planner.compute_dcm_trajectory(footsteps)
        omega = simple_planner.omega

        # 첫 스텝 구간에서 zmp = footsteps[0]
        step0_samples = simple_planner._samples_for(0)
        zmp0 = np.array(footsteps[0])
        for k in range(step0_samples):
            expected_vel = omega * (ref_dcm[k] - zmp0)
            np.testing.assert_allclose(ref_dcm_vel[k], expected_vel, atol=1e-10)

        # 두 번째 스텝 구간에서 zmp = footsteps[1]
        step1_start = simple_planner._step_start_idx(1)
        step1_samples = simple_planner._samples_for(1)
        zmp1 = np.array(footsteps[1])
        for k in range(step1_samples):
            idx = step1_start + k
            expected_vel = omega * (ref_dcm[idx] - zmp1)
            np.testing.assert_allclose(ref_dcm_vel[idx], expected_vel, atol=1e-10)

    def test_dcm_monotonic_within_step(self, simple_planner):
        """단일 스텝 내에서 DCM이 dcm_eos를 향해 수렴하는지 검증."""
        footsteps = [(0.0, 0.0), (0.2, 0.0)]
        ref_dcm, _ = simple_planner.compute_dcm_trajectory(footsteps)

        # 첫 스텝: x 방향으로 DCM이 증가해야 함 (dcm_eos[0].x > footsteps[0].x)
        step0_samples = simple_planner._samples_for(0)
        x_values = ref_dcm[:step0_samples, 0]
        # DCM x는 단조 증가 (zmp=0에서 dcm_eos 방향으로)
        for i in range(1, len(x_values)):
            assert x_values[i] >= x_values[i - 1] - 1e-12


class TestInitDspExtra:
    """init_dsp_extra가 첫 스텝에만 적용되는지 검증."""

    def test_first_step_has_extra_samples(self, planner):
        footsteps = planner.plan_footsteps(3, 0.1, 0.1, np.array([0.0, 0.0]))
        ref_dcm, _ = planner.compute_dcm_trajectory(footsteps)

        # 첫 스텝: int((0.7 + 0.12) / 0.002) = 409 (부동소수점)
        # 나머지: int(0.7 / 0.002) = 350
        total = planner._total_samples(3)
        assert ref_dcm.shape[0] == total
        expected_first = int((planner.step_time + planner.init_dsp_extra) / planner.dt)
        expected_rest = int(planner.step_time / planner.dt)
        assert planner._samples_for(0) == expected_first
        assert planner._samples_for(0) > expected_rest  # 첫 스텝이 더 길어야 함
        assert planner._samples_for(1) == expected_rest
        assert planner._samples_for(2) == expected_rest


class TestNoNaN:
    """NaN이 없는지 검증."""

    def test_no_nan_in_output(self, planner):
        footsteps = planner.plan_footsteps(10, 0.1, 0.1185, np.array([0.035, 0.0]))
        ref_dcm, ref_dcm_vel = planner.compute_dcm_trajectory(footsteps)
        assert not np.any(np.isnan(ref_dcm))
        assert not np.any(np.isnan(ref_dcm_vel))
