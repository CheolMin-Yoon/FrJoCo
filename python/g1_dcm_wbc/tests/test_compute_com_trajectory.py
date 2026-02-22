"""Task 2.4: compute_com_trajectory 단위 테스트

오일러 적분으로 ref_com_pos, ref_com_vel 생성 검증.
DCM-CoM 관계: com_vel[k] = omega * (ref_dcm[k] - com_pos[k])
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
    """init_dsp_extra=0인 단순 플래너."""
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

    def test_shape_matches_ref_dcm(self, planner):
        footsteps = planner.plan_footsteps(5, 0.1, 0.1185, np.array([0.0, 0.0]))
        ref_dcm, _ = planner.compute_dcm_trajectory(footsteps)
        init_com = np.array([0.0, 0.0])
        ref_com_pos, ref_com_vel = planner.compute_com_trajectory(ref_dcm, init_com)
        assert ref_com_pos.shape == ref_dcm.shape
        assert ref_com_vel.shape == ref_dcm.shape

    def test_single_step_shape(self, simple_planner):
        footsteps = [(0.0, 0.1)]
        ref_dcm, _ = simple_planner.compute_dcm_trajectory(footsteps)
        ref_com_pos, ref_com_vel = simple_planner.compute_com_trajectory(
            ref_dcm, np.array([0.0, 0.0])
        )
        assert ref_com_pos.shape == (simple_planner._samples_for(0), 2)
        assert ref_com_vel.shape == (simple_planner._samples_for(0), 2)


class TestEulerIntegration:
    """오일러 적분 관계 검증."""

    def test_com_pos_euler_step(self, simple_planner):
        """com_pos[k+1] = com_pos[k] + com_vel[k] * dt 검증."""
        footsteps = [(0.0, 0.1), (0.1, -0.1)]
        ref_dcm, _ = simple_planner.compute_dcm_trajectory(footsteps)
        init_com = np.array([0.0, 0.0])
        ref_com_pos, ref_com_vel = simple_planner.compute_com_trajectory(ref_dcm, init_com)

        dt = simple_planner.dt
        for k in range(len(ref_com_pos) - 1):
            expected_next = ref_com_pos[k] + ref_com_vel[k] * dt
            np.testing.assert_allclose(ref_com_pos[k + 1], expected_next, atol=1e-12)

    def test_initial_com_position(self, simple_planner):
        """첫 번째 CoM 위치가 init_com_xy와 일치하는지 검증."""
        footsteps = [(0.0, 0.1), (0.1, -0.1)]
        ref_dcm, _ = simple_planner.compute_dcm_trajectory(footsteps)
        init_com = np.array([0.035, 0.01])
        ref_com_pos, _ = simple_planner.compute_com_trajectory(ref_dcm, init_com)
        np.testing.assert_allclose(ref_com_pos[0], init_com, atol=1e-12)


class TestDCMComRelation:
    """DCM-CoM 관계 일관성 검증 (Property 3).

    com_vel[k] = omega * (ref_dcm[k] - com_pos[k])
    """

    def test_velocity_formula(self, simple_planner):
        """각 타임스텝에서 com_vel = omega * (dcm - com) 검증."""
        footsteps = [(0.0, 0.1), (0.1, -0.1), (0.2, 0.1)]
        ref_dcm, _ = simple_planner.compute_dcm_trajectory(footsteps)
        init_com = np.array([0.0, 0.0])
        ref_com_pos, ref_com_vel = simple_planner.compute_com_trajectory(ref_dcm, init_com)

        omega = simple_planner.omega
        for k in range(len(ref_com_pos)):
            expected_vel = omega * (ref_dcm[k] - ref_com_pos[k])
            np.testing.assert_allclose(ref_com_vel[k], expected_vel, atol=1e-12)

    def test_velocity_formula_with_offset_init(self, planner):
        """초기 CoM이 원점이 아닌 경우에도 관계 성립 검증."""
        footsteps = planner.plan_footsteps(3, 0.1, 0.1185, np.array([0.035, 0.0]))
        ref_dcm, _ = planner.compute_dcm_trajectory(footsteps)
        init_com = np.array([0.035, 0.0])
        ref_com_pos, ref_com_vel = planner.compute_com_trajectory(ref_dcm, init_com)

        omega = planner.omega
        for k in range(len(ref_com_pos)):
            expected_vel = omega * (ref_dcm[k] - ref_com_pos[k])
            np.testing.assert_allclose(ref_com_vel[k], expected_vel, atol=1e-12)


class TestComFollowsDCM:
    """CoM이 DCM을 추종하는지 검증."""

    def test_com_converges_toward_dcm(self, simple_planner):
        """CoM이 DCM 방향으로 이동하는지 검증 (단일 스텝, 정적 DCM)."""
        # 상수 DCM으로 테스트: CoM이 DCM에 수렴해야 함
        n = 500
        dcm_target = np.array([1.0, 0.5])
        ref_dcm = np.tile(dcm_target, (n, 1))
        init_com = np.array([0.0, 0.0])
        ref_com_pos, _ = simple_planner.compute_com_trajectory(ref_dcm, init_com)

        # 마지막 CoM이 DCM에 가까워져야 함
        dist_start = np.linalg.norm(ref_com_pos[0] - dcm_target)
        dist_end = np.linalg.norm(ref_com_pos[-1] - dcm_target)
        assert dist_end < dist_start


class TestNoNaN:
    """NaN이 없는지 검증."""

    def test_no_nan_in_output(self, planner):
        footsteps = planner.plan_footsteps(10, 0.1, 0.1185, np.array([0.035, 0.0]))
        ref_dcm, _ = planner.compute_dcm_trajectory(footsteps)
        init_com = np.array([0.035, 0.0])
        ref_com_pos, ref_com_vel = planner.compute_com_trajectory(ref_dcm, init_com)
        assert not np.any(np.isnan(ref_com_pos))
        assert not np.any(np.isnan(ref_com_vel))
