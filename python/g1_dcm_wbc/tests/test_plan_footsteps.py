"""Task 2.2: plan_footsteps 단위 테스트"""

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


class TestPlanFootstepsLength:
    """반환 리스트 길이가 n_steps와 일치하는지 검증."""

    def test_returns_n_steps_elements(self, planner):
        result = planner.plan_footsteps(20, 0.1, 0.1185, np.array([0.035, 0.0]))
        assert len(result) == 20

    def test_single_step(self, planner):
        result = planner.plan_footsteps(1, 0.1, 0.1185, np.array([0.0, 0.0]))
        assert len(result) == 1

    def test_zero_steps(self, planner):
        result = planner.plan_footsteps(0, 0.1, 0.1185, np.array([0.0, 0.0]))
        assert len(result) == 0


class TestPlanFootstepsFirstStep:
    """첫 번째 발자국 (왼발) 좌표 검증."""

    def test_first_step_x_equals_init(self, planner):
        init_xy = np.array([0.035, 0.0])
        result = planner.plan_footsteps(5, 0.1, 0.1185, init_xy)
        assert np.isclose(result[0][0], 0.035)

    def test_first_step_y_equals_step_width(self, planner):
        init_xy = np.array([0.035, 0.0])
        result = planner.plan_footsteps(5, 0.1, 0.1185, init_xy)
        assert np.isclose(result[0][1], 0.1185)


class TestPlanFootstepsAlternation:
    """좌우 교대 패턴 검증: 짝수 인덱스=왼발(+y), 홀수 인덱스=오른발(-y)."""

    def test_odd_index_is_right_foot(self, planner):
        init_xy = np.array([0.0, 0.0])
        result = planner.plan_footsteps(6, 0.1, 0.1, init_xy)
        # i=1: 오른발 → y = -step_width
        assert np.isclose(result[1][1], -0.1)
        # i=3: 오른발
        assert np.isclose(result[3][1], -0.1)

    def test_even_index_is_left_foot(self, planner):
        init_xy = np.array([0.0, 0.0])
        result = planner.plan_footsteps(6, 0.1, 0.1, init_xy)
        # i=0: 왼발 → y = +step_width
        assert np.isclose(result[0][1], 0.1)
        # i=2: 왼발
        assert np.isclose(result[2][1], 0.1)
        # i=4: 왼발
        assert np.isclose(result[4][1], 0.1)


class TestPlanFootstepsXProgression:
    """X 좌표 전진 검증."""

    def test_x_increases_by_step_length(self, planner):
        init_xy = np.array([0.0, 0.0])
        step_length = 0.1
        result = planner.plan_footsteps(5, step_length, 0.1, init_xy)
        # i=0: x = init_xy[0] = 0.0
        assert np.isclose(result[0][0], 0.0)
        # i=1: x = 0.0 + 1*0.1 = 0.1
        assert np.isclose(result[1][0], 0.1)
        # i=2: x = 0.0 + 2*0.1 = 0.2
        assert np.isclose(result[2][0], 0.2)

    def test_x_with_nonzero_init(self, planner):
        init_xy = np.array([0.5, 0.0])
        result = planner.plan_footsteps(3, 0.2, 0.1, init_xy)
        assert np.isclose(result[0][0], 0.5)
        assert np.isclose(result[1][0], 0.7)
        assert np.isclose(result[2][0], 0.9)


class TestPlanFootstepsReturnType:
    """반환 타입이 (x, y) 튜플 리스트인지 검증."""

    def test_returns_list_of_tuples(self, planner):
        result = planner.plan_footsteps(3, 0.1, 0.1, np.array([0.0, 0.0]))
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
