#pragma once
#include <Eigen/Dense>
#include <vector>
#include "config.hpp"

// 발 궤적 생성
// XY: Cycloid, Z: 5th-order Bezier
// WBC_DT(1ms) 해상도로 위치 + 속도 + 가속도 생성

struct FootTrajectoryResult {
    Eigen::MatrixXd left_pos;   // (N, 3)
    Eigen::MatrixXd right_pos;  // (N, 3)
    Eigen::MatrixXd left_vel;   // (N, 3)
    Eigen::MatrixXd right_vel;  // (N, 3)
    Eigen::MatrixXd left_acc;   // (N, 3)
    Eigen::MatrixXd right_acc;  // (N, 3)
};

class FootTrajectory {
public:
    FootTrajectory(double step_time      = STEP_TIME,
                   double dsp_time       = DSP_TIME,
                   double dt             = WBC_DT,
                   double step_height    = STEP_HEIGHT,
                   double init_dsp_extra = 0.0);

    // 기존 호환: 위치만 반환
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> compute(
        const std::vector<Eigen::Vector2d>& footsteps,
        Eigen::Vector3d init_lf,
        Eigen::Vector3d init_rf,
        double step_length);

    // 위치 + 속도 + 가속도 반환
    FootTrajectoryResult computeFull(
        const std::vector<Eigen::Vector2d>& footsteps,
        Eigen::Vector3d init_lf,
        Eigen::Vector3d init_rf,
        double step_length);

    // 타이밍 헬퍼
    double stepTimeFor(int i) const;
    double dspTimeFor(int i)  const;
    int    samplesFor(int i)  const;
    int    totalSamples(int n_steps) const;
    int    stepStartIdx(int step)    const;

private:
    double step_time_, dsp_time_, ssp_time_;
    double dt_, step_height_, init_dsp_extra_;

    // Bezier Z: 위치, 1차 미분(ds 기준), 2차 미분(ds 기준)
    double bezierZ(double s, double gz) const;
    double bezierZ_ds(double s) const;    // dB/ds (gz 독립 부분만, step_height_ 스케일)
    double bezierZ_dds(double s) const;   // d²B/ds² (같은 방식)
};
