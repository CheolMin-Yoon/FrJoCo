#include "trajectory_planner/foot_trajectory.hpp"
#include <cmath>
#include <algorithm>

FootTrajectory::FootTrajectory(
    double step_time, double dsp_time, double dt,
    double step_height, double init_dsp_extra)
    : step_time_(step_time), dsp_time_(dsp_time),
      ssp_time_(step_time - dsp_time),
      dt_(dt), step_height_(step_height),
      init_dsp_extra_(init_dsp_extra)
{}

double FootTrajectory::stepTimeFor(int i) const {
    return (i == 0) ? step_time_ + init_dsp_extra_ : step_time_;
}
double FootTrajectory::dspTimeFor(int i) const {
    return (i == 0) ? dsp_time_ + init_dsp_extra_ : dsp_time_;
}
int FootTrajectory::samplesFor(int i) const {
    return static_cast<int>(stepTimeFor(i) / dt_);
}
int FootTrajectory::totalSamples(int n_steps) const {
    int total = 0;
    for (int i = 0; i < n_steps; ++i) total += samplesFor(i);
    return total;
}
int FootTrajectory::stepStartIdx(int step) const {
    int idx = 0;
    for (int i = 0; i < step; ++i) idx += samplesFor(i);
    return idx;
}

// ── 5차 Bezier Z ──
// 제어점: P = [gz, gz, gz+h, gz+h, gz, gz]
// B(s) = gz + h * [10(1-s)^3 s^2 + 10(1-s)^2 s^3]
//       = gz + h * 10 s^2 (1-s)^2 [(1-s) + s]
//       = gz + h * 10 s^2 (1-s)^2
// 정리: B(s) = gz + h * 10 s^2 (1-s)^2
// 실제로 전개하면: B(s) = gz + h*(10s^2 - 20s^3 + 10s^4)
// 아... 원래 코드의 제어점을 다시 보면:
// c2 = 10(1-s)^3 s^2 * (gz+h), c3 = 10(1-s)^2 s^3 * (gz+h)
// 나머지는 gz. 그래서 h 부분만 추출:
// B_h(s) = h * [10(1-s)^3 s^2 + 10(1-s)^2 s^3]
//        = h * 10 s^2 (1-s)^2 [(1-s) + s]
//        = h * 10 s^2 (1-s)^2

double FootTrajectory::bezierZ(double s, double gz) const
{
    // B(s) = gz + step_height_ * 10 * s^2 * (1-s)^2
    double u = 1.0 - s;
    return gz + step_height_ * 10.0 * s * s * u * u;
}

// dB/ds = step_height_ * 10 * d/ds[s^2(1-s)^2]
//       = step_height_ * 10 * [2s(1-s)^2 - 2s^2(1-s)]
//       = step_height_ * 10 * 2s(1-s)[(1-s) - s]
//       = step_height_ * 20 * s(1-s)(1-2s)
double FootTrajectory::bezierZ_ds(double s) const
{
    return step_height_ * 20.0 * s * (1.0 - s) * (1.0 - 2.0 * s);
}

// d²B/ds² = step_height_ * 20 * d/ds[s(1-s)(1-2s)]
// f(s) = s(1-s)(1-2s) = s - 3s^2 + 2s^3 (전개)
// f'(s) = 1 - 6s + 6s^2
// d²B/ds² = step_height_ * 20 * (1 - 6s + 6s^2)
double FootTrajectory::bezierZ_dds(double s) const
{
    return step_height_ * 20.0 * (1.0 - 6.0 * s + 6.0 * s * s);
}

// ── 기존 호환: 위치만 반환 ──
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> FootTrajectory::compute(
    const std::vector<Eigen::Vector2d>& footsteps,
    Eigen::Vector3d init_lf, Eigen::Vector3d init_rf, double step_length)
{
    auto result = computeFull(footsteps, init_lf, init_rf, step_length);
    return {result.left_pos, result.right_pos};
}

// ── 위치 + 속도 + 가속도 ──
FootTrajectoryResult FootTrajectory::computeFull(
    const std::vector<Eigen::Vector2d>& footsteps,
    Eigen::Vector3d init_lf, Eigen::Vector3d init_rf, double step_length)
{
    int n_steps = static_cast<int>(footsteps.size());
    int total   = totalSamples(n_steps);

    FootTrajectoryResult res;
    res.left_pos.setZero(total, 3);  res.right_pos.setZero(total, 3);
    res.left_vel.setZero(total, 3);  res.right_vel.setZero(total, 3);
    res.left_acc.setZero(total, 3);  res.right_acc.setZero(total, 3);

    Eigen::Vector3d left_pos = init_lf, right_pos = init_rf;
    double gz_lf = init_lf.z(), gz_rf = init_rf.z();
    double foot_x_start = init_lf.x();

    std::vector<Eigen::Vector2d> foot_targets;
    for (int i = 0; i < n_steps; ++i) {
        double fx = (i == 0) ? foot_x_start : foot_x_start + i * step_length;
        foot_targets.emplace_back(fx, footsteps[i].y());
    }

    for (int i = 0; i < n_steps; ++i) {
        int    start       = stepStartIdx(i);
        int    samps       = samplesFor(i);
        double dsp         = dspTimeFor(i);
        double st          = stepTimeFor(i);
        double ssp         = st - dsp;
        bool   right_swing = (i % 2 == 0);
        bool   has_swing   = (i + 1 < n_steps);

        Eigen::Vector2d swing_target = has_swing
            ? foot_targets[i + 1] : Eigen::Vector2d(-1e9, -1e9);

        // swing 변위 (XY)
        double dx_swing = 0, dy_swing = 0;
        if (has_swing) {
            if (right_swing) {
                dx_swing = swing_target.x() - right_pos.x();
                dy_swing = swing_target.y() - right_pos.y();
            } else {
                dx_swing = swing_target.x() - left_pos.x();
                dy_swing = swing_target.y() - left_pos.y();
            }
        }

        for (int k = 0; k < samps; ++k) {
            double t   = k * dt_;
            int    idx = start + k;

            // DSP 구간 또는 swing 없음 → 정지
            if (t < dsp || !has_swing) {
                res.left_pos.row(idx)  = left_pos;
                res.right_pos.row(idx) = right_pos;
                // vel, acc = 0 (이미 setZero)
                continue;
            }

            double phase = std::min((t - dsp) / ssp, 1.0);
            double theta = 2.0 * M_PI * phase;

            // ── Cycloid XY ──
            // 위치: c = (θ - sinθ) / (2π)
            double cycloid = (theta - std::sin(theta)) / (2.0 * M_PI);

            // 속도: dc/dt = (1 - cosθ) / ssp
            double cycloid_vel = (1.0 - std::cos(theta)) / ssp;

            // 가속도: d²c/dt² = 2π sinθ / ssp²
            double cycloid_acc = 2.0 * M_PI * std::sin(theta) / (ssp * ssp);

            // ── Bezier Z ──
            // ds/dt = 1/ssp
            double ds_dt = 1.0 / ssp;
            double z_pos = right_swing ? bezierZ(phase, gz_rf) : bezierZ(phase, gz_lf);
            double z_vel = bezierZ_ds(phase) * ds_dt;
            double z_acc = bezierZ_dds(phase) * ds_dt * ds_dt;

            if (right_swing) {
                double px = right_pos.x() + dx_swing * cycloid;
                double py = right_pos.y() + dy_swing * cycloid;
                res.right_pos.row(idx) = Eigen::Vector3d(px, py, z_pos);
                res.right_vel.row(idx) = Eigen::Vector3d(dx_swing * cycloid_vel,
                                                          dy_swing * cycloid_vel,
                                                          z_vel);
                res.right_acc.row(idx) = Eigen::Vector3d(dx_swing * cycloid_acc,
                                                          dy_swing * cycloid_acc,
                                                          z_acc);
                res.left_pos.row(idx)  = left_pos;
            } else {
                double px = left_pos.x() + dx_swing * cycloid;
                double py = left_pos.y() + dy_swing * cycloid;
                res.left_pos.row(idx)  = Eigen::Vector3d(px, py, z_pos);
                res.left_vel.row(idx)  = Eigen::Vector3d(dx_swing * cycloid_vel,
                                                          dy_swing * cycloid_vel,
                                                          z_vel);
                res.left_acc.row(idx)  = Eigen::Vector3d(dx_swing * cycloid_acc,
                                                          dy_swing * cycloid_acc,
                                                          z_acc);
                res.right_pos.row(idx) = right_pos;
            }
        }

        if (has_swing) {
            if (right_swing)
                right_pos = Eigen::Vector3d(swing_target.x(), swing_target.y(), gz_rf);
            else
                left_pos  = Eigen::Vector3d(swing_target.x(), swing_target.y(), gz_lf);
        }
    }
    return res;
}
