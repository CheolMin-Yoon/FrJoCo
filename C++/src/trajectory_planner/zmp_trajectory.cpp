#include "trajectory_planner/zmp_trajectory.hpp"
#include <cmath>
#include <algorithm>

ZmpTrajectory::ZmpTrajectory(
    double z_c, double g, double step_time, double dsp_time,
    double dt, double init_dsp_extra)
    : z_c_(z_c), omega_(std::sqrt(g / z_c)),
      step_time_(step_time), dsp_time_(dsp_time),
      ssp_time_(step_time - dsp_time),
      dt_(dt), init_dsp_extra_(init_dsp_extra),
      total_samples_(0), walk_samples_(0), n_steps_(0)
{}

// ── 타이밍 헬퍼 ──

double ZmpTrajectory::stepTimeFor(int i) const {
    return (i == 0) ? step_time_ + init_dsp_extra_ : step_time_;
}

double ZmpTrajectory::dspTimeFor(int i) const {
    return (i == 0) ? dsp_time_ + init_dsp_extra_ : dsp_time_;
}

int ZmpTrajectory::samplesFor(int i) const {
    return static_cast<int>(stepTimeFor(i) / dt_);
}

int ZmpTrajectory::stepStartSample(int i) const {
    int idx = 0;
    for (int j = 0; j < i; ++j) idx += samplesFor(j);
    return idx;
}

int ZmpTrajectory::stepEndSample(int i) const {
    return stepStartSample(i) + samplesFor(i);
}

// ── 발자국 계획 ──

std::vector<Eigen::Vector2d> ZmpTrajectory::planFootsteps(
    int n_steps, double step_length, double step_width, Eigen::Vector2d init_xy)
{
    std::vector<Eigen::Vector2d> footsteps;
    for (int i = 0; i < n_steps; ++i) {
        double x = (i == 0) ? init_xy.x() : init_xy.x() + i * step_length;
        double y = (i == 0) ? step_width : (i % 2 != 0) ? -step_width : step_width;
        footsteps.emplace_back(x, y);
    }
    return footsteps;
}

// ── ZMP ref 생성 (Kajita Preview Control 방식) ──

void ZmpTrajectory::generateZmpRef(
    const std::vector<Eigen::Vector2d>& footsteps, int horizon)
{
    n_steps_ = static_cast<int>(footsteps.size());

    // 전체 보행 시간 계산
    walk_samples_ = 0;
    for (int i = 0; i < n_steps_; ++i)
        walk_samples_ += samplesFor(i);

    // preview horizon 여유분 포함
    total_samples_ = walk_samples_ + horizon;

    zmp_ref_x_ = Eigen::VectorXd::Zero(total_samples_);
    zmp_ref_y_ = Eigen::VectorXd::Zero(total_samples_);

    for (int i = 0; i < n_steps_; ++i) {
        int t_start = stepStartSample(i);
        int t_end   = std::min(stepEndSample(i), total_samples_);
        int dsp_samples = static_cast<int>(dspTimeFor(i) / dt_);
        int ssp_start   = t_start + dsp_samples;

        if (i == 0) {
            // 첫 스텝: 전체 구간 첫 발 위치 (이전 발 없음)
            for (int k = t_start; k < t_end; ++k) {
                zmp_ref_x_(k) = footsteps[0].x();
                zmp_ref_y_(k) = footsteps[0].y();
            }
        } else {
            const auto& prev = footsteps[i - 1];
            const auto& curr = footsteps[i];

            // DSP: 코사인 보간 ramp
            for (int k = 0; k < dsp_samples; ++k) {
                int idx = t_start + k;
                if (idx >= total_samples_) break;
                double alpha = 0.5 * (1.0 - std::cos(M_PI * k / dsp_samples));
                zmp_ref_x_(idx) = (1.0 - alpha) * prev.x() + alpha * curr.x();
                zmp_ref_y_(idx) = (1.0 - alpha) * prev.y() + alpha * curr.y();
            }

            // SSP: 현재 발 위치에 고정
            for (int k = ssp_start; k < t_end; ++k) {
                zmp_ref_x_(k) = curr.x();
                zmp_ref_y_(k) = curr.y();
            }
        }
    }

    // 마지막 발자국 이후 유지
    int last_filled = stepEndSample(n_steps_ - 1);
    if (last_filled < total_samples_) {
        for (int k = last_filled; k < total_samples_; ++k) {
            zmp_ref_x_(k) = footsteps.back().x();
            zmp_ref_y_(k) = footsteps.back().y();
        }
    }
}

// ── ZMP ref 슬라이스 ──

void ZmpTrajectory::getZmpRefSlice(
    int current_idx, int horizon,
    Eigen::VectorXd& zmp_ref_x,
    Eigen::VectorXd& zmp_ref_y) const
{
    zmp_ref_x.resize(horizon);
    zmp_ref_y.resize(horizon);

    for (int k = 0; k < horizon; ++k) {
        int idx = current_idx + k;
        if (idx < total_samples_) {
            zmp_ref_x(k) = zmp_ref_x_(idx);
            zmp_ref_y(k) = zmp_ref_y_(idx);
        } else {
            // 범위 초과 시 마지막 값 유지
            zmp_ref_x(k) = zmp_ref_x_(total_samples_ - 1);
            zmp_ref_y(k) = zmp_ref_y_(total_samples_ - 1);
        }
    }
}
