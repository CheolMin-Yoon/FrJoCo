// ============================================================
// [보존용] 기존 DCM backward recursion + CoM forward integration
// ============================================================
#include "trajectory_planner/dcm_trajectory.hpp"
#include <cmath>

DcmTrajectory::DcmTrajectory(
    double z_c, double g, double step_time, double dsp_time,
    double dt, double init_dsp_extra)
    : z_c_(z_c), omega_(std::sqrt(g / z_c)),
      step_time_(step_time), dsp_time_(dsp_time),
      ssp_time_(step_time - dsp_time),
      dt_(dt), init_dsp_extra_(init_dsp_extra)
{}

double DcmTrajectory::stepTimeFor(int i) const {
    return (i == 0 && init_dsp_extra_ > 0) ? step_time_ + init_dsp_extra_ : step_time_;
}
double DcmTrajectory::dspTimeFor(int i) const {
    return (i == 0 && init_dsp_extra_ > 0) ? dsp_time_ + init_dsp_extra_ : dsp_time_;
}
int DcmTrajectory::samplesFor(int i) const {
    return static_cast<int>(stepTimeFor(i) / dt_);
}
int DcmTrajectory::totalSamples(int n_steps) const {
    int total = 0;
    for (int i = 0; i < n_steps; ++i) total += samplesFor(i);
    return total;
}
int DcmTrajectory::stepStartIdx(int step) const {
    int idx = 0;
    for (int i = 0; i < step; ++i) idx += samplesFor(i);
    return idx;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> DcmTrajectory::computeDcmTrajectory(
    const std::vector<Eigen::Vector2d>& footsteps)
{
    int n_steps = static_cast<int>(footsteps.size());
    int total   = totalSamples(n_steps);

    Eigen::MatrixXd ref_dcm(total, 2), ref_dcm_vel(total, 2);

    std::vector<Eigen::Vector2d> dcm_eos(n_steps);
    dcm_eos[n_steps - 1] = footsteps[n_steps - 1];
    for (int i = n_steps - 2; i >= 0; --i) {
        double exp_neg = std::exp(-omega_ * stepTimeFor(i + 1));
        dcm_eos[i] = footsteps[i + 1] + (dcm_eos[i + 1] - footsteps[i + 1]) * exp_neg;
    }

    for (int i = 0; i < n_steps; ++i) {
        int start = stepStartIdx(i);
        int samps = samplesFor(i);
        double st = stepTimeFor(i);
        Eigen::Vector2d zmp    = footsteps[i];
        Eigen::Vector2d xi_end = dcm_eos[i];

        for (int k = 0; k < samps; ++k) {
            double t_rem = st - k * dt_;
            Eigen::Vector2d dcm = zmp + (xi_end - zmp) * std::exp(-omega_ * t_rem);
            ref_dcm.row(start + k)     = dcm;
            ref_dcm_vel.row(start + k) = omega_ * (dcm - zmp);
        }
    }
    return {ref_dcm, ref_dcm_vel};
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> DcmTrajectory::computeComTrajectory(
    const Eigen::MatrixXd& ref_dcm, Eigen::Vector2d init_com_xy)
{
    int N = ref_dcm.rows();
    Eigen::MatrixXd com_pos(N, 2), com_vel(N, 2);
    Eigen::Vector2d cur = init_com_xy;

    for (int k = 0; k < N; ++k) {
        Eigen::Vector2d dx = omega_ * (ref_dcm.row(k).transpose() - cur);
        com_pos.row(k) = cur;
        com_vel.row(k) = dx;
        cur += dx * dt_;
    }
    return {com_pos, com_vel};
}
