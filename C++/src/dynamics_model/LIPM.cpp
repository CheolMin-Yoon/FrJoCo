#include "dynamics_model/LIPM.hpp"
#include <cmath>

LIPM::LIPM(double z_c, double dt, double gravity)
    : z_c_(z_c), g_(gravity), omega_(std::sqrt(gravity / z_c))
{
    // 이산시간 Ad (3x3)
    Ad_ = Eigen::MatrixXd::Identity(3, 3);
    Ad_(0, 1) = dt;
    Ad_(0, 2) = 0.5 * dt * dt;
    Ad_(1, 2) = dt;

    // 이산시간 Bd (3x1)
    Bd_.resize(3, 1);
    Bd_(0, 0) = std::pow(dt, 3) / 6.0;
    Bd_(1, 0) = std::pow(dt, 2) / 2.0;
    Bd_(2, 0) = dt;

    // 출력 행렬 Cd (1x3)
    Cd_.resize(1, 3);
    Cd_ << 1.0, 0.0, -z_c_ / g_;
}
