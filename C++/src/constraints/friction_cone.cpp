#include "constraints/friction_cone.hpp"
#include <limits>

FrictionCone::FrictionCone(double mu)
    : mu_(mu)
{
    // f = [fx, fy, fz]
    // 선형화된 마찰 원뿔: |fx| <= mu*fz, |fy| <= mu*fz, fz >= 0
    // A * f, l <= A*f <= u 형태
    A_.resize(5, 3);
    l_.resize(5);
    u_.resize(5);

    buildConstraint();
}

void FrictionCone::buildConstraint()
{
    A_.setZero();

    // mu_eff = mu / sqrt(2)
    double mu_eff = mu_ / std::sqrt(2.0);
    double INF = std::numeric_limits<double>::infinity();
    
    // 수식 1: 1 * F_x - mu_eff * F_z <= 0
    A_(0, 0) = 1.0; A_(0, 2) = -mu_eff; l_(0) = -INF; u_(0) = 0.0;
    
    // 수식 2: -1 * F_x - mu_eff * F_z <= 0
    A_(1, 0) = -1.0; A_(1, 2) = -mu_eff;  l_(1) = -INF; u_(1) = 0.0;
    
    // 수식 3: 1 * F_y - mu_eff * F_z <= 0
    A_(2, 1) = 1.0;  A_(2, 2) = -mu_eff;  l_(2) = -INF; u_(2) = 0.0;
    
    // 수식 4: -1 * F_y - mu_eff * F_z <= 0
    A_(3, 1) = -1.0; A_(3, 2) = -mu_eff;  l_(3) = -INF; u_(3) = 0.0;
    
    // 수식 5: 0 <= F_z <= INF (지면 반력은 무조건 양수)
    A_(4, 2) = 1.0;                       l_(4) = 0.0;  u_(4) = INF;
}

void FrictionCone::update(const Eigen::VectorXd& state)
{
    // TODO: 상태에 따른 동적 업데이트 (필요 시)
    (void)state;
}
