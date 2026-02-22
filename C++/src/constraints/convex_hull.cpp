#include "constraints/convex_hull.hpp"

ConvexHull::ConvexHull(const std::vector<Eigen::Vector2d>& contact_points)
    : contact_points_(contact_points)
{
    buildConstraint();
}

void ConvexHull::setContactPoints(const std::vector<Eigen::Vector2d>& contact_points)
{
    contact_points_ = contact_points;
    buildConstraint();
}

void ConvexHull::buildConstraint()
{
    // TODO: 접촉점으로부터 볼록 껍질 half-plane 부등식 구성
    // A * [zmp_x; zmp_y] <= b  형태
}

void ConvexHull::update(const Eigen::VectorXd& state)
{
    // TODO: 접촉 상태 변화 시 재구성
    (void)state;
}
