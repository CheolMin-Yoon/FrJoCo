#pragma once
#include "constraints/constraint_core.hpp"
#include <vector>

// 지지 다각형(Support Polygon) 볼록 껍질 제약
// ZMP가 발 접촉 영역 내에 있어야 함

class ConvexHull : public ConstraintCore {
public:
    explicit ConvexHull(const std::vector<Eigen::Vector2d>& contact_points);

    void setContactPoints(const std::vector<Eigen::Vector2d>& contact_points);

    void update(const Eigen::VectorXd& state) override;
    Eigen::MatrixXd getA()          const override { return A_; }
    Eigen::VectorXd getLowerBound() const override { return l_; }
    Eigen::VectorXd getUpperBound() const override { return u_; }

private:
    std::vector<Eigen::Vector2d> contact_points_;
    Eigen::MatrixXd A_;
    Eigen::VectorXd l_;
    Eigen::VectorXd u_;

    void buildConstraint();
};
