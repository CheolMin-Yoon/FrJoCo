#pragma once
#include "constraints/constraint_core.hpp"

// 태스크 공간 제약 (관절 한계, 속도 한계 등)
// A * x, l <= A*x <= u

class TaskSpace : public ConstraintCore {
public:
    TaskSpace(int dof);

    void update(const Eigen::VectorXd& state) override;
    Eigen::MatrixXd getA()          const override { return A_; }
    Eigen::VectorXd getLowerBound() const override { return l_; }
    Eigen::VectorXd getUpperBound() const override { return u_; }

private:
    int dof_;
    Eigen::MatrixXd A_;
    Eigen::VectorXd l_;
    Eigen::VectorXd u_;
};
