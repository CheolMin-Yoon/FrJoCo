#pragma once
#include "constraints/constraint_core.hpp"

class FrictionCone : public ConstraintCore {
public:
    explicit FrictionCone(double mu);

    void setFriction(double mu) { mu_ = mu; }

    void update(const Eigen::VectorXd& state) override;
    Eigen::MatrixXd getA()          const override { return A_; }
    Eigen::VectorXd getLowerBound() const override { return l_; }
    Eigen::VectorXd getUpperBound() const override { return u_; }

private:
    double mu_;
    Eigen::MatrixXd A_;
    Eigen::VectorXd l_;
    Eigen::VectorXd u_;

    void buildConstraint();
};
