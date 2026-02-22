#pragma once
#include <Eigen/Dense>

class ConstraintCore {
public:
    virtual ~ConstraintCore() = default;

    virtual void update(const Eigen::VectorXd& state) = 0;

    virtual Eigen::MatrixXd getA()          const = 0;
    virtual Eigen::VectorXd getLowerBound() const = 0;
    virtual Eigen::VectorXd getUpperBound() const = 0;
};
