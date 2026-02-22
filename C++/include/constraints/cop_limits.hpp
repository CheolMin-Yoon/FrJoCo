#pragma once
#include "constraints/constraint_core.hpp"
#include <Eigen/src/Core/Matrix.h>

class CoPLimits : public ConstraintCore {
    public:

    CoPLimits(double dX_max, double dX_min, double dY_max, double dY_min);

    void update(const Eigen::VectorXd& contact_state) override;

    Eigen::MatrixXd getA()          const override { return A_; }
    Eigen::VectorXd getLowerBound() const override { return l_; }
    Eigen::VectorXd getUpperBound() const override { return u_; }

    private:

    double dX_max_, dX_min_;
    double dY_max_, dY_min_;

    Eigen::MatrixXd A_;
    Eigen::VectorXd l_;
    Eigen::VectorXd u_;
    
    void buildConstraint();
};