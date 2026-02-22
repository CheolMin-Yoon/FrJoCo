#pragma once

#include <Eigen/Dense>
#include <proxsuite/proxqp/dense/dense.hpp>

class ForceOptimizer {
public:
    // 최적화 변수 12개, 부등식 제약조건 마찰콘 10 + CoP 8 = 18개
    ForceOptimizer(int num_vars = 12, int num_constraints = 18);

    void updateObjective(const Eigen::MatrixXd& K,
                         const Eigen::VectorXd& u_vec,
                         const Eigen::MatrixXd& W);

    void updateConstraints(const Eigen::MatrixXd& A,
                           const Eigen::VectorXd& l,
                           const Eigen::VectorXd& u);

    bool solve();

    // 최적화 결과 — qp_.results.x를 여기에 저장
    Eigen::VectorXd opt_F_;

private:
    int num_vars_;
    int n_ineq_;

    proxsuite::proxqp::dense::QP<double> qp_;

    Eigen::MatrixXd H_;
    Eigen::VectorXd g_;
    Eigen::MatrixXd A_;
    Eigen::VectorXd l_;
    Eigen::VectorXd u_;

    bool is_initialized_;
};
