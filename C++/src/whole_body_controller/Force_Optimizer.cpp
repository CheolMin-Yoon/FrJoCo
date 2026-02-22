#include "whole_body_controller/Force_Optimizer.hpp"
#include <iostream>

ForceOptimizer::ForceOptimizer(int num_vars, int num_constraints)
    : num_vars_(num_vars),
      n_ineq_(num_constraints),
      qp_(num_vars, 0, num_constraints),
      opt_F_(Eigen::VectorXd::Zero(num_vars)),
      is_initialized_(false)
{
    H_.resize(num_vars_, num_vars_);
    g_.resize(num_vars_);
    A_.resize(n_ineq_, num_vars_);
    l_.resize(n_ineq_);
    u_.resize(n_ineq_);

    qp_.settings.eps_abs = 1e-4;
    qp_.settings.eps_rel = 1e-4;
    qp_.settings.max_iter = 1000;
}

void ForceOptimizer::updateObjective(const Eigen::MatrixXd& K,
                                     const Eigen::VectorXd& u_vec,
                                     const Eigen::MatrixXd& W)
{
    // min (KF - u)^T(KF - u) + F^T W F
    // H = 2*(K^T K + W),  g = -2*K^T u
    H_ = 2.0 * (K.transpose() * K + W);
    g_ = -2.0 * (K.transpose() * u_vec);
}

void ForceOptimizer::updateConstraints(const Eigen::MatrixXd& A,
                                       const Eigen::VectorXd& l,
                                       const Eigen::VectorXd& u)
{
    A_ = A;
    l_ = l;
    u_ = u;
}

bool ForceOptimizer::solve()
{
    if (!is_initialized_) {
        qp_.init(H_, g_, std::nullopt, std::nullopt, A_, l_, u_);
        is_initialized_ = true;
    } else {
        qp_.update(H_, g_, std::nullopt, std::nullopt, A_, l_, u_);
    }

    qp_.solve();

    if (qp_.results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
        opt_F_ = qp_.results.x;  // primal solution → opt_F_
        return true;
    }

    std::cerr << "[ForceOptimizer] solve failed, status="
              << static_cast<int>(qp_.results.info.status) << "\n";
    return false;
}
