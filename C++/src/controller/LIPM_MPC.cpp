#include "controller/LIPM_MPC.hpp"

LIPM_MPC::LIPM_MPC(const LIPM& model, int horizon, double alpha, double gamma)
    : model_(model), N_(horizon), alpha_(alpha), gamma_(gamma),
      qp_(2 * horizon, 0, 0)
{
    Mx_qp_.resize(N_, 3);
    Mu_qp_.resize(N_, N_);
    H_qp_.resize(2 * N_, 2 * N_);
    grad_qp_.resize(2 * N_);

    buildPredictionMatrices();

    qp_.settings.initial_guess =
        proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
}

void LIPM_MPC::buildPredictionMatrices()
{
    Eigen::MatrixXd A = model_.getA(); // 3×3
    Eigen::MatrixXd B = model_.getB(); // 3×1
    Eigen::MatrixXd C = model_.getC(); // 1×3

    Eigen::MatrixXd Px(3 * N_, 3);
    Eigen::MatrixXd Pu(3 * N_, N_);
    Pu.setZero();

    Eigen::MatrixXd A_pow = A;
    for (int i = 0; i < N_; ++i) {
        Px.block(3 * i, 0, 3, 3) = A_pow;

        // Pu(i,j) = A^(i-j)*B  →  j=i: B, j=i-1: A*B, ...
        Eigen::MatrixXd col_val = B;
        for (int j = i; j >= 0; --j) {
            Pu.block(3 * i, j, 3, 1) = col_val;
            col_val = A * col_val;
        }
        A_pow = A * A_pow;
    }

    Mx_qp_.setZero();
    Mu_qp_.setZero();
    for (int i = 0; i < N_; ++i) {
        Mx_qp_.row(i) = C * Px.block(3 * i, 0, 3, 3);
        for (int j = 0; j <= i; ++j)
            Mu_qp_(i, j) = (C * Pu.block(3 * i, j, 3, 1))(0, 0);
    }

    // H = blkdiag(Hx, Hy),  Hx = Hy = alpha*I + gamma*Mu^T*Mu
    Eigen::MatrixXd H_single = alpha_ * Eigen::MatrixXd::Identity(N_, N_)
                              + gamma_ * Mu_qp_.transpose() * Mu_qp_;
    H_qp_.setZero();
    H_qp_.topLeftCorner(N_, N_)     = H_single;
    H_qp_.bottomRightCorner(N_, N_) = H_single;
}

void LIPM_MPC::updateGradient(
    const Eigen::VectorXd& x0, const Eigen::VectorXd& y0,
    const Eigen::VectorXd& zmp_ref_x, const Eigen::VectorXd& zmp_ref_y)
{
    // grad = 2*gamma * Mu^T * (Mx*x0 - zmp_ref)  per axis
    grad_qp_.head(N_) = 2.0 * gamma_ * Mu_qp_.transpose() * (Mx_qp_ * x0 - zmp_ref_x);
    grad_qp_.tail(N_) = 2.0 * gamma_ * Mu_qp_.transpose() * (Mx_qp_ * y0 - zmp_ref_y);
}

std::pair<double, double> LIPM_MPC::solve(
    const Eigen::VectorXd& x0, const Eigen::VectorXd& y0,
    const Eigen::VectorXd& zmp_ref_x, const Eigen::VectorXd& zmp_ref_y)
{
    updateGradient(x0, y0, zmp_ref_x, zmp_ref_y);

    if (!is_initialized_) {
        qp_.init(H_qp_, grad_qp_,
                 std::nullopt, std::nullopt,
                 std::nullopt, std::nullopt, std::nullopt);
        is_initialized_ = true;
    } else {
        qp_.update(std::nullopt, grad_qp_,
                   std::nullopt, std::nullopt,
                   std::nullopt, std::nullopt, std::nullopt);
    }

    qp_.solve();

    // U = [jerk_x(0..N-1), jerk_y(0..N-1)]
    return { qp_.results.x(0), qp_.results.x(N_) };
}
