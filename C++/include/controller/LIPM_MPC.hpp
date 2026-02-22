#pragma once
#include <Eigen/Dense>
#include "config.hpp"
#include "dynamics_model/LIPM.hpp"
#include <proxsuite/proxqp/dense/dense.hpp>

// Kajita 2003 Preview Control 방식 MPC (x/y 통합)
// min  alpha*||U||² + gamma*||Zmp_pred - Zmp_ref||²
// 상태: x=[x,dx,ddx], y=[y,dy,ddy]  입력: U=[jerk_x(0..N-1), jerk_y(0..N-1)] (2N×1)

class LIPM_MPC {
public:
    LIPM_MPC(const LIPM& model, int horizon = MPC_HORIZON,
             double alpha = MPC_ALPHA, double gamma = MPC_GAMMA);

    // x0: [x,dx,ddx] (3×1), y0: [y,dy,ddy] (3×1)
    // zmp_ref_x, zmp_ref_y: 목표 ZMP 시퀀스 (N×1)
    // 반환: {jerk_x, jerk_y} — receding horizon 첫 번째 입력
    std::pair<double, double> solve(
        const Eigen::VectorXd& x0, const Eigen::VectorXd& y0,
        const Eigen::VectorXd& zmp_ref_x, const Eigen::VectorXd& zmp_ref_y);

    Eigen::MatrixXd getHessian()  const { return H_qp_;    }
    Eigen::VectorXd getGradient() const { return grad_qp_; }

private:
    LIPM   model_;
    int    N_;
    double alpha_;
    double gamma_;

    // 단축 예측 행렬 (단일 축)
    Eigen::MatrixXd Mx_qp_;   // N×3
    Eigen::MatrixXd Mu_qp_;   // N×N

    // 통합 QP 행렬 (2N×2N, 2N×1)
    Eigen::MatrixXd H_qp_;    // 2N×2N  블록 대각, 상수
    Eigen::VectorXd grad_qp_; // 2N×1   매 스텝 갱신

    proxsuite::proxqp::dense::QP<double> qp_;
    bool is_initialized_ = false;

    void buildPredictionMatrices();
    void updateGradient(const Eigen::VectorXd& x0, const Eigen::VectorXd& y0,
                        const Eigen::VectorXd& zmp_ref_x, const Eigen::VectorXd& zmp_ref_y);
};
