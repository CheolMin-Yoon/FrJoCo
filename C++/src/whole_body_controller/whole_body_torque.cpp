#include "whole_body_controller/whole_body_torque.hpp"
#include "config.hpp"
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>

WholeBodyTorqueGenerator::WholeBodyTorqueGenerator(int nv, int na, int nc)
    : nv_(nv), na_(na), nc_(nc)
{
    // G = [M   -S^T]  크기: (nv+nc) × (nv+na)
    //     [J_c   0 ]
    G_.setZero(nv_ + nc_, nv_ + na_);
    f_.setZero(nv_ + nc_);

    // 가중치: ddq는 작게, tau는 적당히
    W_.setIdentity(nv_ + na_, nv_ + na_);
    W_.block(0,   0,   nv_, nv_) *= WBT_W_DDQ;  // ddq 페널티
    W_.block(nv_, nv_, na_, na_) *= WBT_W_TAU;  // tau 페널티

    // -S^T: floating base(6)는 0, actuated joints(na)는 -I (고정)
    G_.block(6, nv_, na_, na_) = -Eigen::MatrixXd::Identity(na_, na_);

    J_c_.setZero(nc_, nv_);
    JdotQdot_.setZero(nc_);
}

Eigen::VectorXd WholeBodyTorqueGenerator::compute(
    const pinocchio::Model& model,
    pinocchio::Data& data,
    int right_foot_id,
    int left_foot_id,
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& dq,
    const Eigen::VectorXd& F_hat)
{
    // =======================================================================
    // 1. Pinocchio 동역학 연산
    // =======================================================================
    // data.M (질량 행렬), data.nle (코리올리/중력) 최신화
    pinocchio::crba(model, data, q);
    data.M.triangularView<Eigen::StrictlyLower>() =
        data.M.transpose().triangularView<Eigen::StrictlyLower>();

    pinocchio::nonLinearEffects(model, data, q, dq);

    // JdotQdot 계산에 필요한 자코비안 시간 미분
    pinocchio::computeJointJacobiansTimeVariation(model, data, q, dq);

    Eigen::MatrixXd& M = data.M;
    Eigen::VectorXd& N = data.nle;

    // 오른발 자코비안 및 Jdot*qdot (LOCAL_WORLD_ALIGNED 기준)
    pinocchio::getFrameJacobian(model, data, right_foot_id,
        pinocchio::LOCAL_WORLD_ALIGNED, J_c_.middleRows(0, 6));
    JdotQdot_.segment(0, 6) =
        pinocchio::getFrameClassicalAcceleration(model, data, right_foot_id,
            pinocchio::LOCAL_WORLD_ALIGNED).toVector();

    // 왼발 자코비안 및 Jdot*qdot
    pinocchio::getFrameJacobian(model, data, left_foot_id,
        pinocchio::LOCAL_WORLD_ALIGNED, J_c_.middleRows(6, 6));
    JdotQdot_.segment(6, 6) =
        pinocchio::getFrameClassicalAcceleration(model, data, left_foot_id,
            pinocchio::LOCAL_WORLD_ALIGNED).toVector();

    // =======================================================================
    // 2. 논문 식 (19): G z = f 행렬 블록 조립
    // =======================================================================
    // [ 상단부 ] M(q) 행렬 (-S^T는 생성자에서 고정)
    G_.block(0, 0, nv_, nv_) = M;

    // [ 하단부 ] J_c(q) 행렬 (우하단 0은 이미 0)
    G_.block(nv_, 0, nc_, nv_) = J_c_;

    f_.head(nv_) = -N + J_c_.transpose() * F_hat;
    f_.tail(nc_) = -JdotQdot_;  // ddP_des = 0 (발이 땅에 고정)

    // =======================================================================
    // 3. 논문 식 (20): z = (G^T G + W)^{-1} G^T f
    // =======================================================================
    Eigen::VectorXd z = (G_.transpose() * G_ + W_).ldlt().solve(G_.transpose() * f_);

    // z = [ddq(nv), tau(na)]  →  뒤쪽 na_ 개가 관절 토크
    return z.tail(na_);
}
