#pragma once

#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

// WholeBodyTorqueGenerator
//
// 논문 식 (20): 최적 지면 반력 F_hat으로부터 관절 토크 계산
//
// 전신 동역학:
//   M(q)*ddq + h(q,dq) = S^T * tau + J_c^T * F
//
// 결정변수: x = [ddq (nv), tau (na)]
// 목적함수: min x^T W x
// 등식 제약: G * x = f
//   G = [M  -S^T]   f = J_c^T * F - h
//       [S    0 ]       [0          ]  (actuated joints only)
//
// nv = 29 (floating base 6 + joints 23)
// na = 23 (actuated joints)
// nc = 접촉 자코비안 행 수 (접촉점 수 × 3 or 6)

class WholeBodyTorqueGenerator {
public:
    // nv: 속도 차원 (29), na: 액추에이터 차원 (23), nc: 접촉 공간 차원
    WholeBodyTorqueGenerator(int nv = 29, int na = 23, int nc = 12);

    // F_hat: ForceOptimizer에서 나온 최적 지면 반력 (12×1)
    // q, dq: Pinocchio 일반화 좌표/속도
    // right_foot_id, left_foot_id: Pinocchio frame ID
    Eigen::VectorXd compute(const pinocchio::Model& model,
                            pinocchio::Data& data,
                            int right_foot_id,
                            int left_foot_id,
                            const Eigen::VectorXd& q,
                            const Eigen::VectorXd& dq,
                            const Eigen::VectorXd& F_hat);

private:
    int nv_;  // 속도 차원 (floating base 6 + joints 23 = 29)
    int na_;  // 액추에이터 차원 (23)
    int nc_;  // 접촉 공간 차원 (접촉점 수 × 3)

    // 등식 제약 행렬: G * [ddq; tau] = f
    Eigen::MatrixXd G_;       // (nv + nc) × (nv + na)
    Eigen::VectorXd f_;       // (nv + nc) × 1

    // 가중치 행렬 (regularization)
    Eigen::MatrixXd W_;       // (nv + na) × (nv + na)

    // 접촉 자코비안
    Eigen::MatrixXd J_c_;     // nc × nv
    Eigen::VectorXd JdotQdot_; // nc × 1  (J_dot * q_dot)
};
