#pragma once

#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include "config.hpp"
#include "dynamics_model/com_dynamics.hpp"
#include "whole_body_controller/Force_Optimizer.hpp"
#include "whole_body_controller/whole_body_torque.hpp"
#include "whole_body_controller/whole_body_ik.hpp"
#include "whole_body_controller/tasks/balance_task.hpp"
#include "constraints/friction_cone.hpp"
#include "constraints/cop_limits.hpp"

// ============================================================
// Layer 3: Whole Body Controller
//
// 두 가지 모드:
//   (A) IK + PD: WholeBodyIK → PD 토크 (위치 제어, 디버깅용)
//   (B) DBFC:    Task → CenterOfMass → ForceOptimizer → WholeBodyTorque
//                (힘 제어, 논문 식 19-20)
//
// 메인 루프에서 매 WBC 스텝(1kHz)마다 호출
// ============================================================

class WBC {
public:
    WBC(int nv = 29, int na = 23, int nc = 12);

    // ── IK 모드 (경로 A) ──
    // MPC에서 나온 목표 CoM + 발 궤적 → IK → PD 토크
    Eigen::VectorXd computeIK(const pinocchio::Model& model,
                               pinocchio::Data& data,
                               const Eigen::VectorXd& q,
                               const Eigen::VectorXd& dq,
                               int rf_frame_id,
                               int lf_frame_id,
                               const Eigen::Vector3d& com_des,
                               const Eigen::Vector3d& rf_pos_des,
                               const Eigen::Vector3d& lf_pos_des);

    // ── DBFC 모드 (경로 B) ──
    // Task → 최적 반력 → 전신 토크
    Eigen::VectorXd computeDBFC(const pinocchio::Model& model,
                                 pinocchio::Data& data,
                                 const Eigen::VectorXd& q,
                                 const Eigen::VectorXd& dq,
                                 int rf_frame_id,
                                 int lf_frame_id,
                                 const Eigen::Vector3d& com_des,
                                 const Eigen::Vector3d& com_dot_des,
                                 const Eigen::Vector3d& rf_pos,
                                 const Eigen::Vector3d& lf_pos);

    // 서브모듈 접근 (디버깅/시각화용)
    WholeBodyIK& getIK() { return ik_; }
    const Eigen::VectorXd& getOptimalForce() const { return force_qp_.opt_F_; }

private:
    int nv_, na_, nc_;

    // 서브모듈
    WholeBodyIK              ik_;
    BalanceTask              balance_task_;
    CenterOfMass             com_dynamics_;
    ForceOptimizer           force_qp_;
    WholeBodyTorqueGenerator torque_gen_;
};
