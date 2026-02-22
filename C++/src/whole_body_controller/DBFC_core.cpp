#include "whole_body_controller/DBFC_core.hpp"

WBC::WBC(int nv, int na, int nc)
    : nv_(nv), na_(na), nc_(nc),
      ik_(nv, na),
      balance_task_(100.0, 20.0),   // Kp, Kd for CoM PD
      com_dynamics_(ROBOT_MASS, GRAVITY),
      force_qp_(nc, 18),            // 12 vars, 18 ineq constraints
      torque_gen_(nv, na, nc)
{
}

Eigen::VectorXd WBC::computeIK(const pinocchio::Model& model,
                                 pinocchio::Data& data,
                                 const Eigen::VectorXd& q,
                                 const Eigen::VectorXd& dq,
                                 int rf_frame_id,
                                 int lf_frame_id,
                                 const Eigen::Vector3d& com_des,
                                 const Eigen::Vector3d& rf_pos_des,
                                 const Eigen::Vector3d& lf_pos_des)
{
    // 1. Differential IK → q_des, v_des
    ik_.compute(model, data, q, dq,
                rf_frame_id, lf_frame_id,
                com_des, rf_pos_des, lf_pos_des);

    // 2. PD 토크 (actuated joints, na개)
    return ik_.computePDTorque(q, dq);
}

Eigen::VectorXd WBC::computeDBFC(const pinocchio::Model& model,
                                   pinocchio::Data& data,
                                   const Eigen::VectorXd& q,
                                   const Eigen::VectorXd& dq,
                                   int rf_frame_id,
                                   int lf_frame_id,
                                   const Eigen::Vector3d& com_des,
                                   const Eigen::Vector3d& com_dot_des,
                                   const Eigen::Vector3d& rf_pos,
                                   const Eigen::Vector3d& lf_pos)
{
    // ── 1. Balance Task: 목표 CoM 가속도 계산 ──
    Eigen::Vector3d com_curr = data.com[0];
    Eigen::Vector3d com_dot_curr = data.vcom[0];
    balance_task_.update(com_curr, com_dot_curr, com_des, com_dot_des);
    Eigen::Vector3d ddc_des = balance_task_.getTaskCommand();

    // ── 2. CoM Dynamics: K*F = u 행렬 구성 ──
    Eigen::Vector3d dL = Eigen::Vector3d::Zero();  // 각운동량 변화율 (0 가정)
    com_dynamics_.updateDynamics(com_curr, lf_pos, rf_pos, ddc_des, dL);

    // ── 3. Force Optimizer: 최적 지면 반력 F_hat ──
    Eigen::MatrixXd W_force = 1e-4 * Eigen::MatrixXd::Identity(nc_, nc_);
    force_qp_.updateObjective(com_dynamics_.getK(), com_dynamics_.getU(), W_force);

    // TODO: 제약조건 (마찰콘 + CoP) 업데이트
    // force_qp_.updateConstraints(A_combined, l_combined, u_combined);

    force_qp_.solve();

    // ── 4. Whole Body Torque: F_hat → tau ──
    return torque_gen_.compute(model, data, rf_frame_id, lf_frame_id,
                               q, dq, force_qp_.opt_F_);
}
