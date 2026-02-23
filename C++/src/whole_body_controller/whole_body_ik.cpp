#include "whole_body_controller/whole_body_ik.hpp"
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <cmath>

WholeBodyIK::WholeBodyIK(int nv, int na, double dt)
    : nv_(nv), na_(na), dt_(dt)
{
    // 태스크 차원: CoM(3) + RF(6) + LF(6) = 15
    constexpr int task_dim = 15;

    J_stack_.setZero(task_dim, nv_);
    dx_err_.setZero(task_dim);

    q_des_.setZero(nv_ + 1);  // nq = nv + 1 (quaternion)
    v_des_.setZero(nv_);

    // 기본 PD 게인
    Kp_ = Eigen::VectorXd::Constant(na_, IK_KP);
    Kd_ = Eigen::VectorXd::Constant(na_, IK_KD);
}

void WholeBodyIK::setGains(const Eigen::VectorXd& Kp, const Eigen::VectorXd& Kd)
{
    Kp_ = Kp;
    Kd_ = Kd;
}

Eigen::Vector3d WholeBodyIK::orientationError(const Eigen::Matrix3d& R_des,
                                               const Eigen::Matrix3d& R_curr)
{
    // R_err = R_des * R_curr^T
    Eigen::Matrix3d R_err = R_des * R_curr.transpose();

    // log map: axis-angle → 3D vector
    double cos_angle = (R_err.trace() - 1.0) * 0.5;
    cos_angle = std::clamp(cos_angle, -1.0, 1.0);
    double angle = std::acos(cos_angle);

    if (angle < 1e-8) return Eigen::Vector3d::Zero();

    Eigen::Vector3d axis;
    axis << R_err(2,1) - R_err(1,2),
            R_err(0,2) - R_err(2,0),
            R_err(1,0) - R_err(0,1);
    axis *= angle / (2.0 * std::sin(angle));
    return axis;
}

void WholeBodyIK::compute(const pinocchio::Model& model,
                           pinocchio::Data& data,
                           const Eigen::VectorXd& q_curr,
                           const Eigen::VectorXd& dq_curr,
                           int rf_frame_id,
                           int lf_frame_id,
                           const Eigen::Vector3d& com_des,
                           const Eigen::Vector3d& rf_pos_des,
                           const Eigen::Vector3d& lf_pos_des,
                           const Eigen::Vector3d& rf_vel_ff,
                           const Eigen::Vector3d& lf_vel_ff,
                           const Eigen::Matrix3d& rf_ori_des,
                           const Eigen::Matrix3d& lf_ori_des)
{
    // ── 0. 첫 호출 시 q_des를 현재 q로 초기화 ──
    if (first_call_) {
        q_des_ = q_curr;
        v_des_.setZero();
        first_call_ = false;
    }

    // ── 1. 현재 상태 읽기 ──
    Eigen::Vector3d com_curr = data.com[0];

    const auto& rf_se3 = data.oMf[rf_frame_id];
    const auto& lf_se3 = data.oMf[lf_frame_id];

    Eigen::Vector3d rf_pos_curr = rf_se3.translation();
    Eigen::Vector3d lf_pos_curr = lf_se3.translation();
    Eigen::Matrix3d rf_ori_curr = rf_se3.rotation();
    Eigen::Matrix3d lf_ori_curr = lf_se3.rotation();

    // ── 2. 태스크 오차 계산 ──
    // CoM 위치 오차 (3)
    dx_err_.segment<3>(0) = com_des - com_curr;

    // 오른발 위치 + 자세 오차 (6)
    dx_err_.segment<3>(3) = rf_pos_des - rf_pos_curr;
    dx_err_.segment<3>(6) = orientationError(rf_ori_des, rf_ori_curr);

    // 왼발 위치 + 자세 오차 (6)
    dx_err_.segment<3>(9)  = lf_pos_des - lf_pos_curr;
    dx_err_.segment<3>(12) = orientationError(lf_ori_des, lf_ori_curr);

    // ── 3. 자코비안 스택 ──
    // CoM 자코비안 (3 × nv) — 내부적으로 joint jacobians도 계산
    pinocchio::jacobianCenterOfMass(model, data, q_curr, false);
    Eigen::MatrixXd J_com = data.Jcom;

    // joint jacobians 명시적 계산 (getFrameJacobian 전제조건)
    pinocchio::computeJointJacobians(model, data, q_curr);
    pinocchio::updateFramePlacements(model, data);

    // 오른발 자코비안 (6 × nv)
    Eigen::MatrixXd J_rf = Eigen::MatrixXd::Zero(6, nv_);
    pinocchio::getFrameJacobian(model, data, rf_frame_id,
                                pinocchio::LOCAL_WORLD_ALIGNED, J_rf);

    // 왼발 자코비안 (6 × nv)
    Eigen::MatrixXd J_lf = Eigen::MatrixXd::Zero(6, nv_);
    pinocchio::getFrameJacobian(model, data, lf_frame_id,
                                pinocchio::LOCAL_WORLD_ALIGNED, J_lf);

    J_stack_.block(0,  0, 3, nv_) = J_com;
    J_stack_.block(3,  0, 6, nv_) = J_rf;
    J_stack_.block(9,  0, 6, nv_) = J_lf;

    // ── 4. Damped Pseudo-Inverse ──
    // dq = J^T (J J^T + λ²I)^{-1} dx_err / dt
    constexpr int task_dim = 15;
    Eigen::MatrixXd JJt = J_stack_ * J_stack_.transpose()
                         + damping_ * Eigen::MatrixXd::Identity(task_dim, task_dim);
    Eigen::VectorXd dx_rate = dx_err_ * (IK_TASK_GAIN / dt_);  // 오차를 속도로 변환 (gain 적용)

    // feedforward velocity 추가 (궤적의 해석적 미분값)
    dx_rate.segment<3>(3) += rf_vel_ff;   // 오른발 선속도
    dx_rate.segment<3>(9) += lf_vel_ff;   // 왼발 선속도

    v_des_ = J_stack_.transpose() * JJt.ldlt().solve(dx_rate);

    // v_des 클램핑 (전체, floating base 포함)
    for (int i = 0; i < v_des_.size(); ++i)
        v_des_(i) = std::clamp(v_des_(i), -IK_V_MAX, IK_V_MAX);

    // ── 5. 관절 적분 (q_des ⊕= v_des * dt) ──
    // floating base 위치 (0:3)
    q_des_.head<3>() += v_des_.head<3>() * dt_;

    // floating base 자세 (quaternion, q[3:7], v[3:6])
    Eigen::Vector3d omega_dt = v_des_.segment<3>(3) * dt_;
    double angle = omega_dt.norm();
    Eigen::Quaterniond q_fb(q_des_(6), q_des_(3), q_des_(4), q_des_(5)); // xyzw → Eigen wxyz
    if (angle > 1e-10) {
        Eigen::Vector3d axis = omega_dt / angle;
        Eigen::Quaterniond dq_rot(Eigen::AngleAxisd(angle, axis));
        q_fb = (dq_rot * q_fb).normalized();
    }
    q_des_(3) = q_fb.x(); q_des_(4) = q_fb.y();
    q_des_(5) = q_fb.z(); q_des_(6) = q_fb.w();

    // actuated joints (7:nq)
    q_des_.tail(na_) += v_des_.tail(na_) * dt_;
}

Eigen::VectorXd WholeBodyIK::computePDTorque(const Eigen::VectorXd& q_curr,
                                              const Eigen::VectorXd& v_curr) const
{
    // actuated joints만 (floating base 6 제외)
    // q_curr[7:], v_curr[6:]
    Eigen::VectorXd q_act_err = q_des_.tail(na_) - q_curr.tail(na_);
    Eigen::VectorXd v_act_err = v_des_.tail(na_) - v_curr.tail(na_);

    return Kp_.cwiseProduct(q_act_err) + Kd_.cwiseProduct(v_act_err);
}
