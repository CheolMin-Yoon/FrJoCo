#pragma once

#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include "config.hpp"

// Differential IK (Resolved Motion Rate Control)
//
// 스택된 태스크:
//   [J_com ]     [dx_com_err ]
//   [J_rf  ] dq = [dx_rf_err  ]
//   [J_lf  ]     [dx_lf_err  ]
//
// dq = J_stack^+ * dx_err  (damped pseudo-inverse)
// q_des = q_curr + dq * dt
//
// 발 자세(orientation)는 항등 회전 유지 (6DoF task)

class WholeBodyIK {
public:
    WholeBodyIK(int nv = 29, int na = 23, double dt = WBC_DT);

    // Differential IK 계산
    // feedforward 속도: 궤적의 해석적 미분값 (Cycloid/Bezier)
    void compute(const pinocchio::Model& model,
                 pinocchio::Data& data,
                 const Eigen::VectorXd& q_curr,
                 const Eigen::VectorXd& dq_curr,
                 int rf_frame_id,
                 int lf_frame_id,
                 const Eigen::Vector3d& com_des,
                 const Eigen::Vector3d& rf_pos_des,
                 const Eigen::Vector3d& lf_pos_des,
                 const Eigen::Vector3d& rf_vel_ff = Eigen::Vector3d::Zero(),
                 const Eigen::Vector3d& lf_vel_ff = Eigen::Vector3d::Zero(),
                 const Eigen::Matrix3d& rf_ori_des = Eigen::Matrix3d::Identity(),
                 const Eigen::Matrix3d& lf_ori_des = Eigen::Matrix3d::Identity());

    // 결과 접근
    const Eigen::VectorXd& getDesiredQ() const { return q_des_; }
    const Eigen::VectorXd& getDesiredV() const { return v_des_; }

    // 내장 PD 토크 계산 (actuated joints만, na개)
    Eigen::VectorXd computePDTorque(const Eigen::VectorXd& q_curr,
                                    const Eigen::VectorXd& v_curr) const;

    // PD 게인 설정
    void setGains(const Eigen::VectorXd& Kp, const Eigen::VectorXd& Kd);

private:
    int nv_, na_;
    double dt_;
    double damping_ = IK_DAMPING;  // damped pseudo-inverse 정규화
    bool first_call_ = true;       // 첫 호출 시 q_des_ 초기화용

    Eigen::VectorXd q_des_;   // nq
    Eigen::VectorXd v_des_;   // nv

    // PD 게인 (na 차원, actuated joints)
    Eigen::VectorXd Kp_;
    Eigen::VectorXd Kd_;

    // 작업 공간 자코비안 스택
    Eigen::MatrixXd J_stack_;  // (3 + 6 + 6) × nv = 15 × nv
    Eigen::VectorXd dx_err_;   // 15 × 1

    // 회전 오차 계산 (log(R_des * R_curr^T))
    static Eigen::Vector3d orientationError(const Eigen::Matrix3d& R_des,
                                            const Eigen::Matrix3d& R_curr);
};
