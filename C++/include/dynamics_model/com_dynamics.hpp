#pragma once
#include <Eigen/Dense>
#include "config.hpp"

// Dynamic Balance Force Control for Compliant Humanoid Robots
// Section 2. CoM dynamics
//
// 식 (1): [D1; D2] * f = u
//   D1 = [I  0  I  0]          (3×12) 선운동량
//   D2 = [(r_R×)  I  (r_L×)  I] (3×12) 각운동량
//   f  = [f_R; tau_R; f_L; tau_L] (12×1)
//   u  = [m*ddx_des + F_g; dL]    (6×1)

class CenterOfMass {
public:
    CenterOfMass(double mass = 35.0, double gravity = GRAVITY);

    // com_pos       : CoM 절대 위치 (world frame)
    // l_foot_pos    : 왼발 절대 위치 (world frame)
    // r_foot_pos    : 오른발 절대 위치 (world frame)
    // desired_ddcom : 원하는 CoM 가속도 (world frame)
    // dL            : 각운동량 변화율
    void updateDynamics(const Eigen::Vector3d& com_pos,
                        const Eigen::Vector3d& l_foot_pos,
                        const Eigen::Vector3d& r_foot_pos,
                        const Eigen::Vector3d& desired_ddcom,
                        const Eigen::Vector3d& dL);

    Eigen::MatrixXd getK() const { return K_; }  // 6×12
    Eigen::VectorXd getU() const { return u_; }  // 6×1

private:
    double m_;
    double g_;

    Eigen::MatrixXd D1_; // 3×12
    Eigen::MatrixXd D2_; // 3×12
    Eigen::MatrixXd K_;  // 6×12
    Eigen::VectorXd u_;  // 6×1
};
