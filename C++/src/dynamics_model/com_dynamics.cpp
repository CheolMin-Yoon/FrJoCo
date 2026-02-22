#include "dynamics_model/com_dynamics.hpp"
#include "utils/math_utils.hpp"

CenterOfMass::CenterOfMass(double mass, double gravity)
    : m_(mass), g_(gravity)
{
    D1_.resize(3, 12);
    D2_.resize(3, 12);
    K_.resize(6, 12);
    u_.resize(6);

    // 식 (2): D1 = [I  0  I  0]
    D1_.setZero();
    D1_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity(); // f_R
    D1_.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity(); // f_L
}

void CenterOfMass::updateDynamics(const Eigen::Vector3d& com_pos,
                                   const Eigen::Vector3d& l_foot_pos,
                                   const Eigen::Vector3d& r_foot_pos,
                                   const Eigen::Vector3d& desired_ddcom,
                                   const Eigen::Vector3d& dL)
{
    // CoM 기준 상대 위치 (world frame)
    Eigen::Vector3d r_R = r_foot_pos - com_pos;
    Eigen::Vector3d r_L = l_foot_pos - com_pos;

    // 식 (3): D2 = [(r_R×)  I  (r_L×)  I]
    D2_.setZero();
    D2_.block<3, 3>(0, 0) = frmoco::skewSymmetric(r_R);
    D2_.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    D2_.block<3, 3>(0, 6) = frmoco::skewSymmetric(r_L);
    D2_.block<3, 3>(0, 9) = Eigen::Matrix3d::Identity();

    K_.block<3, 12>(0, 0) = D1_;
    K_.block<3, 12>(3, 0) = D2_;

    // 식 (1) 우변: u = [m*ddx_des + F_g; dL]
    const Eigen::Vector3d F_g(0.0, 0.0, m_ * g_);
    u_.head<3>() = m_ * desired_ddcom + F_g;
    u_.tail<3>() = dL;
}
