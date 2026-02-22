#pragma once
#include <Eigen/Dense>
#include "config.hpp"

// LIPM 이산시간 상태방정식
// 상태: [x, x_dot, x_ddot], 입력: jerk (u)
// x_{k+1} = Ad * x_k + Bd * u_k
// ZMP 출력: p = Cd * x  →  Cd = [1, 0, -z_c/g]

class LIPM {
public:
    LIPM(double z_c = COM_HEIGHT, double dt = MPC_DT, double gravity = GRAVITY);

    // LIPM_MPC에서 사용하는 getter
    Eigen::MatrixXd getA() const { return Ad_; }
    Eigen::MatrixXd getB() const { return Bd_; }
    Eigen::MatrixXd getC() const { return Cd_; }

private:
    double z_c_, g_, omega_;

    Eigen::MatrixXd Ad_; // 3x3
    Eigen::MatrixXd Bd_; // 3x1
    Eigen::MatrixXd Cd_; // 1x3
};
