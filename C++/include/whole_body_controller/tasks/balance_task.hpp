#pragma once

#include "whole_body_controller/tasks/Task_core.hpp"
#include <Eigen/Dense>

// 논문 식 (24): CoM PD 제어 → 목표 가속도 생성
class BalanceTask : public TaskCore {
public:
    BalanceTask(double kp, double kd);

    void update(const Eigen::Vector3d& com_curr,
                const Eigen::Vector3d& com_dot_curr,
                const Eigen::Vector3d& com_des,
                const Eigen::Vector3d& com_dot_des = Eigen::Vector3d::Zero());

    Eigen::VectorXd getTaskCommand() const override;

    void setGains(double kp, double kd) { kp_ = kp; kd_ = kd; }

private:
    double kp_, kd_;
    Eigen::Vector3d ddc_des_;
};
