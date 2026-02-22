#include "whole_body_controller/tasks/balance_task.hpp"

BalanceTask::BalanceTask(double kp, double kd)
    : kp_(kp), kd_(kd)
{
    ddc_des_.setZero();
}

void BalanceTask::update(const Eigen::Vector3d& com_curr,
                         const Eigen::Vector3d& com_dot_curr,
                         const Eigen::Vector3d& com_des,
                         const Eigen::Vector3d& com_dot_des)
{
    // 식 (24): ddc_des = Kp*(c_des - c) + Kd*(dc_des - dc)
    ddc_des_ = kp_ * (com_des - com_curr) + kd_ * (com_dot_des - com_dot_curr);
}

Eigen::VectorXd BalanceTask::getTaskCommand() const
{
    return ddc_des_;
}
