#include "constraints/task_space.hpp"

TaskSpace::TaskSpace(int dof)
    : dof_(dof)
{
    A_ = Eigen::MatrixXd::Identity(dof_, dof_);
    l_.resize(dof_);
    u_.resize(dof_);
}

void TaskSpace::update(const Eigen::VectorXd& state)
{
    // TODO: 상태 기반 태스크 공간 제약 업데이트
    (void)state;
}
