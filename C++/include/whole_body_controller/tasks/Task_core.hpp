#pragma once
#include <Eigen/Dense>

// WBC Task 기반 클래스
// 각 Task는 목표 가속도(또는 가상 힘)를 반환
class TaskCore {
public:
    virtual ~TaskCore() = default;

    // Task가 계산한 목표 명령 (가속도, 가상 힘 등)
    virtual Eigen::VectorXd getTaskCommand() const = 0;
};
