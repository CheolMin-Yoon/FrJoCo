#pragma once
// ============================================================
// [보존용] 기존 DCM backward recursion + CoM forward integration
// ZmpTrajectory에서 분리됨. 빌드에 포함하지 않아도 됨.
// ============================================================
#include <Eigen/Dense>
#include <vector>
#include "config.hpp"

class DcmTrajectory {
public:
    DcmTrajectory(
        double z_c            = COM_HEIGHT,
        double g              = GRAVITY,
        double step_time      = STEP_TIME,
        double dsp_time       = DSP_TIME,
        double dt             = WBC_DT,
        double init_dsp_extra = 0.3
    );

    // DCM 궤적 (역방향 계산 → 순방향 생성)
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> computeDcmTrajectory(
        const std::vector<Eigen::Vector2d>& footsteps
    );

    // CoM 궤적 (DCM 적분)
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> computeComTrajectory(
        const Eigen::MatrixXd& ref_dcm,
        Eigen::Vector2d init_com_xy
    );

    // 타이밍 헬퍼
    double stepTimeFor(int i) const;
    double dspTimeFor(int i)  const;
    int    samplesFor(int i)  const;
    int    totalSamples(int n_steps) const;
    int    stepStartIdx(int step)    const;

    double getOmega() const { return omega_; }
    double getDt()    const { return dt_; }

private:
    double z_c_, omega_;
    double step_time_, dsp_time_, ssp_time_;
    double dt_, init_dsp_extra_;
};
