#pragma once
#include <Eigen/Dense>
#include <vector>
#include "config.hpp"

// ZMP Reference 궤적 생성 (Kajita Preview Control 방식)
//
// 발자국 시퀀스로부터 전체 보행에 대한 ZMP ref 배열을 생성
//   - DSP 구간: 이전 발 → 현재 발로 코사인 보간 (ramp)
//   - SSP 구간: 현재 발 위치에 고정 (flat)
//   - 마지막 발 이후: 마지막 발 좌표 유지
//   - preview horizon 여유분 포함

class ZmpTrajectory {
public:
    ZmpTrajectory(
        double z_c            = COM_HEIGHT,
        double g              = GRAVITY,
        double step_time      = STEP_TIME,
        double dsp_time       = DSP_TIME,
        double dt             = MPC_DT,
        double init_dsp_extra = 0.0
    );

    // 발자국 계획
    std::vector<Eigen::Vector2d> planFootsteps(
        int n_steps          = N_STEPS,
        double step_length   = STEP_LENGTH,
        double step_width    = STEP_WIDTH,
        Eigen::Vector2d init_xy = Eigen::Vector2d(0.035, 0.0)
    );

    // 전체 ZMP ref 배열 생성 (preview horizon 여유분 포함)
    // footsteps: planFootsteps() 결과
    // horizon: MPC preview horizon (여유분 추가용)
    void generateZmpRef(
        const std::vector<Eigen::Vector2d>& footsteps,
        int horizon = MPC_HORIZON
    );

    // 현재 인덱스 기준으로 horizon개의 ZMP ref 슬라이스 반환
    void getZmpRefSlice(
        int current_idx, int horizon,
        Eigen::VectorXd& zmp_ref_x,
        Eigen::VectorXd& zmp_ref_y
    ) const;

    // 전체 ZMP ref 접근
    const Eigen::VectorXd& getZmpRefX() const { return zmp_ref_x_; }
    const Eigen::VectorXd& getZmpRefY() const { return zmp_ref_y_; }
    int getTotalSamples() const { return total_samples_; }
    int getWalkSamples()  const { return walk_samples_; }

    // 타이밍 헬퍼 (FootTrajectory에서도 사용)
    double stepTimeFor(int i) const;
    double dspTimeFor(int i)  const;
    int    samplesFor(int i)  const;
    int    stepStartSample(int i) const;
    int    stepEndSample(int i)   const;

    double getOmega() const { return omega_; }
    double getDt()    const { return dt_; }

private:
    double z_c_, omega_;
    double step_time_, dsp_time_, ssp_time_;
    double dt_, init_dsp_extra_;

    Eigen::VectorXd zmp_ref_x_;   // 전체 ZMP ref (walk + horizon 여유분)
    Eigen::VectorXd zmp_ref_y_;
    int total_samples_;            // zmp_ref 배열 크기
    int walk_samples_;             // 보행 구간만의 샘플 수
    int n_steps_;
};
