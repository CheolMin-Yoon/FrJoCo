#pragma once
#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include "config.hpp"
#include "dynamics_model/LIPM.hpp"
#include "controller/LIPM_MPC.hpp"
#include "trajectory_planner/zmp_trajectory.hpp"
#include "trajectory_planner/foot_trajectory.hpp"
#include "whole_body_controller/whole_body_ik.hpp"
#include "whole_body_controller/Force_Optimizer.hpp"
#include "whole_body_controller/whole_body_torque.hpp"
#include "constraints/friction_cone.hpp"
#include "constraints/cop_limits.hpp"
#include "whole_body_controller/tasks/balance_task.hpp"
#include "dynamics_model/com_dynamics.hpp"

class G1WalkingController {
public:
    G1WalkingController(const pinocchio::Model& model,
                        Eigen::Vector2d init_com_xy = Eigen::Vector2d(0.035, 0.0),
                        double com_height = COM_HEIGHT,
                        const Eigen::Vector3d& init_rf_pos = Eigen::Vector3d::Zero(),
                        const Eigen::Vector3d& init_lf_pos = Eigen::Vector3d::Zero());

    // MPC 100Hz — 궤적 추종, 상태 보정
    void mpcLoop(const pinocchio::Data& pin_data, double sim_time);

    // WBC 1kHz — IK + ForceOpt + TorqueGen
    Eigen::VectorXd wbcLoop(const pinocchio::Model& model,
                            pinocchio::Data& data,
                            const Eigen::VectorXd& q,
                            const Eigen::VectorXd& dq,
                            int rf_frame_id,
                            int lf_frame_id,
                            double sim_time);

    // Standing balance only — MPC 없이 초기 자세 유지
    Eigen::VectorXd standingLoop(const pinocchio::Model& model,
                                 pinocchio::Data& data,
                                 const Eigen::VectorXd& q,
                                 const Eigen::VectorXd& dq,
                                 int rf_frame_id,
                                 int lf_frame_id);

    // MPC 상태 [pos, vel, acc]
    Eigen::Vector3d x_state_;
    Eigen::Vector3d y_state_;

    // 발 궤적 (WBC_DT 해상도, Nx3)
    Eigen::MatrixXd l_foot_traj_;
    Eigen::MatrixXd r_foot_traj_;
    Eigen::MatrixXd l_foot_vel_;
    Eigen::MatrixXd r_foot_vel_;
    Eigen::MatrixXd l_foot_acc_;
    Eigen::MatrixXd r_foot_acc_;

    // 전체 CoM ref 궤적 (MPC_DT 해상도, Nx2)
    Eigen::MatrixXd com_ref_traj_;

    // 시간 기반 인덱스 접근
    int getMpcIndex(double t) const {
        return std::min(static_cast<int>(t / MPC_DT),
                        zmp_traj_.getWalkSamples() - 1);
    }
    int getWbcIndex(double t) const {
        int total = foot_traj_.totalSamples(N_STEPS);
        return std::min(static_cast<int>(t / WBC_DT), total - 1);
    }

    // 시각화용 접근자
    int getTrajectoryIndex() const { return traj_idx_; }
    const ZmpTrajectory& getZmpTrajectory() const { return zmp_traj_; }
    const std::vector<Eigen::Vector2d>& getFootsteps() const { return footsteps_; }
    const Eigen::MatrixXd& getComRefTraj() const { return com_ref_traj_; }
    WholeBodyIK& getIK() { return ik_; }

    // 타이밍 모니터링 (μs)
    long long getMpcSolveUs()    const { return mpc_solve_us_; }
    long long getForceSolveUs()  const { return force_solve_us_; }
    long long getTorqueGenUs()   const { return torque_gen_us_; }

private:
    void updateState(double u_x, double u_y);

    // 마찰원뿔 제약을 양발 12차원에 맞게 블록 대각 조립
    void buildFrictionBlock(Eigen::MatrixXd& A_fric,
                            Eigen::VectorXd& l_fric,
                            Eigen::VectorXd& u_fric) const;

    // 접촉 상태 판별 (스텝 인덱스 기반)
    Eigen::Vector2d getContactState(double sim_time) const;

    LIPM           lipm_model_;
    LIPM_MPC       mpc_solver_;
    ZmpTrajectory  zmp_traj_;
    FootTrajectory foot_traj_;
    WholeBodyIK    ik_;

    // WBC 모듈
    ForceOptimizer          force_opt_;
    WholeBodyTorqueGenerator torque_gen_;
    FrictionCone            fric_cone_;
    CoPLimits               cop_limits_;
    BalanceTask             balance_task_;
    CenterOfMass            com_dynamics_;

    std::vector<Eigen::Vector2d> footsteps_;

    int traj_idx_;  // MPC_DT 해상도 인덱스
    double total_mass_;  // Pinocchio에서 가져온 로봇 총 질량

    // standing 모드용 초기 목표
    Eigen::Vector3d init_com_;
    Eigen::Vector3d init_rf_pos_;
    Eigen::Vector3d init_lf_pos_;

    // 타이밍 (μs)
    long long mpc_solve_us_   = 0;
    long long force_solve_us_ = 0;
    long long torque_gen_us_  = 0;
};
