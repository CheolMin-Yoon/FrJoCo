#include "main_controller/g1_walking_controller.hpp"
#include "config.hpp"
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <limits>
#include <chrono>

G1WalkingController::G1WalkingController(const pinocchio::Model& model,
                                         Eigen::Vector2d init_com_xy, double com_height,
                                         const Eigen::Vector3d& init_rf_pos,
                                         const Eigen::Vector3d& init_lf_pos)
    : lipm_model_(com_height, MPC_DT),
      mpc_solver_(lipm_model_),
      zmp_traj_(com_height, GRAVITY, STEP_TIME, DSP_TIME, MPC_DT),
      foot_traj_(),
      ik_(model.nv, model.nv - 6, WBC_DT),
      force_opt_(12, 24),
      torque_gen_(model.nv, model.nv - 6, 12),
      fric_cone_(FRICTION_MU),
      cop_limits_(0.05, -0.05, 0.02, -0.02),
      balance_task_(BAL_KP, BAL_KD),
      com_dynamics_(pinocchio::computeTotalMass(model), GRAVITY),
      traj_idx_(0),
      total_mass_(pinocchio::computeTotalMass(model))
{
    x_state_.setZero();
    y_state_.setZero();

    // standing 모드용 초기 목표 (FK 기반 실제 위치)
    init_com_ = Eigen::Vector3d(init_com_xy.x(), init_com_xy.y(), com_height);
    if (init_rf_pos.isZero() && init_lf_pos.isZero()) {
        // fallback: 하드코딩
        init_rf_pos_ = Eigen::Vector3d(0.0, G1_INIT_RF_Y, G1_INIT_FOOT_Z);
        init_lf_pos_ = Eigen::Vector3d(0.0, G1_INIT_LF_Y, G1_INIT_FOOT_Z);
    } else {
        init_rf_pos_ = init_rf_pos;
        init_lf_pos_ = init_lf_pos;
    }

    // 1. 발자국 계획
    footsteps_ = zmp_traj_.planFootsteps(N_STEPS, STEP_LENGTH, STEP_WIDTH, init_com_xy);

    // 2. ZMP ref 배열 생성 (MPC_DT 해상도)
    zmp_traj_.generateZmpRef(footsteps_, MPC_HORIZON);

    // 3. 발 궤적 생성 (WBC_DT 해상도, 위치+속도+가속도)
    Eigen::Vector3d init_lf(0.0, G1_INIT_LF_Y, G1_INIT_FOOT_Z);
    Eigen::Vector3d init_rf(0.0, G1_INIT_RF_Y, G1_INIT_FOOT_Z);
    auto ft_result = foot_traj_.computeFull(footsteps_, init_lf, init_rf, STEP_LENGTH);
    l_foot_traj_ = ft_result.left_pos;
    r_foot_traj_ = ft_result.right_pos;
    l_foot_vel_  = ft_result.left_vel;
    r_foot_vel_  = ft_result.right_vel;
    l_foot_acc_  = ft_result.left_acc;
    r_foot_acc_  = ft_result.right_acc;

    // 4. 전체 CoM ref 궤적 (오프라인 MPC forward simulation)
    {
        int walk_samples = zmp_traj_.getWalkSamples();
        com_ref_traj_.resize(walk_samples, 2);

        Eigen::Vector3d x_sim = Eigen::Vector3d::Zero();
        Eigen::Vector3d y_sim = Eigen::Vector3d::Zero();
        x_sim(0) = init_com_xy.x();
        y_sim(0) = init_com_xy.y();

        LIPM_MPC offline_mpc(lipm_model_);
        const Eigen::MatrixXd& A = lipm_model_.getA();
        const Eigen::MatrixXd& B = lipm_model_.getB();

        for (int i = 0; i < walk_samples; ++i) {
            com_ref_traj_(i, 0) = x_sim(0);
            com_ref_traj_(i, 1) = y_sim(0);

            Eigen::VectorXd zmp_rx, zmp_ry;
            zmp_traj_.getZmpRefSlice(i, MPC_HORIZON, zmp_rx, zmp_ry);
            auto [ux, uy] = offline_mpc.solve(x_sim, y_sim, zmp_rx, zmp_ry);

            x_sim = A * x_sim + B.col(0) * ux;
            y_sim = A * y_sim + B.col(0) * uy;
        }
    }
}

// ── MPC 100Hz ──
void G1WalkingController::mpcLoop(const pinocchio::Data& pin_data, double sim_time)
{
    int walk_samples = zmp_traj_.getWalkSamples();
    traj_idx_ = getMpcIndex(sim_time);
    if (traj_idx_ >= walk_samples) return;

    // 실측값으로 상태 보정
    x_state_(0) = pin_data.com[0].x();
    x_state_(1) = pin_data.vcom[0].x();
    y_state_(0) = pin_data.com[0].y();
    y_state_(1) = pin_data.vcom[0].y();

    // CoM 가속도 보정: ddx = omega^2 * (x - p_zmp_ref)
    const auto& zmp_x = zmp_traj_.getZmpRefX();
    const auto& zmp_y = zmp_traj_.getZmpRefY();
    double omega2 = GRAVITY / COM_HEIGHT;
    x_state_(2) = omega2 * (x_state_(0) - zmp_x(traj_idx_));
    y_state_(2) = omega2 * (y_state_(0) - zmp_y(traj_idx_));

    // MPC QP
    Eigen::VectorXd zmp_ref_x, zmp_ref_y;
    zmp_traj_.getZmpRefSlice(traj_idx_, MPC_HORIZON, zmp_ref_x, zmp_ref_y);

    auto t_mpc_start = std::chrono::high_resolution_clock::now();
    auto [u_x, u_y] = mpc_solver_.solve(x_state_, y_state_, zmp_ref_x, zmp_ref_y);
    auto t_mpc_end = std::chrono::high_resolution_clock::now();
    mpc_solve_us_ = std::chrono::duration_cast<std::chrono::microseconds>(t_mpc_end - t_mpc_start).count();

    updateState(u_x, u_y);
}

// ── WBC 1kHz ──
Eigen::VectorXd G1WalkingController::wbcLoop(
    const pinocchio::Model& model,
    pinocchio::Data& data,
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& dq,
    int rf_frame_id,
    int lf_frame_id,
    double sim_time)
{
    // 시간 → WBC 인덱스 (발 궤적 해상도)
    int wbc_idx = getWbcIndex(sim_time);
    int total_wbc = static_cast<int>(l_foot_traj_.rows());
    if (wbc_idx >= total_wbc) wbc_idx = total_wbc - 1;

    // MPC 인덱스 (CoM ref 해상도)
    int mpc_idx = getMpcIndex(sim_time);
    int total_mpc = static_cast<int>(com_ref_traj_.rows());
    if (mpc_idx >= total_mpc) mpc_idx = total_mpc - 1;

    // 목표 CoM (MPC에서 나온 현재 상태 사용)
    Eigen::Vector3d com_des(x_state_(0), y_state_(0), COM_HEIGHT);

    // 목표 발 위치 + feedforward 속도 (WBC_DT 해상도 궤적에서 읽기)
    Eigen::Vector3d rf_pos_des = r_foot_traj_.row(wbc_idx).transpose();
    Eigen::Vector3d lf_pos_des = l_foot_traj_.row(wbc_idx).transpose();
    Eigen::Vector3d rf_vel_ff  = r_foot_vel_.row(wbc_idx).transpose();
    Eigen::Vector3d lf_vel_ff  = l_foot_vel_.row(wbc_idx).transpose();

    // ── (1) Differential IK → q_des, v_des ──
    ik_.compute(model, data, q, dq,
                rf_frame_id, lf_frame_id,
                com_des, rf_pos_des, lf_pos_des,
                rf_vel_ff, lf_vel_ff);

    // τ_fb = PD(Δq, Δv)
    Eigen::VectorXd tau_fb = ik_.computePDTorque(q, dq);

    // ── (2) ForceOptimizer → F_hat (최적 지면 반력) ──
    // 접촉 상태 판별
    Eigen::Vector2d contact = getContactState(sim_time);

    // BalanceTask: CoM PD 보정 (MPC 사이 구간 오차 보상)
    Eigen::Vector3d com_curr = data.com[0];
    Eigen::Vector3d com_dot_curr = data.vcom[0];
    Eigen::Vector3d com_dot_des(x_state_(1), y_state_(1), 0.0);
    balance_task_.update(com_curr, com_dot_curr, com_des, com_dot_des);
    Eigen::Vector3d ddc_pd = balance_task_.getTaskCommand();

    // MPC 피드포워드 + PD 보정 합산
    Eigen::Vector3d ddc_des(x_state_(2) + ddc_pd(0),
                            y_state_(2) + ddc_pd(1),
                            ddc_pd(2));

    // com_dynamics로 K(6×12), u(6×1) 구성 — 모멘트 항 포함
    Eigen::Vector3d rf_pos_curr = data.oMf[rf_frame_id].translation();
    Eigen::Vector3d lf_pos_curr = data.oMf[lf_frame_id].translation();
    Eigen::Vector3d dL = Eigen::Vector3d::Zero();
    com_dynamics_.updateDynamics(com_curr, lf_pos_curr, rf_pos_curr, ddc_des, dL);

    Eigen::MatrixXd W = Eigen::MatrixXd::Identity(12, 12) * FORCE_OPT_REG;
    force_opt_.updateObjective(com_dynamics_.getK(), com_dynamics_.getU(), W);

    // 제약 조건 조립: 마찰원뿔(10행) + CoP(8행) = 18행
    Eigen::MatrixXd A_fric; Eigen::VectorXd l_fric, u_fric;
    buildFrictionBlock(A_fric, l_fric, u_fric);

    cop_limits_.update(contact);
    Eigen::MatrixXd A_cop = cop_limits_.getA();
    Eigen::VectorXd l_cop = cop_limits_.getLowerBound();
    Eigen::VectorXd u_cop = cop_limits_.getUpperBound();

    // 스택: [A_fric; A_cop; A_swing] (최대 24 × 12)
    // 마찰원뿔(10) + CoP(8) + 스윙발 강제 0(6) = 24
    Eigen::MatrixXd A_all = Eigen::MatrixXd::Zero(24, 12);
    Eigen::VectorXd l_all = Eigen::VectorXd::Zero(24);
    Eigen::VectorXd u_all = Eigen::VectorXd::Zero(24);

    A_all.topRows(10)       = A_fric;  l_all.head(10)       = l_fric;  u_all.head(10)       = u_fric;
    A_all.middleRows(10, 8) = A_cop;   l_all.segment(10, 8) = l_cop;   u_all.segment(10, 8) = u_cop;

    // 스윙 발: 6개 변수 전부 0으로 강제 (l=0, u=0 등식 제약)
    // 오른발 스윙 → col 0~5, 왼발 스윙 → col 6~11
    {
        int swing_offset = -1;
        if (contact(0) < 0.5) swing_offset = 0;   // 오른발 스윙
        if (contact(1) < 0.5) swing_offset = 6;   // 왼발 스윙

        if (swing_offset >= 0) {
            for (int j = 0; j < 6; ++j) {
                A_all(18 + j, swing_offset + j) = 1.0;
                l_all(18 + j) = 0.0;
                u_all(18 + j) = 0.0;
            }
        } else {
            // 양발 접촉 → 스윙 제약 비활성 (무한 범위)
            double INF = std::numeric_limits<double>::infinity();
            for (int j = 0; j < 6; ++j) {
                A_all(18 + j, j) = 1.0;  // 아무 열이나 (dummy)
                l_all(18 + j) = -INF;
                u_all(18 + j) = INF;
            }
        }
    }

    force_opt_.updateConstraints(A_all, l_all, u_all);

    auto t_fo_start2 = std::chrono::high_resolution_clock::now();
    force_opt_.solve();
    auto t_fo_end2 = std::chrono::high_resolution_clock::now();
    force_solve_us_ = std::chrono::duration_cast<std::chrono::microseconds>(t_fo_end2 - t_fo_start2).count();

    Eigen::VectorXd F_hat = force_opt_.opt_F_;  // 12×1

    // ── (3) WholeBodyTorqueGenerator → τ_ff ──
    auto t_tg_start2 = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd tau_ff = torque_gen_.compute(
        model, data, rf_frame_id, lf_frame_id, q, dq, F_hat);
    auto t_tg_end2 = std::chrono::high_resolution_clock::now();
    torque_gen_us_ = std::chrono::duration_cast<std::chrono::microseconds>(t_tg_end2 - t_tg_start2).count();

    // ── (4) τ = τ_ff + τ_fb ──
    return tau_ff + tau_fb;
}

// ── 마찰원뿔 블록 대각 조립 ──
// FrictionCone은 (5,3) — 한 발의 [Fx,Fy,Fz]에 대한 제약
// 양발 12차원에 맞게 (10,12) 블록 대각으로 확장
void G1WalkingController::buildFrictionBlock(
    Eigen::MatrixXd& A_fric,
    Eigen::VectorXd& l_fric,
    Eigen::VectorXd& u_fric) const
{
    Eigen::MatrixXd A1 = fric_cone_.getA();   // (5, 3)
    Eigen::VectorXd l1 = fric_cone_.getLowerBound();
    Eigen::VectorXd u1 = fric_cone_.getUpperBound();

    A_fric.setZero(10, 12);
    l_fric.resize(10);
    u_fric.resize(10);

    // 오른발: 결정변수 [0,1,2] = [Fx_R, Fy_R, Fz_R]
    A_fric.block(0, 0, 5, 3) = A1;
    l_fric.head(5) = l1;
    u_fric.head(5) = u1;

    // 왼발: 결정변수 [6,7,8] = [Fx_L, Fy_L, Fz_L]
    A_fric.block(5, 6, 5, 3) = A1;
    l_fric.tail(5) = l1;
    u_fric.tail(5) = u1;
}

// ── 접촉 상태 판별 ──
// 스텝 i가 짝수면 오른발 스윙, 홀수면 왼발 스윙
// DSP 구간이면 양발 접촉
Eigen::Vector2d G1WalkingController::getContactState(double sim_time) const
{
    // 기본: 양발 접촉
    Eigen::Vector2d contact(1.0, 1.0);  // [right, left]

    // 현재 어떤 스텝의 어느 phase인지 계산
    double t_acc = 0.0;
    int n_steps = static_cast<int>(footsteps_.size());
    for (int i = 0; i < n_steps; ++i) {
        double st = foot_traj_.stepTimeFor(i);
        double dsp = foot_traj_.dspTimeFor(i);
        double ssp = st - dsp;

        if (sim_time < t_acc + st) {
            double t_in_step = sim_time - t_acc;
            bool in_ssp = (t_in_step >= dsp) && (i + 1 < n_steps);
            if (in_ssp) {
                bool right_swing = (i % 2 == 0);
                if (right_swing) contact(0) = 0.0;  // 오른발 스윙
                else             contact(1) = 0.0;  // 왼발 스윙
            }
            break;
        }
        t_acc += st;
    }
    return contact;
}

// ── Standing Balance 1kHz ──
Eigen::VectorXd G1WalkingController::standingLoop(
    const pinocchio::Model& model,
    pinocchio::Data& data,
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& dq,
    int rf_frame_id,
    int lf_frame_id)
{
    // ── (1) IK: 초기 CoM + 초기 발 위치 유지 ──
    ik_.compute(model, data, q, dq,
                rf_frame_id, lf_frame_id,
                init_com_, init_rf_pos_, init_lf_pos_,
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    Eigen::VectorXd tau_fb = ik_.computePDTorque(q, dq);

    // ── (2) ForceOptimizer: 양발 접촉, 중력 지지만 ──
    // BalanceTask PD 보정
    Eigen::Vector3d com_curr = data.com[0];
    Eigen::Vector3d com_dot_curr = data.vcom[0];
    balance_task_.update(com_curr, com_dot_curr, init_com_, Eigen::Vector3d::Zero());
    Eigen::Vector3d ddc_pd = balance_task_.getTaskCommand();

    // com_dynamics로 K(6×12), u(6×1) 구성 — 모멘트 항 포함
    Eigen::Vector3d rf_pos_curr = data.oMf[rf_frame_id].translation();
    Eigen::Vector3d lf_pos_curr = data.oMf[lf_frame_id].translation();
    Eigen::Vector3d dL = Eigen::Vector3d::Zero();
    com_dynamics_.updateDynamics(com_curr, lf_pos_curr, rf_pos_curr, ddc_pd, dL);

    Eigen::MatrixXd W = Eigen::MatrixXd::Identity(12, 12) * FORCE_OPT_REG;
    force_opt_.updateObjective(com_dynamics_.getK(), com_dynamics_.getU(), W);

    // 제약: 양발 접촉 (스윙 없음)
    Eigen::Vector2d contact(1.0, 1.0);

    Eigen::MatrixXd A_fric; Eigen::VectorXd l_fric, u_fric;
    buildFrictionBlock(A_fric, l_fric, u_fric);

    cop_limits_.update(contact);
    Eigen::MatrixXd A_cop = cop_limits_.getA();
    Eigen::VectorXd l_cop = cop_limits_.getLowerBound();
    Eigen::VectorXd u_cop = cop_limits_.getUpperBound();

    Eigen::MatrixXd A_all = Eigen::MatrixXd::Zero(24, 12);
    Eigen::VectorXd l_all = Eigen::VectorXd::Zero(24);
    Eigen::VectorXd u_all = Eigen::VectorXd::Zero(24);

    A_all.topRows(10)       = A_fric;  l_all.head(10)       = l_fric;  u_all.head(10)       = u_fric;
    A_all.middleRows(10, 8) = A_cop;   l_all.segment(10, 8) = l_cop;   u_all.segment(10, 8) = u_cop;

    // 양발 접촉 → 스윙 제약 비활성
    double INF = std::numeric_limits<double>::infinity();
    for (int j = 0; j < 6; ++j) {
        A_all(18 + j, j) = 1.0;
        l_all(18 + j) = -INF;
        u_all(18 + j) = INF;
    }

    force_opt_.updateConstraints(A_all, l_all, u_all);

    auto t_fo_start = std::chrono::high_resolution_clock::now();
    force_opt_.solve();
    auto t_fo_end = std::chrono::high_resolution_clock::now();
    force_solve_us_ = std::chrono::duration_cast<std::chrono::microseconds>(t_fo_end - t_fo_start).count();

    Eigen::VectorXd F_hat = force_opt_.opt_F_;

    // ── (3) τ_ff ──
    auto t_tg_start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd tau_ff = torque_gen_.compute(
        model, data, rf_frame_id, lf_frame_id, q, dq, F_hat);
    auto t_tg_end = std::chrono::high_resolution_clock::now();
    torque_gen_us_ = std::chrono::duration_cast<std::chrono::microseconds>(t_tg_end - t_tg_start).count();

    // 디버그 출력 (100스텝마다)
    static int dbg_cnt = 0;
    if (++dbg_cnt % 500 == 1) {
        std::cout << "\n[Standing Debug]\n";
        std::cout << "  com_des: " << init_com_.transpose() << "\n";
        std::cout << "  com_cur: " << data.com[0].transpose() << "\n";
        std::cout << "  ddc_pd : " << ddc_pd.transpose() << "\n";
        std::cout << "  F_hat  : " << F_hat.transpose() << "\n";
        std::cout << "  tau_ff norm: " << tau_ff.norm()
                  << "  tau_fb norm: " << tau_fb.norm() << "\n";
        std::cout << "  tau_ff head5: " << tau_ff.head(5).transpose() << "\n";
        std::cout << "  tau_fb head5: " << tau_fb.head(5).transpose() << "\n";
    }

    return tau_ff + tau_fb;
}

void G1WalkingController::updateState(double u_x, double u_y)
{
    const Eigen::MatrixXd& A = lipm_model_.getA();
    const Eigen::MatrixXd& B = lipm_model_.getB();
    x_state_ = A * x_state_ + B.col(0) * u_x;
    y_state_ = A * y_state_ + B.col(0) * u_y;
}
