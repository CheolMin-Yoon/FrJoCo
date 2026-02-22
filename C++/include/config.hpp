#pragma once

// ============================================================
// 제어 주기 설정
// ============================================================
constexpr double MJ_HZ          = 1000.0;           // MuJoCo 시뮬레이션 주기
constexpr double MJ_TIMESTEP    = 1.0 / MJ_HZ;
constexpr double WBC_HZ         = 1000.0;
constexpr double MPC_HZ         = 100.0;
constexpr double WBC_DT         = 1.0 / WBC_HZ;
constexpr double MPC_DT         = 1.0 / MPC_HZ;
constexpr int    MPC_DECIMATION = static_cast<int>(MJ_HZ / MPC_HZ);

// ============================================================
// 보행 파라미터
// ============================================================
constexpr double GRAVITY        = 9.81;
constexpr double STEP_TIME      = 0.8;
constexpr double DSP_TIME       = 0.12;
constexpr double STEP_HEIGHT    = 0.06;
constexpr double STEP_LENGTH    = 0.1;
constexpr double STEP_WIDTH     = 0.1185;
constexpr int    N_STEPS        = 20;

// ============================================================
// 로봇 물리
// ============================================================
constexpr double COM_HEIGHT     = 0.69;

// ============================================================
// G1 초기 발 위치
// ============================================================
constexpr double G1_INIT_LF_Y   =  STEP_WIDTH;
constexpr double G1_INIT_RF_Y   = -STEP_WIDTH;
constexpr double G1_INIT_FOOT_Z =  0.0;

// ============================================================
// MPC 파라미터
// ============================================================
constexpr int    MPC_HORIZON    = 160;
constexpr double MPC_ALPHA      = 1e-6;   // jerk 페널티
constexpr double MPC_GAMMA      = 1.0;    // ZMP 추종 페널티

// ============================================================
// WBC 게인
// ============================================================
// IK PD 게인 (actuated joints)
// 감쇠비 1일 때 300 - 35, 200 - 28.5, 100 - 20, 75 - 17.5
constexpr double IK_KP          = 300.0;
constexpr double IK_KD          = 35.0;

// IK damped pseudo-inverse 정규화
constexpr double IK_DAMPING     = 1e-2;

// IK 관절 속도 클램핑 (rad/s)
constexpr double IK_V_MAX       = 30.0;

// BalanceTask CoM PD 게인
constexpr double BAL_KP         = 20.0;
constexpr double BAL_KD         = 9.0;

// ForceOptimizer regularization
constexpr double FORCE_OPT_REG  = 1e-4;

// WholeBodyTorqueGenerator 가중치
constexpr double WBT_W_DDQ      = 1e-6;   // ddq 페널티
constexpr double WBT_W_TAU      = 1e-4;   // tau 페널티

// 마찰 계수
constexpr double FRICTION_MU    = 1.0;
