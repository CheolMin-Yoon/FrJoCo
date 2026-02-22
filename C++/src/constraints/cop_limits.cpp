// CoP 제약조건 (Center of Pressure)
//
// 결정변수 벡터 (12차원):
//   오른발: [F_Rx, F_Ry, F_Rz, M_Rx, M_Ry, M_Rz]  → 인덱스 0~5
//   왼발:   [F_Lx, F_Ly, F_Lz, M_Lx, M_Ly, M_Lz]  → 인덱스 6~11
//
// CoP 정의:
//   CoP_x = -M_y / F_z  →  dX_min * F_z ≤ -M_y ≤ dX_max * F_z
//   CoP_y =  M_x / F_z  →  dY_min * F_z ≤  M_x ≤ dY_max * F_z
//
// 선형 제약 (F_z 곱해서 정리):
//   dX_min * F_z ≤ -M_y ≤ dX_max * F_z
//   →  -M_y - dX_max * F_z ≤ 0   (상한)
//   →  -M_y - dX_min * F_z ≥ 0   (하한)
//
//   dY_min * F_z ≤  M_x ≤ dY_max * F_z
//   →   M_x - dY_max * F_z ≤ 0   (상한)
//   →   M_x - dY_min * F_z ≥ 0   (하한)

#include "constraints/cop_limits.hpp"
#include <limits>

CoPLimits::CoPLimits(double dX_max, double dX_min, double dY_max, double dY_min)
    : dX_max_(dX_max), dX_min_(dX_min), dY_max_(dY_max), dY_min_(dY_min)
{
    A_.resize(8, 12);
    l_.resize(8);
    u_.resize(8);

    buildConstraint();
}

void CoPLimits::buildConstraint()
{
    A_.setZero();
    double INF = std::numeric_limits<double>::infinity();

    // =======================================================================
    // [1] 오른발 CoP 제약 (F_Rz=col2, M_Rx=col3, M_Ry=col4)
    // =======================================================================
    // 1. M_Rx - dY_max * F_Rz ≤ 0
    A_(0, 3) = 1.0;  A_(0, 2) = -dY_max_;  l_(0) = -INF; u_(0) = 0.0;

    // 2. M_Rx - dY_min * F_Rz ≥ 0
    A_(1, 3) = 1.0;  A_(1, 2) = -dY_min_;  l_(1) = 0.0;  u_(1) = INF;

    // 3. -M_Ry - dX_max * F_Rz ≤ 0
    A_(2, 4) = -1.0; A_(2, 2) = -dX_max_;  l_(2) = -INF; u_(2) = 0.0;

    // 4. -M_Ry - dX_min * F_Rz ≥ 0
    A_(3, 4) = -1.0; A_(3, 2) = -dX_min_;  l_(3) = 0.0;  u_(3) = INF;

    // =======================================================================
    // [2] 왼발 CoP 제약 (F_Lz=col8, M_Lx=col9, M_Ly=col10)
    // =======================================================================
    // 5. M_Lx - dY_max * F_Lz ≤ 0
    A_(4, 9) = 1.0;  A_(4, 8) = -dY_max_;  l_(4) = -INF; u_(4) = 0.0;

    // 6. M_Lx - dY_min * F_Lz ≥ 0
    A_(5, 9) = 1.0;  A_(5, 8) = -dY_min_;  l_(5) = 0.0;  u_(5) = INF;

    // 7. -M_Ly - dX_max * F_Lz ≤ 0
    A_(6, 10) = -1.0; A_(6, 8) = -dX_max_; l_(6) = -INF; u_(6) = 0.0;

    // 8. -M_Ly - dX_min * F_Lz ≥ 0
    A_(7, 10) = -1.0; A_(7, 8) = -dX_min_; l_(7) = 0.0;  u_(7) = INF;
}

void CoPLimits::update(const Eigen::VectorXd& contact_state)
{
    buildConstraint();

    // 스윙 발: 모멘트를 0으로 강제 (수치 안정성)
    if (contact_state(0) < 0.5) {  // 오른발 Swing
        for (int i = 0; i < 4; ++i) { l_(i) = 0.0; u_(i) = 0.0; }
    }
    if (contact_state(1) < 0.5) {  // 왼발 Swing
        for (int i = 4; i < 8; ++i) { l_(i) = 0.0; u_(i) = 0.0; }
    }
}