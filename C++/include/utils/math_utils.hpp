#pragma once
#include <Eigen/Dense>

namespace frmoco {

// 벡터 v에 대한 skew-symmetric 행렬 (외적 행렬)
// v × w = skew(v) * w
inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d S;
    S <<  0.0,  -v(2),  v(1),
         v(2),   0.0,  -v(0),
        -v(1),  v(0),   0.0;
    return S;
}

} // namespace frmoco
